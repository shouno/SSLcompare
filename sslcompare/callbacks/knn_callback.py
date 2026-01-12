# 学習中の knn による評価

# callbacks/knn_callback.py
from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback

# Reuse your evaluator (created earlier)
from evaluators.knn import KNNConfig, knn_accuracy


@torch.no_grad()
def _collect_embeddings(
    trainer,
    pl_module,
    dataloader,
    *,
    normalize: bool,
    max_samples: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collect (features, labels) from a labeled dataloader.
    - features: (N, D) on CPU
    - labels:   (N,)   on CPU

    Notes:
      - Assumes dataloader yields (x, y) or (x, y, *rest)
      - Uses pl_module.forward_features(x)
      - DDP: gathers across ranks using trainer.strategy.all_gather
    """
    pl_module.eval()

    feats_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    seen = 0
    device = pl_module.device

    for batch in dataloader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise ValueError(
                "Eval dataloader must yield (x, y) or (x, y, ...). "
                f"Got: {type(batch)}"
            )

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        z = pl_module.forward_features(x)  # (B, D) expected
        if z.ndim != 2:
            # For safety: if (B, D, 1, 1) etc.
            z = z.flatten(1)

        if normalize:
            z = F.normalize(z, dim=-1)

        feats_list.append(z.detach())
        labels_list.append(y.detach())

        seen += z.shape[0]
        if max_samples is not None and seen >= max_samples:
            break

    feats = torch.cat(feats_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    if max_samples is not None and feats.shape[0] > max_samples:
        feats = feats[:max_samples]
        labels = labels[:max_samples]

    # DDP gather
    if getattr(trainer, "world_size", 1) > 1:
        feats = trainer.strategy.all_gather(feats).reshape(-1, feats.shape[-1])
        labels = trainer.strategy.all_gather(labels).reshape(-1)

    return feats.cpu(), labels.cpu()


class KNNCallback(Callback):
    """
    kNN evaluator callback.

    Assumptions:
      - pl_module has forward_features(x) -> (B, D)
      - datamodule provides labeled eval loaders:
          * train_eval_dataloader()
          * val_eval_dataloader()
        If not, falls back to train_dataloader()/val_dataloader().

    Logs:
      - eval/knn_acc
      - (optional) eval/knn_cfg_* (as text-ish hyperparams via log_dict)
    """

    def __init__(
        self,
        *,
        every_n_epochs: int = 1,
        cfg: Optional[KNNConfig] = None,
        max_train_samples: Optional[int] = 20000,
        max_val_samples: Optional[int] = None,
        log_config: bool = False,
        log_prefix: str = "eval/knn",
        device: Optional[str] = "cuda",
    ) -> None:
        super().__init__()
        self.every_n_epochs = int(every_n_epochs)
        self.cfg = cfg or KNNConfig()
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.log_config = log_config
        self.log_prefix = log_prefix.rstrip("/")
        self.device = device  # device used by evaluator; None -> use tensors' device

        if self.every_n_epochs <= 0:
            raise ValueError("every_n_epochs must be >= 1")

    def _get_eval_loaders(self, trainer):
        dm = trainer.datamodule
        if dm is None:
            raise RuntimeError("trainer.datamodule is None. Provide a DataModule.")

        # Preferred: explicit eval loaders (labeled)
        if hasattr(dm, "train_eval_dataloader") and hasattr(dm, "val_eval_dataloader"):
            train_loader = dm.train_eval_dataloader()
            val_loader = dm.val_eval_dataloader()
            return train_loader, val_loader

        # Fallback
        if hasattr(dm, "train_dataloader") and hasattr(dm, "val_dataloader"):
            return dm.train_dataloader(), dm.val_dataloader()

        raise RuntimeError(
            "DataModule must implement (train_eval_dataloader, val_eval_dataloader) "
            "or at least (train_dataloader, val_dataloader)."
        )

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        epoch = int(getattr(trainer, "current_epoch", 0))
        if epoch % self.every_n_epochs != 0:
            return

        # Avoid interfering with training mode
        was_training = pl_module.training
        try:
            train_loader, val_loader = self._get_eval_loaders(trainer)

            Xtr, ytr = _collect_embeddings(
                trainer,
                pl_module,
                train_loader,
                normalize=self.cfg.normalize,  # keep consistent with cfg
                max_samples=self.max_train_samples,
            )
            Xva, yva = _collect_embeddings(
                trainer,
                pl_module,
                val_loader,
                normalize=self.cfg.normalize,
                max_samples=self.max_val_samples,
            )

            # Compute on a chosen device if available
            eval_device = None
            if self.device is not None:
                try:
                    eval_device = torch.device(self.device)
                except Exception:
                    eval_device = None

            # Only global zero computes + logs to avoid duplicated heavy work
            if getattr(trainer, "is_global_zero", True):
                acc, _ = knn_accuracy(
                    train_feats=Xtr,
                    train_labels=ytr,
                    query_feats=Xva,
                    query_labels=yva,
                    cfg=self.cfg,
                    device=eval_device,
                )

                pl_module.log(
                    f"{self.log_prefix}_acc",
                    float(acc),
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    rank_zero_only=True,
                )

                if self.log_config:
                    # Log config as individual scalars where possible
                    # (W&B handles these nicely as run config too, but this is model-agnostic)
                    cfg_dict = asdict(self.cfg)
                    # Avoid logging large/irrelevant fields as metrics
                    for k, v in cfg_dict.items():
                        if isinstance(v, (int, float, bool, str)):
                            pl_module.log(
                                f"{self.log_prefix}_cfg/{k}",
                                float(v) if isinstance(v, (int, float, bool)) else 0.0,
                                prog_bar=False,
                                on_step=False,
                                on_epoch=True,
                                logger=True,
                                rank_zero_only=True,
                            )

        finally:
            if was_training:
                pl_module.train()
