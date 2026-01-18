# 学習中の knn による評価

# sslcompare/callbacks/knn_callback.py
from __future__ import annotations

from typing import Optional

import torch
from lightning.pytorch.callbacks import Callback

from sslcompare.evaluators.utils import collect_embeddings
from sslcompare.evaluators.knn import KNNConfig, knn_accuracy


class KNNCallback(Callback):
    """
    Periodic kNN evaluation on labeled eval loaders.

    Requires DataModule to implement:
      - train_eval_dataloader()
      - val_eval_dataloader()

    Requires model to implement:
      - forward_features(x) -> (B, D)
    """

    def __init__(
        self,
        *,
        every_n_epochs: int = 5,
        cfg: Optional[KNNConfig] = None,
        max_train_samples: Optional[int] = 20000,
        max_val_samples: Optional[int] = None,
        log_name: str = "eval/knn_acc",
        eval_device: str = "cuda",
    ):
        self.every_n_epochs = int(every_n_epochs)
        self.cfg = cfg or KNNConfig()
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.log_name = log_name
        self.eval_device = eval_device

    def _get_loaders(self, trainer):
        dm = trainer.datamodule
        if dm is None:
            raise RuntimeError("trainer.datamodule is None")

        if not (hasattr(dm, "train_eval_dataloader") and hasattr(dm, "val_eval_dataloader")):
            raise RuntimeError("DataModule must implement train_eval_dataloader() and val_eval_dataloader().")

        return dm.train_eval_dataloader(), dm.val_eval_dataloader()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = int(getattr(trainer, "current_epoch", 0))
        if epoch % self.every_n_epochs != 0:
            return

        # Collect on all ranks (gather inside), compute+log only on global zero
        train_loader, val_loader = self._get_loaders(trainer)

        Xtr, ytr = collect_embeddings(
            trainer, pl_module, train_loader,
            max_samples=self.max_train_samples,
        )
        Xva, yva = collect_embeddings(
            trainer, pl_module, val_loader,
            max_samples=self.max_val_samples,
        )

        if not getattr(trainer, "is_global_zero", True):
            return

        device = torch.device(self.eval_device if (self.eval_device == "cpu" or torch.cuda.is_available()) else "cpu")
        acc, _ = knn_accuracy(Xtr, ytr, Xva, yva, cfg=self.cfg, device=device)

        pl_module.log(self.log_name, float(acc), prog_bar=True, on_step=False, on_epoch=True, logger=True, rank_zero_only=True)
