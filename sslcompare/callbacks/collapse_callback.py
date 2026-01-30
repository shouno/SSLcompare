# sslcompare/callbacks/collapse_callback.py
from __future__ import annotations

from typing import Optional

from lightning.pytorch.callbacks import Callback

from sslcompare.evaluators.utils import collect_embeddings
from sslcompare.evaluators.collapse import CollapseConfig, collapse_metrics


class CollapseCallback(Callback):
    """
    Periodic collapse diagnostics on labeled eval features (or train_eval).

    Logs:
      - eval/collapse_feat_var_mean
      - eval/collapse_feat_std_mean
      - eval/collapse_cov_rank
      - eval/collapse_sv_ratio_minmax
    """

    def __init__(
        self,
        *,
        every_n_epochs: int = 1,
        cfg: Optional[CollapseConfig] = None,
        max_samples: int = 5000,  # keep small (SVD)
        log_prefix: str = "eval/collapse",
    ):
        self.every_n_epochs = int(every_n_epochs)
        self.cfg = cfg or CollapseConfig()
        self.max_samples = int(max_samples)
        self.log_prefix = log_prefix.rstrip("/")

    def _get_loader(self, trainer):
        dm = trainer.datamodule
        if dm is None:
            raise RuntimeError("trainer.datamodule is None")
        if hasattr(dm, "train_eval_dataloader"):
            return dm.train_eval_dataloader()
        raise RuntimeError("DataModule must implement train_eval_dataloader() for collapse metrics.")

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = int(getattr(trainer, "current_epoch", 0))
        if epoch % self.every_n_epochs != 0:
            return

        loader = self._get_loader(trainer)

        X, _ = collect_embeddings(
            trainer, pl_module, loader,
            normalize=self.cfg.normalize,
            max_samples=self.max_samples,
        )

        if not getattr(trainer, "is_global_zero", True):
            return

        m = collapse_metrics(X, cfg=self.cfg)
        for k, v in m.items():
            pl_module.log(f"{self.log_prefix}_{k}", float(v), prog_bar=False, on_step=False, on_epoch=True, logger=True, rank_zero_only=True)
