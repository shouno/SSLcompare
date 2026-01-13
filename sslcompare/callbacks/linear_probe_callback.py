# sslcompare/callbacks/linear_probe_callback.py
from __future__ import annotations

from typing import Optional

from lightning.pytorch.callbacks import Callback

from sslcompare.evaluators.utils import collect_embeddings
from sslcompare.evaluators.linear_probe import LinearProbeConfig, fit_linear_probe


class LinearProbeCallback(Callback):
    """
    Periodic linear probe on frozen features (fixed-step SGD).
    """

    def __init__(
        self,
        *,
        every_n_epochs: int = 10,
        cfg: Optional[LinearProbeConfig] = None,
        max_train_samples: Optional[int] = 20000,
        max_val_samples: Optional[int] = None,
        log_name: str = "eval/linear_acc",
    ):
        self.every_n_epochs = int(every_n_epochs)
        self.cfg = cfg or LinearProbeConfig()
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.log_name = log_name

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

        train_loader, val_loader = self._get_loaders(trainer)

        # For linear probe, normalization is optional; here we keep it False by default
        Xtr, ytr = collect_embeddings(
            trainer, pl_module, train_loader,
            normalize=False, max_samples=self.max_train_samples
        )
        Xva, yva = collect_embeddings(
            trainer, pl_module, val_loader,
            normalize=False, max_samples=self.max_val_samples
        )

        if not getattr(trainer, "is_global_zero", True):
            return

        acc, _ = fit_linear_probe(Xtr, ytr, Xva, yva, cfg=self.cfg)
        pl_module.log(self.log_name, float(acc), prog_bar=True, on_step=False, on_epoch=True, logger=True, rank_zero_only=True)
