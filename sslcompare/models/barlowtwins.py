import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSSLModule
from .utils import ProjectionMLP

###############################################
# 6. Barlow Twins
###############################################


class BarlowTwinsModule(BaseSSLModule):
    def __init__(self, lambd: float = 5e-4, **kwargs):
        super().__init__(**kwargs)
        self.projection = ProjectionMLP(self.feat_dim, 8192, 8192)
        self.lambd = lambd

    def training_step(self, batch, batch_idx):
        (x1, x2), _ = batch
        z1 = self.projection(self.encoder(x1))
        z2 = self.projection(self.encoder(x2))

        # For stability, transform fp32 from fp16
        z1 = z1.float()
        z2 = z2.float()

        # Standardize per batch (more stable than +1e-8 in fp16)
        eps = 1e-4
        z1 = z1 - z1.mean(0)
        z2 = z2 - z2.mean(0)
        z1 = z1 / z1.std(0, unbiased=False).clamp_min(eps)
        z2 = z2 / z2.std(0, unbiased=False).clamp_min(eps)

        # Cross-correlation matrix (fp32)
        c = (z1.T @ z2) / z1.size(0)

        # Diagonal terms (should be 1)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()

        # Off-diagonal terms (should be 0)
        off_diag = (
            (c.flatten()[1:].view(c.size(0) - 1, c.size(0) + 1)[:, :-1].flatten())
            .pow_(2)
            .sum()
        )

        loss = on_diag + self.lambd * off_diag

        # Log metrics
        # ---- metrics（追加しても軽い）----
        with torch.no_grad():
            c_diag = torch.diagonal(c)
            c_diag_mean = c_diag.mean()
            off = (c - torch.diag(c_diag))
            c_offdiag_rms = off.pow(2).mean().sqrt()
        # lr（安全に）
        lr = None
        if getattr(self, "trainer", None) is not None and getattr(self.trainer, "optimizers", None):
            lr = self.trainer.optimizers[0].param_groups[0].get("lr", None)

        # ---- Log metrics（命名統一 + 粒度明示 + DDP方針）----
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # loss 分解（比較・診断の核）
        self.log("ssl/on_diag", on_diag, on_step=True, on_epoch=True, sync_dist=True)
        self.log("ssl/off_diag", off_diag, on_step=True, on_epoch=True, sync_dist=True)

        # 相関行列の状態（議論用）
        self.log("ssl/c_diag_mean", c_diag_mean, on_step=False, on_epoch=True, sync_dist=True)
        self.log("ssl/c_offdiag_rms", c_offdiag_rms, on_step=False, on_epoch=True, sync_dist=True)

        # 定数・診断（同期不要）
        self.log("ssl/lambda", float(self.lambd), on_step=False, on_epoch=True, sync_dist=False)
        if lr is not None:
            self.log("train/lr", float(lr), on_step=True, on_epoch=True, sync_dist=False)
    
        return loss
