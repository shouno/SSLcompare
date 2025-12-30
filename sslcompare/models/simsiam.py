import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSSLModule
from .utils import ProjectionMLP, PredictionMLP

###############################################
# 5. SimSiam
###############################################


class SimSiamModule(BaseSSLModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.projection = ProjectionMLP(self.feat_dim)
        self.prediction = PredictionMLP()

    def _loss(self, p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute negative cosine similarity loss."""
        z = z.detach()  # stop-grad
        return -F.cosine_similarity(p, z, dim=1).mean()

    def training_step(self, batch, batch_idx):
        (x1, x2), _ = batch
        z1 = self.projection(self.encoder(x1))
        z2 = self.projection(self.encoder(x2))
        p1 = self.prediction(z1)
        p2 = self.prediction(z2)
        loss_12 = self._loss(p1, z2)
        loss_21 = self._loss(p2, z1)
        loss = 0.5 * (loss_12 + loss_21)

        # cosine similarity（解釈用：loss = -cos の平均）
        with torch.no_grad():
            cos_12 = F.cosine_similarity(p1, z2.detach(), dim=1).mean()
            cos_21 = F.cosine_similarity(p2, z1.detach(), dim=1).mean()
            cos_mean = 0.5 * (cos_12 + cos_21)

            # collapse 診断（分散とノルム）
            z_std = 0.5 * (z1.std(dim=0).mean() + z2.std(dim=0).mean())
            p_std = 0.5 * (p1.std(dim=0).mean() + p2.std(dim=0).mean())
            z_norm = 0.5 * (z1.norm(dim=1).mean() + z2.norm(dim=1).mean())
            p_norm = 0.5 * (p1.norm(dim=1).mean() + p2.norm(dim=1).mean())

        # lr（安全に）
        lr = None
        if getattr(self, "trainer", None) is not None and getattr(self.trainer, "optimizers", None):
            lr = self.trainer.optimizers[0].param_groups[0].get("lr", None)

        # ---- logging（統一命名）----
        # loss/cos/std/norm は epoch 比較に使うので sync_dist=True
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("ssl/cos_sim", cos_mean, on_step=True, on_epoch=True, sync_dist=True)
        self.log("ssl/cos_sim_12", cos_12, on_step=False, on_epoch=True, sync_dist=True)
        self.log("ssl/cos_sim_21", cos_21, on_step=False, on_epoch=True, sync_dist=True)
        self.log("repr/z_std", z_std, on_step=False, on_epoch=True, sync_dist=True)
        self.log("repr/p_std", p_std, on_step=False, on_epoch=True, sync_dist=True)
        self.log("repr/z_norm", z_norm, on_step=False, on_epoch=True, sync_dist=True)
        self.log("repr/p_norm", p_norm, on_step=False, on_epoch=True, sync_dist=True)

        # lr は同期不要、float化して安全に
        if lr is not None:
            self.log("train/lr", float(lr), on_step=True, on_epoch=True, sync_dist=False)

        return loss