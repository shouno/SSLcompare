import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSSLModule
from .common import ProjectionMLP, PredictionMLP

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

        loss = (self._loss(p1, z2) + self._loss(p2, z1)) / 2

        # Log metrics with explicit sync_dist for multi-GPU
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", torch.tensor(lr, device=self.device), on_epoch=True, prog_bar=True, sync_dist=True)
    
        # 追加のメトリクス
        self.log("cosine_sim_p1_z2", F.cosine_similarity(p1, z2.detach(), dim=1).mean(), on_epoch=True, sync_dist=True)
        self.log("cosine_sim_p2_z1", F.cosine_similarity(p2, z1.detach(), dim=1).mean(), on_epoch=True, sync_dist=True)
        # Log metrics
        #self.log("train_loss", loss)
        #self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'])

        return loss
