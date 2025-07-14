import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSSLModule


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

        # Log metrics
        self.log("train_loss", loss)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'])

        return loss
