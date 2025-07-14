import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSSLModule

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

        # Standardize per batch (with stability)
        z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-8)
        z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-8)

        # Cross-correlation matrix
        c = (z1.T @ z2) / z1.size(0)

        # Diagonal terms (should be 1)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()

        # Off-diagonal terms (should be 0)
        off_diag = (c.flatten()[1:].view(c.size(0) - 1,
                    c.size(0) + 1)[:, :-1].flatten()).pow_(2).sum()

        loss = on_diag + self.lambd * off_diag

        # Log metrics
        self.log("train_loss", loss)
        self.log("on_diag_loss", on_diag)
        self.log("off_diag_loss", off_diag)
        self.log("lambda", self.lambd)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'])

        return loss
