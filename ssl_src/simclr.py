import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSSLModule

###############################################
# 3. SimCLR
###############################################


class SimCLRModule(BaseSSLModule):
    def __init__(self, temperature: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.projection = ProjectionMLP(self.feat_dim)
        self.temperature = temperature

    def nt_xent(self, z: torch.Tensor) -> torch.Tensor:
        """NT-Xent loss computation."""
        # z: [2B, D]
        z = F.normalize(z, dim=1)
        sim = torch.exp(z @ z.T / self.temperature)
        mask = (~torch.eye(sim.size(0), dtype=bool, device=self.device)).float()
        sim = sim * mask  # remove self-similarity

        # positives are off-diagonal blocks
        B = z.size(0) // 2
        pos = torch.cat(
            [torch.arange(B, 2 * B), torch.arange(0, B)]).to(self.device)
        pos_sim = torch.diagonal(sim[:, pos])
        loss = -torch.log(pos_sim / (sim.sum(dim=1) + 1e-8)
                          )  # Add epsilon for stability
        return loss.mean()

    def training_step(self, batch, batch_idx):
        (x1, x2), _ = batch
        z1 = self.projection(self.encoder(x1))
        z2 = self.projection(self.encoder(x2))
        loss = self.nt_xent(torch.cat([z1, z2], dim=0))

        # Log additional metrics
        self.log("train_loss", loss)
        self.log("temperature", self.temperature)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'])

        return loss
