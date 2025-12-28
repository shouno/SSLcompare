import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSSLModule
from .utils import ProjectionMLP

###############################################
# 3. SimCLR
###############################################


class SimCLRModule(BaseSSLModule):
    def __init__(self, temperature: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.projection = ProjectionMLP(self.feat_dim)
        self.temperature = temperature

    def nt_xent(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """NT-Xent loss computation."""
        batch_size = z1.shape[0]

        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Concatenate z1 and z2
        representations = torch.cat([z1, z2], dim=0)

        # Compute similarity matrix
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )

        # Create labels for positive pairs
        # Positive pairs are (i, i+batch_size) and (i+batch_size, i)
        labels = torch.cat(
            [torch.arange(batch_size) + batch_size, torch.arange(batch_size)], dim=0
        ).to(self.device)

        # Apply temperature
        similarity_matrix = similarity_matrix / self.temperature

        # Compute loss
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss

    def training_step(self, batch, batch_idx):
        (x1, x2), _ = batch
        z1 = self.projection(self.encoder(x1))
        z2 = self.projection(self.encoder(x2))
        loss = self.nt_xent(z1, z2)

        # Log additional metrics
        self.log("train_loss", loss)
        self.log("temperature", self.temperature)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])

        return loss
