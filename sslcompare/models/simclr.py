import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import BaseSSLModule
from .utils import ProjectionMLP

class SimCLRModule(BaseSSLModule):
    def __init__(self, temperature: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.projection = ProjectionMLP(self.feat_dim)
        self.temperature = temperature


    def nt_xent(self, z1: Tensor, z2: Tensor) -> Tensor:
        """NT-Xent loss (SimCLR). Mask self-similarity on the diagonal."""
        n = z1.shape[0]
        device = z1.device

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        z = torch.cat([z1, z2], dim=0)  # (2N, D)

        # cosine similarity via matmul (faster than pairwise cosine_similarity)
        sim = (z @ z.T) / self.temperature  # (2N, 2N)

        # mask self-contrast (diagonal) to remove trivial matches
        diag = torch.eye(2 * n, device=device, dtype=torch.bool)
        # fp16 では -1e9 が overflow するので dtype に応じた最小値を使う
        neg_inf = torch.finfo(sim.dtype).min
        # ただし min は極端すぎて NaN の原因になる場合があるので、少し手前にしてもOK
        sim = sim.masked_fill(diag, neg_inf)

        # positives: (i -> i+N), (i+N -> i)
        labels = torch.arange(2 * n, device=device)
        labels = (labels + n) % (2 * n)

        return F.cross_entropy(sim, labels)

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
