from typing import Union, Tuple
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


    def nt_xent(
        self, z1: Tensor, z2: Tensor, return_metrics: bool = False
        ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
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
        mask_val = torch.tensor(-1e4, device=device, dtype=sim.dtype)
        sim_masked = sim.masked_fill(diag, mask_val)

        # positives: (i -> i+N), (i+N -> i)
        labels = torch.arange(2 * n, device=device)
        labels = (labels + n) % (2 * n)

        loss = F.cross_entropy(sim_masked, labels)
        if not return_metrics:
            return loss
        
        # metrics 計算
        with torch.no_grad():
            # pos: i と i+N の類似度（cos/temperature）
            pos = torch.cat([sim.diag(n), sim.diag(-n)], dim=0)  # (2N,)
            pos_sim = (pos * self.temperature).mean()           # 温度除去してcos平均に

            # neg: 対角と正例を除いた平均との差（cos）
            neg = sim.clone()
            neg.fill_diagonal_(0.0)
            neg[torch.arange(n, device=device), torch.arange(n, device=device) + n] = 0.0
            neg[torch.arange(n, device=device) + n, torch.arange(n, device=device)] = 0.0
            denom = (2 * n) * (2 * n - 2)  # self と pos を除くので -2
            neg_sim = (neg.sum() / (denom + 1e-8)) * self.temperature
        return loss, pos_sim, neg_sim

    def training_step(self, batch, batch_idx):
        (x1, x2), _ = batch
        z1 = self.projection(self.encoder(x1))
        z2 = self.projection(self.encoder(x2))
        loss, pos_sim, neg_sim = self.nt_xent(z1, z2, return_metrics=True)

        # LR 取得
        lr = None
        if getattr(self, 'trainer', None) is not None and getattr(self.trainer, 'optimizers', None):
            lr = self.trainer.optimizers[0].param_groups[0].get("lr", None)

        dev = loss.device
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("ssl/pos_sim", pos_sim.to(dev), on_step=True, on_epoch=True, sync_dist=True)
        self.log("ssl/neg_sim", neg_sim.to(dev), on_step=True, on_epoch=True, sync_dist=True)
        # あとは診断用なので sync_dist=False
        self.log("ssl/temperature", float(self.temperature), on_step=True, on_epoch=True, sync_dist=False)
        self.log("train/lr", lr, on_step=True, on_epoch=True, sync_dist=False)

        return loss
