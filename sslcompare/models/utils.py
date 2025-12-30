import torch
import torch.nn as nn


# SSL 共通の Common building blocks をいれる
#    - Projection header commonly used in SSL methods
#    - Prediction header used in BYOL/SimSiam

# 最終的に重くなったら分離．きっと linear head とかも入れたくなるとは思うけど，スタートはこんな感じで


class ProjectionMLP(nn.Module):
    """Two-layer projection head used by most SSL methods."""

    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PredictionMLP(nn.Module):
    """Small MLP for BYOL/SimSiam predictor network."""

    def __init__(self, dim: int = 2048, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
