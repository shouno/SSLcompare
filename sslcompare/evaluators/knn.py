# sslcompare/evaluators/knn.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F

Metric = Literal["cosine", "dot", "l2"]

# NOTE:
# - normalize=True + metric in {"cosine","dot"} makes them equivalent (cosine sim).
# - Use metric="dot" with normalize=False if you want raw inner-product kNN.

@dataclass(frozen=True)
class KNNConfig:
    k: int = 20
    temperature: float = 0.07     # used for weighted voting
    metric: Metric = "cosine"
    normalize: bool = True        # apply L2-normalization to embeddings
    weighted: bool = True         # softmax-weighted vote vs hard majority
    chunk_size: int = 4096        # query chunk size for memory safety
    eps: float = 1e-12


@torch.no_grad()
def knn_predict(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    query_feats: torch.Tensor,
    *,
    cfg: KNNConfig = KNNConfig(),
    num_classes: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if train_feats.ndim != 2 or query_feats.ndim != 2:
        raise ValueError("train_feats and query_feats must be 2D (N, D)")
    if train_feats.shape[1] != query_feats.shape[1]:
        raise ValueError("Feature dims must match.")
    if train_labels.ndim != 1 or train_labels.shape[0] != train_feats.shape[0]:
        raise ValueError("train_labels must align with train_feats")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr = train_feats.to(device=device, non_blocking=True)
    ytr = train_labels.to(device=device, non_blocking=True)
    Xq = query_feats.to(device=device, non_blocking=True)

    if cfg.normalize:
        Xtr = F.normalize(Xtr, dim=-1, eps=cfg.eps)
        Xq = F.normalize(Xq, dim=-1, eps=cfg.eps)

    if num_classes is None:
        num_classes = int(ytr.max().item() + 1)

    k = int(cfg.k)
    if k <= 0:
        raise ValueError("cfg.k must be >= 1")
    if k > Xtr.shape[0]:
        raise ValueError(f"cfg.k={k} > N_train={Xtr.shape[0]}")

    preds = []

    if cfg.metric == "l2":
        tr_norm2 = (Xtr * Xtr).sum(dim=1)  # (N_train,)

    for start in range(0, Xq.shape[0], cfg.chunk_size):
        q = Xq[start : start + cfg.chunk_size]  # (B, D)

        if cfg.metric in ("cosine", "dot"):
            sim = q @ Xtr.T
            top = sim.topk(k=k, dim=1, largest=True, sorted=False)
            idx, val = top.indices, top.values
            neigh_y = ytr[idx]

            if cfg.weighted:
                w = (val / float(cfg.temperature)).softmax(dim=1)
                votes = torch.zeros((q.size(0), num_classes), device=device, dtype=torch.float32)
                votes.scatter_add_(1, neigh_y, w)
            else:
                votes = torch.zeros((q.size(0), num_classes), device=device, dtype=torch.int32)
                votes.scatter_add_(1, neigh_y, torch.ones_like(neigh_y, dtype=torch.int32))

            preds.append(votes.argmax(dim=1))

        elif cfg.metric == "l2":
            q_norm2 = (q * q).sum(dim=1)  # (B,)
            dot = q @ Xtr.T
            dist2 = q_norm2[:, None] + tr_norm2[None, :] - 2.0 * dot
            top = dist2.topk(k=k, dim=1, largest=False, sorted=False)
            idx = top.indices
            neigh_y = ytr[idx]

            if cfg.weighted:
                val = -top.values
                w = (val / float(cfg.temperature)).softmax(dim=1)
                votes = torch.zeros((q.size(0), num_classes), device=device, dtype=torch.float32)
                votes.scatter_add_(1, neigh_y, w)
            else:
                votes = torch.zeros((q.size(0), num_classes), device=device, dtype=torch.int32)
                votes.scatter_add_(1, neigh_y, torch.ones_like(neigh_y, dtype=torch.int32))
            preds.append(votes.argmax(dim=1))
        else:
            raise ValueError(f"Unknown metric: {cfg.metric}")

    return torch.cat(preds, dim=0).to(train_labels.device)


@torch.no_grad()
def knn_accuracy(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    query_feats: torch.Tensor,
    query_labels: torch.Tensor,
    *,
    cfg: KNNConfig = KNNConfig(),
    num_classes: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[float, torch.Tensor]:
    """
    Returns:
      acc (float), pred_labels (Tensor[N_query])
    """
    pred = knn_predict(
        train_feats=train_feats,
        train_labels=train_labels,
        query_feats=query_feats,
        cfg=cfg,
        num_classes=num_classes,
        device=device,
    )
    y = query_labels.to(pred.device)
    acc = (pred == y).float().mean().item()
    return acc, pred
