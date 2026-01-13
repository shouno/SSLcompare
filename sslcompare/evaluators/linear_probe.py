# sslcompare/evaluators/linear_probe.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class LinearProbeConfig:
    lr: float = 0.1
    weight_decay: float = 0.0
    momentum: float = 0.9
    max_steps: int = 500          # number of SGD steps (fixed compute)
    batch_size: int = 256
    device: str = "cuda"          # "cuda" or "cpu"
    seed: int = 0


@torch.no_grad()
def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def fit_linear_probe(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    val_feats: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    cfg: LinearProbeConfig = LinearProbeConfig(),
    num_classes: Optional[int] = None,
) -> Tuple[float, torch.nn.Module]:
    """
    Fit a linear classifier on frozen features with fixed-step SGD.
    Returns:
      val_acc, trained_classifier
    """
    g = torch.Generator()
    g.manual_seed(cfg.seed)

    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")

    Xtr = train_feats.to(device=device, dtype=torch.float32)
    ytr = train_labels.to(device=device, dtype=torch.long)
    Xva = val_feats.to(device=device, dtype=torch.float32)
    yva = val_labels.to(device=device, dtype=torch.long)

    if num_classes is None:
        num_classes = int(ytr.max().item() + 1)

    D = Xtr.shape[1]
    clf = torch.nn.Linear(D, num_classes).to(device)

    opt = torch.optim.SGD(
        clf.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    clf.train()
    n = Xtr.size(0)
    bs = min(cfg.batch_size, n)

    for _ in range(int(cfg.max_steps)):
        idx = torch.randint(0, n, (bs,), generator=g, device=device)
        xb = Xtr[idx]
        yb = ytr[idx]
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(clf(xb), yb)
        loss.backward()
        opt.step()

    clf.eval()
    with torch.no_grad():
        acc = _accuracy(clf(Xva), yva)

    return acc, clf
