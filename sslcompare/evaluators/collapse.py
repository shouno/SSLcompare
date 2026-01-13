# sslcompare/evaluators/collapse.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CollapseConfig:
    normalize: bool = True
    eps: float = 1e-6
    rank_tol: float = 1e-3  # relative to max singular value


@torch.no_grad()
def collapse_metrics(
    feats: torch.Tensor,
    *,
    cfg: CollapseConfig = CollapseConfig(),
) -> Dict[str, float]:
    """
    Compute simple collapse diagnostics from features.

    feats: (N, D) on CPU or GPU.

    Returns dict with:
      - feat_var_mean: mean variance across dims
      - feat_std_mean: mean std across dims
      - cov_rank: effective rank of covariance (thresholded)
      - sv_ratio_minmax: min(sv)/max(sv) of centered features
    """
    X = feats.float()
    if cfg.normalize:
        X = F.normalize(X, dim=-1, eps=cfg.eps)

    Xc = X - X.mean(dim=0, keepdim=True)          # center
    var = Xc.var(dim=0, unbiased=False)           # (D,)
    std = torch.sqrt(var + cfg.eps)

    # SVD on (N, D) (might be heavy if huge; use subsampling at callback level)
    # singular values of centered features
    try:
        sv = torch.linalg.svdvals(Xc)             # (min(N,D),)
        sv_max = sv.max().clamp_min(cfg.eps)
        rank = (sv / sv_max > cfg.rank_tol).sum().item()
        sv_ratio = (sv.min() / sv_max).item()
    except Exception:
        # fallback (rare)
        rank = float("nan")
        sv_ratio = float("nan")

    return {
        "feat_var_mean": var.mean().item(),
        "feat_std_mean": std.mean().item(),
        "cov_rank": float(rank),
        "sv_ratio_minmax": float(sv_ratio),
    }
