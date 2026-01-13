# sslcompare/evaluators/utils.py
from __future__ import annotations

from typing import Optional, Tuple
from collections.abc import Sequence

import torch
import torch.nn.functional as F


@torch.no_grad()
def collect_embeddings(
    trainer,
    pl_module,
    dataloader,
    *,
    normalize: bool = True,
    max_samples: Optional[int] = None,
    flatten: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect (features, labels) from a labeled dataloader.

    Returns:
      feats_cpu: (N, D) float32 on CPU
      labels_cpu: (N,) int64 on CPU

    Assumptions:
      - dataloader yields (x, y) or (x, y, ...)
      - pl_module.forward_features(x) returns (B, D) (or flattable to it)

    DDP:
      - gathers across ranks via trainer.strategy.all_gather
      - you can call this on all ranks; it returns identical CPU tensors
    """
    pl_module.eval()

    feats_list = []
    labels_list = []
    seen = 0

    device = pl_module.device

    for batch in dataloader:
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise ValueError("Eval dataloader must yield (x, y) or (x, y, ...).")

        x, y = batch[0], batch[1]

        # --- Robustness: handle SSL-style multi-view inputs ---
        # If eval dataset accidentally uses TwoCrops/MultiCrops transform,
        # x can be a list/tuple of tensors. For evaluation, we use the first view.
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                raise ValueError("Received empty list/tuple for x in eval batch.")
            x = x[0]
        # Some datasets may wrap a single tensor in a sequence-like container.
        # Uncomment if you ever see that:
        # if isinstance(x, Sequence) and not torch.is_tensor(x):
        #     x = x[0]
        # ------------------------------------------------------

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        z = pl_module.forward_features(x)
        if flatten and z.ndim > 2:
            z = z.flatten(1)
        if z.ndim != 2:
            raise ValueError(f"Expected 2D features (B, D), got {tuple(z.shape)}")

        if normalize:
            z = F.normalize(z, dim=-1)

        feats_list.append(z.detach())
        labels_list.append(y.detach())

        seen += z.size(0)
        if max_samples is not None and seen >= max_samples:
            break

    feats = torch.cat(feats_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    if max_samples is not None and feats.size(0) > max_samples:
        feats = feats[:max_samples]
        labels = labels[:max_samples]

    if getattr(trainer, "world_size", 1) > 1:
        feats = trainer.strategy.all_gather(feats).reshape(-1, feats.shape[-1])
        labels = trainer.strategy.all_gather(labels).reshape(-1)

    return feats.float().cpu(), labels.long().cpu()
