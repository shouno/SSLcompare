#
# SimCLR/BYOL/SimSiam/BarlowTwins/SwAV 用の標準的なデータ拡張プリセットを作る
# TransformBuilder
#

# sslcompare/transforms/presets.py
from __future__ import annotations

from typing import Callable, List
import torchvision.transforms as T

from .specs import DatasetSpecs


class TwoCropsTransform:
    """SimCLR/BYOL/SimSiam/Barlow など: 2-view を返す"""

    def __init__(self, base_transform: Callable):
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)


class MultiCropsTransform:
    """SwAV: 2 global + N local を返す"""

    def __init__(
        self, global_transform: Callable, local_transform: Callable, n_local_crops: int
    ):
        self.global_transform = global_transform
        self.local_transform = local_transform
        self.n_local_crops = n_local_crops

    def __call__(self, x) -> List:
        crops = [self.global_transform(x), self.global_transform(x)]
        crops.extend([self.local_transform(x) for _ in range(self.n_local_crops)])
        return crops


def build_ssl_two_crops(
    spec: DatasetSpecs, *, scale=(0.2, 1.0), jitter_strength=None
) -> Callable:
    """SimCLR系の一般的なSSL用 2-view"""
    if jitter_strength is None:
        jitter_strength = spec.jitter_strength
    base = T.Compose(
        [
            T.RandomResizedCrop(spec.input_size, scale=scale),
            T.RandomHorizontalFlip(),
            T.RandomApply(
                [
                    T.ColorJitter(
                        0.4 * jitter_strength,
                        0.4 * jitter_strength,
                        0.4 * jitter_strength,
                        0.1 * jitter_strength,
                    )
                ],
                p=0.8,
            ),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(spec.mean, spec.std),
        ]
    )
    return TwoCropsTransform(base)


def build_swav_multi_crops(
    spec: DatasetSpecs,
    *,
    n_local_crops: int = 6,
    global_scale=(0.14, 1.0),
    local_scale=(0.05, 0.14),
    global_jitter=(0.8, 0.8, 0.8, 0.2),
) -> Callable:
    """CIFAR10 SwAV: global(32) + local(16) の multi-crop"""
    if spec.local_crop_size is None:
        raise ValueError(
            f"spec.local_crop_size is required for SwAV (dataset={spec.name})"
        )

    global_tf = T.Compose(
        [
            T.RandomResizedCrop(spec.input_size, scale=global_scale),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(*global_jitter)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(spec.mean, spec.std),
        ]
    )
    local_tf = T.Compose(
        [
            T.RandomResizedCrop(spec.local_crop_size, scale=local_scale),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(*global_jitter)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(spec.mean, spec.std),
        ]
    )
    return MultiCropsTransform(global_tf, local_tf, n_local_crops=n_local_crops)


def build_mae_single_view(spec: DatasetSpecs, *, scale=(0.2, 1.0)) -> Callable:
    """MAE: 色変換を薄くして単一view"""
    return T.Compose(
        [
            T.RandomResizedCrop(spec.input_size, scale=scale),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(spec.mean, spec.std),
        ]
    )
