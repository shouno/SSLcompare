#
# dataset の mean/std/input_size などの諸元をまとめておく
#

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DatasetSpecs:
    name: str
    input_size: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    jitter_strength: float  # カラージッターの強さを制御
    # SwAVなどで使うことがあるので一緒に持たせてもよい
    local_crop_size: int | None = None


def get_specs(dataset: str) -> DatasetSpecs:
    d = dataset.lower()
    if d == "cifar10":
        return DatasetSpecs(
            name="cifar10",
            input_size=32,
            jitter_strength=1.0,
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
            local_crop_size=16,
        )
    if d == "stl10":
        return DatasetSpecs(
            name="stl10",
            input_size=96,
            jitter_strength=0.5,  # STL-10 は色変換を弱めにする,
            mean=(0.4467, 0.4398, 0.4066),
            std=(0.2241, 0.2215, 0.2239),
            local_crop_size=48,
        )
    if d == "imagenet":
        return DatasetSpecs(
            name="imagenet",
            input_size=224,
            jitter_strength=1.0,
            # ImageNet 標準（多くの実装がこの値を採用）
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            # SwAV の local crop サイズ（ひとまず動く値）
            local_crop_size=96,
        )
    raise ValueError(f"Unknown dataset: {dataset}")
