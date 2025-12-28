#
# dataset x model の組み合わせを生成する Factory パターンを書く
#

# sslcompare/transforms/factory.py
from __future__ import annotations
from typing import Callable

from .specs import get_specs
from .presets import build_ssl_two_crops, build_swav_multi_crops, build_mae_single_view


# method名で Transform のグルーピングをしておく．
_TWO_CROPS_METHODS = {
    "simclr",
    "byol",
    "simsiam",
    "barlow",
    "barlowtwins",
}  # 好きな名前に
_MAE_METHODS = {"mae"}
_SWAV_METHODS = {"swav"}


def build_transform(dataset: str, method: str, **kwargs) -> Callable:
    spec = get_specs(dataset)
    m = method.lower()

    if m in _TWO_CROPS_METHODS:
        # kwargs を preset に渡したいならここで吸収
        return build_ssl_two_crops(spec, scale=kwargs.get("scale", (0.2, 1.0)))

    if m in _SWAV_METHODS:
        return build_swav_multi_crops(
            spec,
            n_local_crops=kwargs.get("n_local_crops", 6),
            global_scale=kwargs.get("global_scale", (0.14, 1.0)),
            local_scale=kwargs.get("local_scale", (0.05, 0.14)),
        )

    if m in _MAE_METHODS:
        return build_mae_single_view(
            spec, 
            img_size=kwargs.get("img_size", 224), # とりあえずImageNet準拠サイズになおす
            scale=kwargs.get("scale", (0.2, 1.0))
        )

    raise ValueError(f"Unknown method: {method}")
