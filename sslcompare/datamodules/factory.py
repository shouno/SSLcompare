from __future__ import annotations
from typing import Any, Dict
import warnings

from .cifar10 import CIFAR10DataModule
from .stl10 import STL10DataModule
from .imagenet import ImageNetDataModule

_ALLOWED_KWARGS: dict[str, set[str]] = { # 取れる引数を決めておく
    "cifar10": {"batch_size", "num_workers"},
    "stl10": {"batch_size", "num_workers", "stl10_split"},
    "imagenet": {"batch_size", "num_workers"},  # 必要なら増やす
}

def _filter_kwargs(dataset: str, kwargs: Dict[str, Any], *, strict: bool = False) -> Dict[str, Any]:
    '''
    許容できるキーワード引数を決めてフィルタリングしておく．例えば stl10_split が cifar10に渡らないように．
    strict が True の場合，未知の引数があれば例外を投げる．
    strict が False の場合，警告だけ出して実行 を続行する．
    '''
    allowed = _ALLOWED_KWARGS.get(dataset)
    if allowed is None:
        raise ValueError(f"Unknown dataset: {dataset}")
    unknown = set(kwargs) - allowed

    filtered = {k: v for k, v in kwargs.items() if k in allowed}

    if unknown:
        # 研究コードとしては「黙って無視」より「落とす」方が健全
        msg = f"Ignored datamodule kwargs for {dataset}: {sorted(unknown)}"
        if strict:
            raise TypeError(msg)
        warnings.warn(msg)
    return filtered


def build_datamodule(dataset: str, data_dir: str, transform, **kwargs):
    '''
    データモジュールの作成．固有の kw 引数はフィルタで落とす．
    '''
    kwargs = _filter_kwargs(dataset, kwargs)

    if dataset == "cifar10":
        return CIFAR10DataModule(data_dir=data_dir, transform=transform, **kwargs)
    if dataset == "stl10":
        return STL10DataModule(data_dir=data_dir, transform=transform, **kwargs)
    if dataset == "imagenet":
        return ImageNetDataModule(data_dir=data_dir, transform=transform, **kwargs)
    raise ValueError(f"Unknown dataset: {dataset}")