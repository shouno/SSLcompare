from __future__ import annotations
from typing import Any, Dict
import warnings

from .cifar10 import CIFAR10DataModule
from .stl10 import STL10DataModule
from .imagenet import ImageNetDataModule


def build_datamodule(dataset: str, data_dir: str, transform, **kwargs):
    '''
    データモジュールの作成．固有の kw 引数はフィルタで落とす．
    '''
    d = dataset.lower()

    # BaseSSLDataModule が受け取れる kwargs を引っ張り出す
    base_kwargs = {k: v for k, v in kwargs.items() if k in {"batch_size", "num_workers"}}

    if dataset == "cifar10":
        return CIFAR10DataModule(data_dir=data_dir, transform=transform, **base_kwargs)
    if dataset == "stl10":
        split = kwargs.get("stl10_split", "unlabeled")
        return STL10DataModule(data_dir=data_dir, transform=transform, split=split, **base_kwargs)
    if dataset == "imagenet":
        return ImageNetDataModule(data_dir=data_dir, transform=transform, **base_kwargs)

    raise ValueError(f"Unknown dataset: {dataset}")