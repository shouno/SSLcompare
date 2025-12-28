# sslcompare/datamodules/factory.py
from sslcompare.datamodules.cifar10 import CIFAR10DataModule
from sslcompare.datamodules.stl10 import STL10DataModule
from sslcompare.datamodules.imagenet import ImageNetDataModule


def build_datamodule(dataset: str, data_dir: str, transform, **kwargs):
    d = dataset.lower()
    if d == "cifar10":
        return CIFAR10DataModule(data_dir, transform, **kwargs)
    if d == "stl10":
        return STL10DataModule(
            data_dir, transform, split=kwargs.get("stl10_split", "unlabeled"), **kwargs
        )
    if d == "imagenet":
        return ImageNetDataModule(data_dir, transform, **kwargs)
    raise ValueError(f"Unknown dataset: {dataset}")
