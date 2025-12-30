from torchvision.datasets import CIFAR10
from .base import BaseSSLDataModule


# CIFAR-10 用の DataModule
class CIFAR10DataModule(BaseSSLDataModule):
    def __init__(self, data_dir, transform, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.transform = transform

    def setup(self, stage=None):
        self.train_dataset = CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.transform,
        )
