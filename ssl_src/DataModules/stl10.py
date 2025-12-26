from torchvision.datasets import STL10
from .base_data import BaseSSLDataModule


class STL10DataModule(BaseSSLDataModule):
    def __init__(self, data_dir, transform, split="unlabeled", **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.transform = transform
        self.split = split

    def setup(self, stage=None):
        self.train_dataset = STL10(
            root=self.data_dir,
            split=self.split,
            download=True,
            transform=self.transform,
        )
