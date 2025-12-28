from torchvision.datasets import STL10
from .base import BaseSSLDataModule


class STL10DataModule(BaseSSLDataModule):
    def __init__(self, data_dir, transform, split="unlabeled", download=True, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        self.download = download

    def prepare_data(self):
        # DDP時に複数rankが同時に download して壊れるのを防ぐため、
        # ここで一度だけ download する（Lightningが通常 rank0 のみ実行）
        if self.download:
            STL10(
                root=self.data_dir,
                split=self.split,
                download=True,
            )

    def setup(self, stage=None):
        self.train_dataset = STL10(
            root=self.data_dir,
            split=self.split,
            download=False,
            transform=self.transform,
        )
