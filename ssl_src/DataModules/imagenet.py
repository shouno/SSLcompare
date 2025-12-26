from pathlib import Path
from torchvision.datasets import ImageFolder
from .base_data import BaseSSLDataModule


# ImageNet 用の DataModule とりあえず，ガワだけ作ってる
class ImageNetDataModule(BaseSSLDataModule):
    def __init__(self, data_dir, transform, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = Path(data_dir)
        self.transform = transform

    def setup(self, stage=None):
        train_dir = self.data_dir / "train"
        if not train_dir.exists():
            raise FileNotFoundError(f"ImageNet expects {train_dir}/<class>/*.jpg")

        self.train_dataset = ImageFolder(
            str(train_dir),
            transform=self.transform,
        )
