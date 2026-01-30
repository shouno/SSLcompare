from torchvision.datasets import STL10
from .base import BaseSSLDataModule


class STL10DataModule(BaseSSLDataModule):
    def __init__(
        self,
        data_dir,
        transform,
        stl10_split="unlabeled",
        download=True,
        eval_split_train="train",
        eval_split_val="test",
        eval_transform=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.transform = transform
        self.split = stl10_split
        self.download = download

        self.eval_split_train = eval_split_train
        self.eval_split_val = eval_split_val
        # Important: eval_transform must be deterministic single-view.
        # Do NOT default to `transform` (which may be TwoCrops for SSL).
        self.eval_transform = eval_transform

    def prepare_data(self):
        if self.download:
            STL10(root=self.data_dir, split=self.split, download=True)
            STL10(root=self.data_dir, split=self.eval_split_train, download=True)
            STL10(root=self.data_dir, split=self.eval_split_val, download=True)

    def setup(self, stage=None):
        if self.eval_transform is None:
            raise RuntimeError(
                "eval_transform is None. You must pass a single-view eval_transform "
                "(e.g., Resize/CenterCrop + ToTensor + Normalize) for kNN/linear evaluation."
            )
        # SSL 学習用（unlabeled 等）
        self.train_dataset = STL10(
            root=self.data_dir,
            split=self.split,
            download=False,
            transform=self.transform,
        )

        # kNN/Linear 評価用（ラベル付き）
        self.train_eval_dataset = STL10(
            root=self.data_dir,
            split=self.eval_split_train,
            download=False,
            transform=self.eval_transform,
        )
        self.val_eval_dataset = STL10(
            root=self.data_dir,
            split=self.eval_split_val,
            download=False,
            transform=self.eval_transform,
        )
