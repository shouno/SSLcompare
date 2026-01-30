import lightning.pytorch as pl
from torch.utils.data import DataLoader


# データローダの基本クラスを作っておく．これを派生してデータローダを作る．
# とりあえず，train_dataloader のみ実装．
class BaseSSLDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, eval_batch_size=None):
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def train_eval_dataloader(self):
        """
        SSL 学習時のチェック用途
        """
        # ラベル付き特徴バンク用（shuffleしない/落とさない）
        ds = getattr(self, "train_eval_dataset", None)
        if ds is None:
            # 互換: もし val_dataset 等しか無いなら fallback も可能
            ds = getattr(self, "train_dataset", None)
        if ds is None:
            raise AttributeError("train_eval_dataset (or train_dataset) is not set.")
        return DataLoader(
            ds,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def val_eval_dataloader(self):
        """
        SSL 学習後の評価用（オフライン前提で）
        """
        ds = getattr(self, "val_eval_dataset", None)
        if ds is None:
            ds = getattr(self, "val_dataset", None)
        if ds is None:
            raise AttributeError("val_eval_dataset (or val_dataset) is not set.")
        return DataLoader(
            ds,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
