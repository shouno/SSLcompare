import lightning.pytorch as pl
from torch.utils.data import DataLoader


# データローダの基本クラスを作っておく．これを派生してデータローダを作る．
# とりあえず，train_dataloader のみ実装．
class BaseSSLDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
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
