import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.models as models


class LinearEvalModule(pl.LightningModule):
    def __init__(
        self,
        pretrained_checkpoint: str,
        num_classes: int = 10,
        base_encoder: str = "resnet50",
        lr: float = 0.1,
        weight_decay: float = 0.0,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load pretrained encoder
        if "mae" in pretrained_checkpoint.lower():
            # MAE has a different architecture
            self.encoder = self._load_mae_encoder(pretrained_checkpoint)
            self.feat_dim = self.encoder.encoder_norm.normalized_shape[0]
        else:
            # Load standard SSL encoder
            self.encoder = getattr(models, base_encoder)(weights=None)
            self.feat_dim = self._get_feat_dim()
            self.encoder.fc = nn.Identity()

            # Load pretrained weights
            checkpoint = torch.load(pretrained_checkpoint)
            state_dict = checkpoint["state_dict"]

            # Remove prefix if necessary
            encoder_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("encoder."):
                    encoder_state_dict[k.replace("encoder.", "")] = v

            self.encoder.load_state_dict(encoder_state_dict, strict=False)

        # Freeze encoder if specified
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Linear classifier
        self.classifier = nn.Linear(self.feat_dim, num_classes)

        # Metrics
        self.train_acc = pl.metrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = pl.metrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = pl.metrics.Accuracy(task="multiclass", num_classes=num_classes)

    def _get_feat_dim(self) -> int:
        """Get feature dimension dynamically based on encoder."""
        if hasattr(self.encoder, "fc"):
            return self.encoder.fc.in_features
        elif hasattr(self.encoder, "classifier"):
            if isinstance(self.encoder.classifier, nn.Sequential):
                return self.encoder.classifier[0].in_features
            else:
                return self.encoder.classifier.in_features
        else:
            return 2048

    def _load_mae_encoder(self, checkpoint_path):
        """Load MAE encoder from checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint["state_dict"]

        # Extract encoder components from MAE
        from ssl_src.mae import MAEModule

        # Get hyperparameters from checkpoint
        hparams = checkpoint.get("hyper_parameters", {})
        mae = MAEModule(**hparams)

        # Load full model state
        mae.load_state_dict(state_dict)

        # Return only the encoder part
        return mae

    def forward(self, x):
        with torch.no_grad() if self.hparams.freeze_backbone else torch.enable_grad():
            if hasattr(self.encoder, "forward_encoder"):
                # MAE encoder
                features, _, _ = self.encoder.forward_encoder(x, mask_ratio=0.0)
                features = features[:, 0]  # Use CLS token
            else:
                features = self.encoder(x)

        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Metrics
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)

        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Metrics
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y)

        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Metrics
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)

        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        # Only optimize classifier if backbone is frozen
        if self.hparams.freeze_backbone:
            params = self.classifier.parameters()
        else:
            params = self.parameters()

        optimizer = torch.optim.SGD(
            params,
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
        )

        return [optimizer], [scheduler]


class CIFAR10LinearEvalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 256,
        num_workers: int = 8,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # CIFAR-10 normalization
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2470, 0.2435, 0.2616)

        # Simple augmentation for linear eval
        self.train_transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )

        self.val_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = CIFAR10(
                self.data_dir,
                train=True,
                transform=self.train_transform,
                download=True,
            )
            self.val_dataset = CIFAR10(
                self.data_dir,
                train=False,
                transform=self.val_transform,
                download=True,
            )

        if stage == "test" or stage is None:
            self.test_dataset = CIFAR10(
                self.data_dir,
                train=False,
                transform=self.val_transform,
                download=True,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# 評価用スクリプト
if __name__ == "__main__":
    import argparse
    from pytorch_lightning.loggers import WandbLogger

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to pretrained checkpoint"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to CIFAR-10 data"
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--freeze_backbone", action="store_true", help="Freeze encoder weights"
    )

    args = parser.parse_args()

    # Data module
    dm = CIFAR10LinearEvalDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Model
    model = LinearEvalModule(
        pretrained_checkpoint=args.checkpoint,
        num_classes=10,
        lr=args.lr,
        freeze_backbone=args.freeze_backbone,
    )

    # Logger
    logger = WandbLogger(
        project="ssl-linear-eval", name=f"linear_eval_{args.checkpoint.split('/')[-1]}"
    )

    # Trainer
    trainer = pl.Trainer(
        logger=logger,
        accelerator="auto",
        devices="auto",
        max_epochs=args.max_epochs,
        precision="16-mixed",
    )

    # Train and test
    trainer.fit(model, dm)
    trainer.test(model, dm)
