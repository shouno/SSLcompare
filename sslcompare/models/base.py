import torch
import torch.nn as nn
import torchvision.models as models
import lightning.pytorch as pl


###############################################
# Base SSL module with Lightning
###############################################


class BaseSSLModule(pl.LightningModule):
    def __init__(
        self,
        base_encoder: str = "resnet50",
        lr: float = 0.2,
        weight_decay: float = 1e-6,
        warmup_epochs: int = 10,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # backbone (MAE will override this)
        if "vit" not in base_encoder:
            self.encoder = getattr(models, base_encoder)(weights=None)
            self.feat_dim = self._get_feat_dim()
            self.encoder.fc = nn.Identity()

        # For learning rate warmup
        self.warmup_epochs = warmup_epochs

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
            # Default fallback
            return 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        特徴取り出し，forward()と別にしているのは，callback でこちらを利用するため．
        forward() は，学習とか下流タスクにつなげるため，変にいじるよりインターフェースを付け加える
        """
        return self.encoder(x)

    # Override in subclasses
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        # LARS optimizer for better performance with large batch sizes
        # (SimCLR, BYOL, SwAV benefit from LARS)
        if hasattr(self, "use_lars") and self.use_lars:
            from torch.optim import SGD

            # LARS wrapper would go here if available
            # For now, use SGD with adjusted LR
            base_lr = self.hparams.lr
            batch_size = (
                self.trainer.datamodule.batch_size
                if hasattr(self.trainer, "datamodule")
                else 256
            )
            # Linear scaling rule
            lr = base_lr * batch_size / 256

            opt = SGD(
                self.parameters(),
                lr=lr,
                weight_decay=self.hparams.weight_decay,
                momentum=0.9,
            )
        else:
            opt = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
                momentum=0.9,
            )

        # Cosine annealing with linear warmup
        def lr_lambda(current_epoch):
            # Linear warmup
            if current_epoch < self.warmup_epochs:
                return float(current_epoch) / float(max(1, self.warmup_epochs))
            # Cosine annealing
            progress = float(current_epoch - self.warmup_epochs) / float(
                max(1, self.trainer.max_epochs - self.warmup_epochs)
            )
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
