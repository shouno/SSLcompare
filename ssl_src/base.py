import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl

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
        **kwargs
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

    # Override in subclasses
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        opt = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
        )

        # Warmup + Cosine scheduler
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return epoch / self.warmup_epochs
            else:
                return 0.5 * (
                    1
                    + torch.cos(
                        torch.tensor(
                            (epoch - self.warmup_epochs)
                            / (self.trainer.max_epochs - self.warmup_epochs)
                            * 3.14159
                        )
                    )
                )

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return [opt], [{"scheduler": sched, "interval": "epoch"}]
