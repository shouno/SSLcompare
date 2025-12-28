import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseSSLModule
from .utils import ProjectionMLP, PredictionMLP
import torchvision.models as models

###############################################
# BYOL
###############################################


class BYOLModule(BaseSSLModule):
    def __init__(self, ema_decay: float = 0.996, **kwargs):
        super().__init__(**kwargs)
        self.projection = ProjectionMLP(self.feat_dim)
        self.prediction = PredictionMLP()

        # target encoder (EMA)
        self.target_encoder = getattr(models, self.hparams.base_encoder)(weights=None)
        self.target_encoder.fc = nn.Identity()
        self.target_projection = ProjectionMLP(self.feat_dim)
        self.ema_decay = ema_decay

        # Initialize target networks
        self._initialize_target_networks()

        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projection.parameters():
            p.requires_grad = False

    def _initialize_target_networks(self):
        """Initialize target networks with online network weights."""
        for p, q in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            p.data.copy_(q.data)
        for p, q in zip(
            self.target_projection.parameters(), self.projection.parameters()
        ):
            p.data.copy_(q.data)

    @torch.no_grad()
    def _update_target(self):
        """Update target networks with exponential moving average."""
        for p, q in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            p.data = p.data * self.ema_decay + q.data * (1 - self.ema_decay)
        for p, q in zip(
            self.target_projection.parameters(), self.projection.parameters()
        ):
            p.data = p.data * self.ema_decay + q.data * (1 - self.ema_decay)

    def training_step(self, batch, batch_idx):
        (x1, x2), _ = batch

        # Online predictions
        q1 = self.prediction(self.projection(self.encoder(x1)))
        q2 = self.prediction(self.projection(self.encoder(x2)))

        # Target projections
        with torch.no_grad():
            y1 = self.target_projection(self.target_encoder(x2))
            y2 = self.target_projection(self.target_encoder(x1))

        # Compute loss
        loss = (
            2 - 2 * (
                F.cosine_similarity(q1, y1.detach(), dim=1).mean()
                + F.cosine_similarity(q2, y2.detach(), dim=1).mean()
            )
            / 2
        )

        # Log metrics
        self.log("train_loss", loss)
        self.log("ema_decay", self.ema_decay)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update target networks after each training step."""
        self._update_target()
