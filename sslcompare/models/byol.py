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

        # --- loss を構成する cos を先に計算（ログにも使う） ---
        cos1 = F.cosine_similarity(q1, y1.detach(), dim=1).mean()
        cos2 = F.cosine_similarity(q2, y2.detach(), dim=1).mean()
        cos_mean = 0.5 * (cos1 + cos2)

        loss = 2 - 2 * cos_mean  # 表現が一致するほど loss↓（BYOLの典型形）

        # Log metrics
        # --- 崩壊チェック（分散が0に寄ると危ない）---
        with torch.no_grad():
            # BYOLは projection/prediction 後より、projection 出力の分散を見ることが多い
            # ここでは q の分散でも簡易診断として十分
            q_std = q1.std(dim=0).mean()
            y_std = y1.std(dim=0).mean()
            q_norm = q1.norm(dim=1).mean()
            y_norm = y1.norm(dim=1).mean()
        # --- lr（安全に） ---
        lr = None
        if getattr(self, "trainer", None) is not None and getattr(self.trainer, "optimizers", None):
            lr = self.trainer.optimizers[0].param_groups[0].get("lr", None)

        # --- ログ（命名統一） ---
        # loss/cos/std は epoch 比較に使えるので sync_dist=True
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("ssl/cos_sim", cos_mean, on_step=True, on_epoch=True, sync_dist=True)
        self.log("repr/q_std", q_std, on_step=True, on_epoch=True, sync_dist=True)
        self.log("repr/y_std", y_std, on_step=True, on_epoch=True, sync_dist=True)
        # norm は診断用なので好みで（同期してもOK）
        self.log("repr/q_norm", q_norm, on_step=True, on_epoch=True, sync_dist=True)
        self.log("repr/y_norm", y_norm, on_step=True, on_epoch=True, sync_dist=True)

        # これは定数/診断：同期不要（floatのままでOK）
        self.log("ssl/ema_decay", float(self.ema_decay), on_step=False, on_epoch=True, sync_dist=False)
        if lr is not None:
            self.log("train/lr", float(lr), on_step=True, on_epoch=True, sync_dist=False)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update target networks after each training step."""
        self._update_target()
