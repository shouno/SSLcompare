import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSSLModule
from .utils import ProjectionMLP, PredictionMLP


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# --- MAE Main Module ---


class MAEModule(BaseSSLModule):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        embed_dim=768,
        encoder_depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mask_ratio=0.75,
        **kwargs,
    ):
        # CIFAR-10用の調整
        if img_size == 224 and patch_size == 4:  # CIFAR-10の場合
            img_size = 32
            # より小さいモデルを使用
            embed_dim = 384
            encoder_depth = 6
            num_heads = 6
            decoder_embed_dim = 256
            decoder_depth = 4
            decoder_num_heads = 8

        # Override base_encoder as MAE uses a specific ViT architecture
        super().__init__(base_encoder="vit_mae", **kwargs)
        self.save_hyperparameters()

        # 1. Encoder
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.encoder_blocks = nn.ModuleList(
            [Block(embed_dim, num_heads) for _ in range(encoder_depth)]
        )
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # 2. Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim)
        )
        self.decoder_blocks = nn.ModuleList(
            [Block(decoder_embed_dim, decoder_num_heads) for _ in range(decoder_depth)]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # 3. Prediction Head
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * 3, bias=True)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize patch embedding
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize position embeddings
        torch.nn.init.normal_(self.pos_embed, std=0.02)
        torch.nn.init.normal_(self.decoder_pos_embed, std=0.02)

        # Initialize other parameters
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

    def _patchify(self, imgs):
        """imgs: (B, 3, H, W) -> (B, L, patch_size**2 * 3)"""
        p = self.patch_embed.patch_size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def _unpatchify(self, x):
        """x: (B, L, patch_size**2 * 3) -> (B, 3, H, W)"""
        p = self.patch_embed.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

    def _random_masking(self, x, mask_ratio):
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]  # Add pos embed without CLS token

        # Masking
        x, mask, ids_restore = self._random_masking(x, mask_ratio)

        # Append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Apply Transformer blocks
        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)

        # Append mask tokens
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2])
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # Add pos embed
        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        return self.decoder_pred(x[:, 1:, :])  # remove cls token

    def forward_loss(self, imgs, pred, mask):
        target = self._patchify(imgs)

        # MSE loss
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Loss per patch

        # Only consider masked patches for loss
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def training_step(self, batch, batch_idx):
        # MAE uses a single image, not two views
        imgs, _ = batch

        latent, mask, ids_restore = self.forward_encoder(imgs, self.hparams.mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)

        self.log_dict(
            {
                "train_loss": loss,
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                "mask_ratio": self.hparams.mask_ratio,
            }
        )
        return loss

    def configure_optimizers(self):
        # MAE typically uses AdamW with a different schedule
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.95),
        )
        # Cosine scheduler with warmup
        t_max = max(
            1, self.trainer.max_epochs - int(self.hparams.get("warmup_epochs", 10))
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max)

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "epoch",
                "frequency": 1,
            },
        }
