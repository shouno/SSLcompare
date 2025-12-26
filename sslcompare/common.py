import torch
import torch.nn as nn
import torchvision.transforms as T
from typing import Optional
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import pytorch_lightning as pl

###############################################
# Common building blocks
#    - Projection header
#    - Prediction header
###############################################


class ProjectionMLP(nn.Module):
    """Two-layer projection head used by most SSL methods."""

    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PredictionMLP(nn.Module):
    """Small MLP for BYOL/SimSiam predictor network."""

    def __init__(self, dim: int = 2048, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


###############################################
# DataModule & Augmentations
###############################################


class SSLTransform:
    """Standard augmentations used by SimCLR-style methods."""

    def __init__(self, input_size: int = 224, strength: float = 1.0):
        self.strength = strength
        self.train_transform = T.Compose(
            [
                T.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomApply(
                    [
                        T.ColorJitter(
                            0.4 * strength,
                            0.4 * strength,
                            0.4 * strength,
                            0.1 * strength,
                        )
                    ],
                    p=0.8,
                ),
                T.RandomGrayscale(p=0.2),
                T.RandomApply(
                    [T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5
                ),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, x):
        x1 = self.train_transform(x)
        x2 = self.train_transform(x)
        return (x1, x2)


class SwAVTransform:
    def __init__(self, input_size: int = 224, n_local_crops: int = 6):
        self.n_local_crops = n_local_crops
        # Two global crops
        self.global_transform1 = T.Compose(
            [
                T.RandomResizedCrop(input_size, scale=(0.14, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        # Several local crops
        self.local_transform = T.Compose(
            [
                T.RandomResizedCrop(96, scale=(0.05, 0.14)),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, x):
        crops = [self.global_transform1(x), self.global_transform1(x)]
        crops.extend([self.local_transform(x) for _ in range(self.n_local_crops)])
        return crops


class MAETransform:
    def __init__(self, input_size: int = 224):
        self.transform = T.Compose(
            [
                T.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, x):
        return self.transform(x)  # Returns a single image


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        method: str,
        batch_size: int = 256,
        num_workers: int = 8,
        input_size: int = 224,
        augmentation_strength: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.data_dir = data_dir
        self.method = method
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = input_size
        self.augmentation_strength = augmentation_strength
        self.transform_kwargs = kwargs

    def setup(self, stage: Optional[str] = None):
        from torchvision.datasets import ImageFolder

        # Select transform based on the method
        if self.method == "swav":
            transform = SwAVTransform(
                self.input_size, self.transform_kwargs.get("n_local_crops", 6)
            )
        elif self.method == "mae":
            transform = MAETransform(self.input_size)
        else:  # Default for SimCLR, BYOL, etc.
            transform = SSLTransform(self.input_size)
        self.dataset = ImageFolder(self.data_dir, transform=transform)
    
    def train_dataloader(self): # [TODO] チェック要件
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        method: str,
        batch_size: int = 256,
        num_workers: int = 8,
        **kwargs
    ):
        super().__init__()
        self.data_dir = data_dir
        self.method = method
        self.batch_size = batch_size
        self.num_workers = num_workers

        # CIFAR-10 の画像サイズ
        input_size = 32

        # CIFAR-10用の正規化パラメータ
        self.cifar10_mean = (0.4914, 0.4822, 0.4465)
        self.cifar10_std = (0.2470, 0.2435, 0.2616)

        if self.method == "swav":
            self.transform = SwAVTransformCIFAR10(
                mean=self.cifar10_mean,
                std=self.cifar10_std,
                n_local_crops=kwargs.get("n_local_crops", 6),
            )
        elif self.method == "mae":
            self.transform = MAETransformCIFAR10(
                input_size=input_size, mean=self.cifar10_mean, std=self.cifar10_std
            )
        else:  # SimCLR, BYOL, BarlowTwins
            self.transform = SSLTransformCIFAR10(
                input_size=input_size, mean=self.cifar10_mean, std=self.cifar10_std
            )

    def setup(self, stage: Optional[str] = None):
        # torchvision から CIFAR10 をダウンロード読み込み
        self.dataset = CIFAR10(
            self.data_dir, train=True, transform=self.transform, download=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )


class SSLTransformCIFAR10:
    def __init__(
        self, input_size=32, mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)
    ):
        self.transform = T.Compose(
            [
                T.RandomResizedCrop(
                    input_size, scale=(0.2, 1.0)
                ),  # スケールはImageNetと同じでも可
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class SwAVTransformCIFAR10:
    def __init__(self, mean, std, n_local_crops=6):
        self.n_local_crops = n_local_crops
        # グローバルビューは32x32
        self.global_transform = T.Compose(
            [
                T.RandomResizedCrop(32, scale=(0.14, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
        # ローカルビューは小さく (例: 16x16)
        self.local_transform = T.Compose(
            [
                T.RandomResizedCrop(16, scale=(0.05, 0.14)),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )

    def __call__(self, x):
        crops = [self.global_transform(x), self.global_transform(x)]
        crops.extend([self.local_transform(x) for _ in range(self.n_local_crops)])
        return crops


class MAETransformCIFAR10:
    def __init__(
        self, input_size=32, mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)
    ):
        self.transform = T.Compose(
            [
                T.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )

    def __call__(self, x):
        return self.transform(x)
