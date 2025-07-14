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
        self.train_transform = T.Compose([
            T.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4 * strength, 0.4 *
                          strength, 0.4 * strength, 0.1 * strength)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply(
                [T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, x):
        x1 = self.train_transform(x)
        x2 = self.train_transform(x)
        return (x1, x2)


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 256, num_workers: int = 8,
                 input_size: int = 224, augmentation_strength: float = 1.0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = input_size
        self.augmentation_strength = augmentation_strength

    def setup(self, stage: Optional[str] = None):
        from torchvision.datasets import ImageFolder
        self.dataset = ImageFolder(
            self.data_dir,
            transform=SSLTransform(self.input_size, self.augmentation_strength)
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size: int = 256, num_workers: int = 8):
        super().__init__()
        self.data_dir, self.batch_size, self.num_workers = data_dir, batch_size, num_workers

    def setup(self, stage=None):
        self.ds = CIFAR10(self.data_dir, train=True, download=True,
                          transform=SSLTransform(input_size=32))

    def train_dataloader(self):
        return DataLoader(self.ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          pin_memory=True, drop_last=True)
