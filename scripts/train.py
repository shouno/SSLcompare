# scripts/train.py
from __future__ import annotations

import argparse
import os
from datetime import datetime

from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from sslcompare.transforms.factory import build_transform, build_eval_transform
from sslcompare.models.factory import build_model
from sslcompare.datamodules.factory import build_datamodule

from sslcompare.evaluators.knn import KNNConfig
from sslcompare.callbacks.knn_callback import KNNCallback


def parse_args():
    p = argparse.ArgumentParser()

    # core
    p.add_argument(
        "--method",
        required=True,
        choices=["simclr", "byol", "simsiam", "barlowtwins", "swav", "mae"],
    )
    p.add_argument("--dataset", required=True, choices=["cifar10", "stl10", "imagenet"])
    p.add_argument("--data_dir", required=True)

    # runtime
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--precision", default="16-mixed")
    p.add_argument("--log_every_n_steps", type=int, default=50)

    # base model/optim
    p.add_argument("--base_encoder", default="resnet50")
    p.add_argument(
        "--lr", type=float, default=0.01
    )  # 多分小規模データセットだと 0.2 は大きすぎる
    p.add_argument("--weight_decay", type=float, default=1e-4) # for small datasets
    p.add_argument("--warmup_epochs", type=int, default=2) # for small datasets

    # method knobs
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--ema_decay", type=float, default=0.996)
    p.add_argument("--lambd", type=float, default=5e-4)

    p.add_argument("--n_prototypes", type=int, default=3000)
    p.add_argument("--n_local_crops", type=int, default=6)
    p.add_argument("--sinkhorn_eps", type=float, default=0.05)
    p.add_argument("--sinkhorn_iters", type=int, default=3)

    # MAE knobs
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--mask_ratio", type=float, default=0.75)

    # dataset knobs
    p.add_argument(
        "--stl10_split", default="unlabeled", choices=["unlabeled", "train", "test"]
    )

    # ckpt
    p.add_argument("--checkpoint_root", default="checkpoints")
    p.add_argument("--save_every_n_epochs", type=int, default=20)
    p.add_argument("--save_top_k", type=int, default=3)
    p.add_argument("--resume_from_checkpoint", default=None)

    # wandb
    p.add_argument("--project", default="sslcompare")
    p.add_argument("--run_name", required=True) # ログ管理のため必ず指定させる

    return p.parse_args()


def cli_main():
    args = parse_args()

    run_name = args.run_name
    run_dir = os.path.join(args.checkpoint_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # 1) transform
    transform = build_transform(
        dataset=args.dataset,
        method=args.method,
        n_local_crops=args.n_local_crops,
        img_size=args.img_size,
        # 将来ここに jitter_strength 等を足しても train.py は変えない方針
    )
    eval_transform = build_eval_transform(dataset=args.dataset)

    # 2) datamodule
    dm_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stl10_split=args.stl10_split,
    )

    dm = build_datamodule(
        dataset=args.dataset, data_dir=args.data_dir, 
        transform=transform, eval_transform=eval_transform,
        **dm_kwargs
    )

    # 3) model
    model = build_model(**vars(args))

    # callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=run_dir,
            filename="periodic_{epoch:03d}",
            save_top_k=-1,
            monitor=None,
            every_n_epochs=args.save_every_n_epochs,
        ),
        ModelCheckpoint(
            dirpath=run_dir,
            filename="best-{epoch:03d}",
            monitor="train/loss_epoch",
            mode="min",
            save_top_k=args.save_top_k,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        KNNCallback(every_n_epochs=5, cfg=KNNConfig(k=20)),
    ]

    logger = WandbLogger(project=args.project, name=run_name, save_dir=run_dir)
    logger.log_hyperparams(vars(args))

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        default_root_dir=run_dir,
        logger=logger,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        precision=args.precision,
        log_every_n_steps=args.log_every_n_steps,
        enable_checkpointing=True,  # callbacksにModelCheckpointがあるので True 必須
    )

    trainer.fit(model, dm, ckpt_path=args.resume_from_checkpoint)


if __name__ == "__main__":
    cli_main()
