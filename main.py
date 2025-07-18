import argparse
import pytorch_lightning as pl
from ssl_src.common import CIFAR10DataModule, ImageNetDataModule
from ssl_src.simclr import SimCLRModule
from ssl_src.byol import BYOLModule
from ssl_src.simsiam import SimSiamModule
from ssl_src.barlowtwins import BarlowTwinsModule
from ssl_src.swav import SwAVModule
from ssl_src.mae import MAEModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import os
from datetime import datetime

def cli_main():
    parser = argparse.ArgumentParser(description="SSL Benchmark Trainer")
    parser.add_argument(
        "--method",
        type=str,
        choices=["simclr", "byol", "simsiam", "barlow", "swav", "mae"],
        default="simclr",
        help="SSL method to use",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["imagenet", "cifar10"],
        default="cifar10",
        help="Training dataset picker",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to training data"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=200, help="Maximum number of epochs"
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.2, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument(
        "--base_encoder",
        type=str,
        default="resnet50",
        help="Base encoder architecture (ignored by MAE)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of data loading workers"
    )
    parser.add_argument(
        "--precision", type=str, default="16-mixed", help="Training precision"
    )
    parser.add_argument(
        "--project_name", type=str, default="ssl-benchmark", help="WandB project name"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--save_top_k", type=int, default=3, help="Number of best checkpoints to save"
    )
    parser.add_argument(
        "--save_every_n_epochs", type=int, default=50, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from"
    )

    # Method-specific arguments
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="Temperature for SimCLR/SwAV"
    )
    parser.add_argument(
        "--ema_decay", type=float, default=0.996, help="EMA decay for BYOL"
    )
    parser.add_argument(
        "--lambd", type=float, default=5e-4, help="Lambda for Barlow Twins"
    )
    parser.add_argument(
        "--n_prototypes", type=int, default=3000, help="Number of prototypes for SwAV"
    )
    parser.add_argument(
        "--n_local_crops", type=int, default=6, help="Number of local crops for SwAV"
    )
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for MAE")
    parser.add_argument(
        "--mask_ratio", type=float, default=0.75, help="Masking ratio for MAE"
    )

    args = parser.parse_args()

    # Adjust settings for CIFAR-10
    if args.dataset == "cifar10":
        if args.patch_size == 16:  # デフォルト値なら変更
            args.patch_size = 4
        # CIFAR-10の場合、n_prototypesも調整
        if args.n_prototypes == 3000:
            args.n_prototypes = 300  # CIFAR-10は10クラスなので少なくする

    # Data module setup
    if args.dataset == "imagenet":
        dm = ImageNetDataModule(
            data_dir=args.data_dir,
            method=args.method,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            n_local_crops=args.n_local_crops,  # for only SwAV
        )
    else:
        dm = CIFAR10DataModule(
            data_dir=args.data_dir,
            method=args.method,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            n_local_crops=args.n_local_crops,  # for only SwAV
        )
        
    # Model selection with method-specific parameters
    common_params = {
        "base_encoder": args.base_encoder,  # mae は使わないので渡さないように注意
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }

    if args.method == "simclr":
        model = SimCLRModule(temperature=args.temperature, **common_params)
    elif args.method == "byol":
        model = BYOLModule(ema_decay=args.ema_decay, **common_params)
    elif args.method == "simsiam":
        model = SimSiamModule(**common_params)
    elif args.method == "barlow":
        model = BarlowTwinsModule(lambd=args.lambd, **common_params)
    elif args.method == "swav":
        model = SwAVModule(
            temperature=args.temperature,
            n_prototypes=args.n_prototypes,
            n_local_crops=args.n_local_crops,
            **common_params,
        )
    elif args.method == "mae":
        # MAE has its own LR and WD recommendations
        mae_params = common_params.copy()
        mae_params.pop("base_encoder")  # base_encoder は ViT 固定なので渡さない
        if args.dataset == "cifar10":
            mae_params["lr"] = 1.5e-3  # CIFAR-10用により高い学習率
            mae_params["weight_decay"] = 0.05
            mae_params["img_size"] = 32
            mae_params["patch_size"] = args.patch_size
        else:
            mae_params["lr"] = 1.5e-4
            mae_params["weight_decay"] = 0.05
            mae_params["img_size"] = 224
            mae_params["patch_size"] = args.patch_size
        mae_params["mask_ratio"] = args.mask_ratio
        model = MAEModule(**mae_params)

    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.method}_{args.base_encoder}_{args.dataset}_{timestamp}"
    checkpoint_dir = os.path.join(args.checkpoint_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Callbacks
    callbacks = []
    # 1. 定期的な保存
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="periodic_{epoch:03d}",
        monitor=None,  # 明示しておく
        every_n_epochs=args.save_every_n_epochs,
        # every_n_epochs=5,
        # verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # 2. best とラスト (based on loss)
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-{epoch:03d}-{train_loss:.4f}",
        monitor="train_loss",
        mode="min",
        save_top_k=args.save_top_k,
        save_last=True,  # 最後のエポックも保存
        # verbose=True,
    )
    callbacks.append(best_checkpoint_callback)
    
    # 3. Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)

    # Logger
    wandb_logger = WandbLogger(
        project=args.project_name, 
        name=run_name,
        save_dir=checkpoint_dir,
    )
    
    # Log hyperparameters
    wandb_logger.log_hyperparams(vars(args))
    
    # Trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        precision=args.precision,
        max_epochs=args.max_epochs,
        log_every_n_steps=50,
        default_root_dir=checkpoint_dir,  # ログとチェックポイントの保存先
        enable_checkpointing=True,
        gradient_clip_val=1.0,  # Gradient clipping for stability
    )

    # Train
    trainer.fit(model, dm, ckpt_path=args.resume_from_checkpoint)  # 学習の再開
    
    # Save final model info
    if trainer.global_rank == 0:
        print(f"\nTraining completed!")
        print(f"Checkpoints saved in: {checkpoint_dir}")
    
        # Save path information for easy access
        with open(os.path.join(checkpoint_dir, "checkpoint_info.txt"), "w") as f:
            f.write(f"Run name: {run_name}\n")
            f.write(f"Method: {args.method}\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Base encoder: {args.base_encoder}\n")
            f.write(f"Epochs: {args.max_epochs}\n")


if __name__ == "__main__":
    cli_main()


