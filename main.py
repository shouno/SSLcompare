import argparse
import pytorch_lightning as pl
from ssl_src.common import CIFAR10DataModule
from ssl_src.simclr import SimCLRModule
from ssl_src.byol import BYOLModule
from ssl_src.simsiam import SimSiamModule
from ssl_src.barlowtwins import BarlowTwinsModule
from pytorch_lightning.loggers import WandbLogger


def cli_main():
    parser = argparse.ArgumentParser(description="SSL Benchmark Trainer")
    parser.add_argument("--method", type=str, choices=["simclr", "byol", "simsiam", "barlow"],
                        default="simclr", help="SSL method to use")
    parser.add_argument("--data_dir", type=str,
                        required=True, help="Path to training data")
    parser.add_argument("--dataset", type=str, choices=['imagenet', 'cifar10'],
                        default="imagenet", help="Training dataset picker")
    parser.add_argument("--max_epochs", type=int, default=200,
                        help="Maximum number of epochs")
    parser.add_argument("--batch_size", type=int,
                        default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.2, help="Learning rate")
    parser.add_argument("--weight_decay", type=float,
                        default=1e-6, help="Weight decay")
    parser.add_argument("--base_encoder", type=str,
                        default="resnet50", help="Base encoder architecture")
    parser.add_argument("--temperature", type=float,
                        default=0.2, help="Temperature for SimCLR")
    parser.add_argument("--ema_decay", type=float,
                        default=0.996, help="EMA decay for BYOL")
    parser.add_argument("--lambd", type=float, default=5e-4,
                        help="Lambda for Barlow Twins")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--precision", type=str,
                        default="16-mixed", help="Training precision")
    parser.add_argument("--project_name", type=str,
                        default="ssl-benchmark", help="WandB project name")

    args = parser.parse_args()

    if args.dataset == 'imagenet':
        dm = ImageNetDataModule(
            args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
        )
    else:
        dm = CIFAR10DataModule(
            args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
        )

    # Model selection with method-specific parameters
    common_params = {
        "base_encoder": args.base_encoder,
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

    # Logger
    wandb_logger = WandbLogger(
        project=args.project_name, name=f"{args.method}_{args.base_encoder}")

    # Trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator="auto",
        devices="auto",
        precision=args.precision,
        max_epochs=args.max_epochs,
        gradient_clip_val=1.0,  # Gradient clipping for stability
        log_every_n_steps=50,
        check_val_every_n_epoch=10,  # Validation can be added later
    )

    # Train
    trainer.fit(model, dm)


if __name__ == "__main__":
    cli_main()
