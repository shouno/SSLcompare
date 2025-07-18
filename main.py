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
import torch
import os
from datetime import datetime


# 詳細なチェックポイント調査コード
class VerboseModelCheckpoint(ModelCheckpoint):
    """非常に詳細なデバッグ情報を出力するModelCheckpoint（修正版）"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"\n🔧 VerboseModelCheckpoint初期化:")
        print(f"  - dirpath: {self.dirpath}")
        print(f"  - filename: {self.filename}")
        print(f"  - monitor: {self.monitor}")
        print(f"  - mode: {self.mode}")
        print(f"  - save_top_k: {self.save_top_k}")
        print(f"  - every_n_epochs: {getattr(self, 'every_n_epochs', None)}")
        print(f"  - save_last: {self.save_last}")
        print(f"  - verbose: {self.verbose}")
        
        # ディレクトリの詳細確認
        if self.dirpath:
            print(f"  - Directory exists: {os.path.exists(self.dirpath)}")
            if os.path.exists(self.dirpath):
                print(f"  - Directory writable: {os.access(self.dirpath, os.W_OK)}")
                print(f"  - Directory permissions: {oct(os.stat(self.dirpath).st_mode)[-3:]}")
    
    def on_train_epoch_end(self, trainer, pl_module):
        """エポック終了時の詳細調査（修正版）"""
        print(f"\n📊 エポック {trainer.current_epoch} 終了時の詳細調査:")
        
        # 1. ログされたメトリクスの確認
        print("  メトリクス情報:")
        logged_metrics = trainer.logged_metrics
        callback_metrics = trainer.callback_metrics
        
        print(f"    - logged_metrics keys: {list(logged_metrics.keys())}")
        print(f"    - callback_metrics keys: {list(callback_metrics.keys())}")
        
        if self.monitor:
            monitor_val_logged = logged_metrics.get(self.monitor)
            monitor_val_callback = callback_metrics.get(self.monitor)
            print(f"    - monitor '{self.monitor}' in logged_metrics: {monitor_val_logged}")
            print(f"    - monitor '{self.monitor}' in callback_metrics: {monitor_val_callback}")
            
            # 詳細な型情報
            if monitor_val_callback is not None:
                print(f"    - monitor value type: {type(monitor_val_callback)}")
                if isinstance(monitor_val_callback, torch.Tensor):
                    print(f"    - monitor value device: {monitor_val_callback.device}")
                    print(f"    - monitor value shape: {monitor_val_callback.shape}")
        
        # 2. 保存条件の詳細チェック
        print("  保存条件チェック:")
        
        # every_n_epochs チェック
        if hasattr(self, 'every_n_epochs') and self.every_n_epochs:
            should_save_periodic = (trainer.current_epoch + 1) % self.every_n_epochs == 0
            print(f"    - 定期保存条件 (every {self.every_n_epochs} epochs): {should_save_periodic}")
        
        # monitor チェック（修正版）
        if self.monitor and self.monitor in callback_metrics:
            current_score = callback_metrics[self.monitor]
            print(f"    - 現在のスコア: {current_score}")
            print(f"    - 過去のベストスコア: {self.best_model_score}")
            print(f"    - ベストスコアの型: {type(self.best_model_score)}")
            
            # None チェックを追加
            if self.best_model_score is not None:
                # テンソルの場合は.item()で値を取得
                if isinstance(current_score, torch.Tensor):
                    current_val = current_score.item()
                else:
                    current_val = current_score
                
                if isinstance(self.best_model_score, torch.Tensor):
                    best_val = self.best_model_score.item()
                else:
                    best_val = self.best_model_score
                
                if self.mode == "min":
                    is_better = current_val < best_val
                else:
                    is_better = current_val > best_val
                print(f"    - スコア改善: {is_better}")
            else:
                print(f"    - 初回エポック（ベストスコア未設定）")
        
        # save_last チェック
        is_last_epoch = trainer.current_epoch == trainer.max_epochs - 1
        print(f"    - 最終エポック: {is_last_epoch}")
        print(f"    - save_last設定: {self.save_last}")
        
        # 3. 実際の親クラスの処理を実行
        print("  親クラスの処理実行中...")
        try:
            result = super().on_train_epoch_end(trainer, pl_module)
            print("  ✅ 親クラスの処理完了")
        except Exception as e:
            print(f"  ❌ 親クラスの処理でエラー: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # 4. 処理後の状態確認
        print("  処理後の状態:")
        print(f"    - best_model_path: {self.best_model_path}")
        print(f"    - last_model_path: {self.last_model_path}")
        print(f"    - best_model_score: {self.best_model_score}")
        
        # 5. 実際にファイルが作成されているかチェック
        if self.dirpath and os.path.exists(self.dirpath):
            files = os.listdir(self.dirpath)
            ckpt_files = [f for f in files if f.endswith('.ckpt')]
            print(f"    - ディレクトリ内のファイル数: {len(files)}")
            print(f"    - .ckptファイル数: {len(ckpt_files)}")
            if ckpt_files:
                print(f"    - .ckptファイル: {ckpt_files}")
                # 最新ファイルのサイズも確認
                for ckpt in ckpt_files:
                    filepath = os.path.join(self.dirpath, ckpt)
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    print(f"      - {ckpt}: {size_mb:.2f} MB")
        
        return result
    
    def _save_checkpoint(self, trainer, filepath):
        """実際のファイル保存処理の詳細調査（修正版）"""
        print(f"\n💾 チェックポイント保存試行:")
        print(f"  - ファイルパス: {filepath}")
        print(f"  - ディレクトリ存在: {os.path.exists(os.path.dirname(filepath))}")
        
        if os.path.exists(os.path.dirname(filepath)):
            print(f"  - ディレクトリ書き込み可能: {os.access(os.path.dirname(filepath), os.W_OK)}")
        
        # ディスク容量チェック
        try:
            import shutil
            disk_usage = shutil.disk_usage(os.path.dirname(filepath))
            free_gb = disk_usage.free / (1024**3)
            print(f"  - 空きディスク容量: {free_gb:.2f} GB")
        except Exception as e:
            print(f"  - ディスク容量チェックエラー: {e}")
        
        # 一時ファイルでの書き込みテスト
        temp_file = os.path.join(os.path.dirname(filepath), "write_test.tmp")
        try:
            with open(temp_file, 'w') as f:
                f.write("test")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            print(f"  - 書き込みテスト: ✅ 成功")
        except Exception as e:
            print(f"  - 書き込みテスト: ❌ 失敗 - {e}")
            return
        
        # モデル情報
        try:
            state_dict = trainer.lightning_module.state_dict()
            print(f"  - モデル状態辞書のキー数: {len(state_dict)}")
            print(f"  - オプティマイザ数: {len(trainer.optimizers)}")
            
            # メモリ使用量の概算
            total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
            print(f"  - 総パラメータ数: {total_params:,}")
        except Exception as e:
            print(f"  - モデル情報取得エラー: {e}")
        
        # 実際の保存処理
        try:
            print(f"  - 保存処理開始...")
            result = super()._save_checkpoint(trainer, filepath)
            print(f"  - 保存処理完了")
            
            # 保存後の確認
            if os.path.exists(filepath):
                file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  - ✅ 保存成功! ファイルサイズ: {file_size_mb:.2f} MB")
                
                # チェックポイントファイルの内容確認
                try:
                    checkpoint = torch.load(filepath, map_location='cpu')
                    print(f"  - チェックポイント内容: {list(checkpoint.keys())}")
                    if 'state_dict' in checkpoint:
                        print(f"  - state_dict キー数: {len(checkpoint['state_dict'])}")
                    if 'epoch' in checkpoint:
                        print(f"  - エポック: {checkpoint['epoch']}")
                except Exception as e:
                    print(f"  - チェックポイント読み込み確認エラー: {e}")
            else:
                print(f"  - ❌ ファイルが作成されていません")
            
            return result
            
        except Exception as e:
            print(f"  - ❌ 保存エラー: {e}")
            import traceback
            traceback.print_exc()
            # エラーでも処理を続行
            return None

def check_checkpoints(checkpoint_dir):
    """チェックポイントディレクトリの内容を確認"""
    import os
    import torch
    
    print(f"\n=== Checkpoint Directory Contents ===")
    print(f"Directory: {checkpoint_dir}")
    
    if not os.path.exists(checkpoint_dir):
        print("❌ Checkpoint directory does not exist!")
        return
    
    files = os.listdir(checkpoint_dir)
    ckpt_files = [f for f in files if f.endswith('.ckpt')]
    
    print(f"Total files: {len(files)}")
    print(f"Checkpoint files: {len(ckpt_files)}")
    
    for f in ckpt_files:
        filepath = os.path.join(checkpoint_dir, f)
        try:
            # チェックポイントファイルの内容を確認
            checkpoint = torch.load(filepath, map_location='cpu')
            print(f"\n✅ {f}:")
            print(f"  - File size: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")
            print(f"  - Keys: {list(checkpoint.keys())}")
            if 'state_dict' in checkpoint:
                print(f"  - Model parameters: {len(checkpoint['state_dict'])} keys")
                # 最初の数個のパラメータ名を表示
                param_names = list(checkpoint['state_dict'].keys())[:5]
                print(f"  - Sample params: {param_names}")
            if 'epoch' in checkpoint:
                print(f"  - Epoch: {checkpoint['epoch']}")
        except Exception as e:
            print(f"❌ {f}: Error loading - {e}")
    
    return ckpt_files

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
    #checkpoint_callback = ModelCheckpoint(
    checkpoint_callback = VerboseModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch:03d}-{train_loss:.4f}",
        monitor="train_loss",  # 明示しておく
        mode="min",
        # every_n_epochs=args.save_every_n_epochs,
        every_n_epochs=1,
        save_top_k=args.save_top_k,   # とりあえず１個
        save_last=True,               # 最後の１個も保存
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    ## 1. Best & Last model checkpoint (based on loss)
    #best_checkpoint_callback = ModelCheckpoint(
    #    dirpath=checkpoint_dir,
    #    filename="best-{epoch:03d}-{train_loss:.4f}",
    #    monitor="train_loss",
    #    mode="min",
    #    save_top_k=args.save_top_k,
    #    save_last=True,  # 最後のエポックも保存
    #    verbose=True,
    #)
    #callbacks.append(best_checkpoint_callback)
    
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
        gradient_clip_val=1.0,  # Gradient clipping for stability
        log_every_n_steps=50,
        check_val_every_n_epoch=10,  # Validation can be added later
        default_root_dir=checkpoint_dir,  # ログとチェックポイントの保存先
        enable_checkpointing=True,
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

    # for Debug
    # trainer.fit()の後に追加
    print(f"\nTraining completed!")
    saved_checkpoints = check_checkpoints(checkpoint_dir)

    if saved_checkpoints:
        print(f"✅ Successfully saved {len(saved_checkpoints)} checkpoints")
    else:
        print("❌ No checkpoints were saved!")
        print("Possible causes:")
        print("1. Training didn't complete properly")
        print("2. Checkpoint callback configuration issue")
        print("3. Disk space or permission issues")
        print("4. PyTorch Lightning version compatibility")




if __name__ == "__main__":
    cli_main()


