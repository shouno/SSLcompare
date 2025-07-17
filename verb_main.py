import argparse
import pytorch_lightning as pl
from ssl_src.common import CIFAR10DataModule, ImageNetDataModule
from ssl_src.simsiam import SimSiamModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import os
import torch
import shutil
from datetime import datetime
import tempfile


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

# テスト用の簡単なコールバック作成関数
def create_test_callbacks(checkpoint_dir):
    """テスト用のコールバックを作成"""
    callbacks = []
    
    # 1. 無条件で毎エポック保存
    force_save = VerboseModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="force-{epoch:03d}",
        every_n_epochs=1,
        save_top_k=-1,
        monitor=None,  # モニタリングなし
        verbose=True,
    )
    callbacks.append(force_save)
    
    # 2. train_lossをモニタリング
    if True:  # train_lossがログされている場合のみ
        best_save = VerboseModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best-{epoch:03d}-{train_loss:.4f}",
            monitor="train_loss",
            mode="min",
            save_top_k=2,
            save_last=True,
            verbose=True,
        )
        callbacks.append(best_save)
    
    return callbacks

# 使用例（main.pyに追加）
def test_checkpoint_system():
    """チェックポイントシステムのテスト"""
    print("🔧 チェックポイントシステムのテスト開始")
    
    # テンポラリディレクトリでのテスト
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"テストディレクトリ: {temp_dir}")
        
        # ダミーのtrainerとmodelでテスト
        # (実際の実装では本物のtrainerとmodelを使用)
        pass


def system_diagnostics():
    """システム診断"""
    print("🔧 システム診断:")
    print(f"  - PyTorch version: {torch.__version__}")
    print(f"  - PyTorch Lightning version: {pl.__version__}")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    print(f"  - CUDA devices: {torch.cuda.device_count()}")
    print(f"  - Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")
    
    # ディスク容量チェック
    disk_usage = shutil.disk_usage('/workspace')
    print(f"  - Workspace disk usage:")
    print(f"    - Total: {disk_usage.total / (1024**3):.2f} GB")
    print(f"    - Used: {disk_usage.used / (1024**3):.2f} GB") 
    print(f"    - Free: {disk_usage.free / (1024**3):.2f} GB")
    
    # PyTorch保存テスト
    test_file = '/workspace/torch_save_test.pth'
    try:
        test_tensor = torch.randn(10, 10)
        torch.save(test_tensor, test_file)
        loaded = torch.load(test_file)
        os.remove(test_file)
        print(f"  - PyTorch save/load test: ✅ Success")
    except Exception as e:
        print(f"  - PyTorch save/load test: ❌ Failed - {e}")

def cli_main():
    system_diagnostics()
    
    parser = argparse.ArgumentParser(description="SSL Investigation")
    parser.add_argument("--method", type=str, default="simsiam")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--max_epochs", type=int, default=3)  # 短時間テスト
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--base_encoder", type=str, default="resnet50")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--project_name", type=str, default="ssl-investigation")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    args = parser.parse_args()

    # データモジュール
    dm = CIFAR10DataModule(
        data_dir=args.data_dir,
        method=args.method,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # モデル
    model = SimSiamModule(
        base_encoder=args.base_encoder,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # チェックポイントディレクトリ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"investigation_{timestamp}"
    checkpoint_dir = os.path.join(args.checkpoint_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"\n🔧 Investigation setup:")
    print(f"  - Checkpoint dir: {checkpoint_dir}")
    print(f"  - Dir exists: {os.path.exists(checkpoint_dir)}")
    print(f"  - Dir writable: {os.access(checkpoint_dir, os.W_OK)}")

    # 詳細調査用コールバック
    callbacks = []
    
    # 1. 最もシンプルな保存（無条件、毎エポック）
    simple_callback = VerboseModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="simple-{epoch:03d}",
        every_n_epochs=1,
        save_top_k=-1,
        monitor=None,  # モニタリング無し
        save_last=False,
        verbose=True,
    )
    callbacks.append(simple_callback)
    
    # 2. train_lossモニタリング付き
    monitored_callback = VerboseModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="monitored-{epoch:03d}",
        monitor="train_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    callbacks.append(monitored_callback)
    
    # 3. 学習率モニター
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)

    # ロガー
    wandb_logger = WandbLogger(
        project=args.project_name,
        name=run_name,
        save_dir=checkpoint_dir,
    )

    # トレーナー（最もシンプルな設定）
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        precision=args.precision,
        max_epochs=args.max_epochs,
        log_every_n_steps=10,
        default_root_dir=checkpoint_dir,
        enable_checkpointing=True,
        # 余計な設定は削除
    )

    print(f"\n🔧 Trainer info:")
    print(f"  - Device IDs: {trainer.device_ids}")
    print(f"  - World size: {trainer.world_size}")
    print(f"  - Global rank: {trainer.global_rank}")
    print(f"  - Local rank: {trainer.local_rank}")

    # 学習開始
    print(f"\n🚀 Starting training...")
    trainer.fit(model, dm)
    
    # 終了後の詳細確認
    print(f"\n📊 Training completed. Final investigation:")
    
    # コールバックの最終状態
    for i, callback in enumerate(trainer.callbacks):
        if isinstance(callback, VerboseModelCheckpoint):
            print(f"  Callback {i}:")
            print(f"    - Type: {callback.__class__.__name__}")
            print(f"    - Best model path: {callback.best_model_path}")
            print(f"    - Last model path: {callback.last_model_path}")
            print(f"    - Best model score: {callback.best_model_score}")
    
    # ディレクトリの最終確認
    if os.path.exists(checkpoint_dir):
        all_files = os.listdir(checkpoint_dir)
        ckpt_files = [f for f in all_files if f.endswith('.ckpt')]
        
        print(f"  Final directory contents:")
        print(f"    - Total files: {len(all_files)}")
        print(f"    - All files: {all_files}")
        print(f"    - .ckpt files: {ckpt_files}")
        
        for ckpt in ckpt_files:
            filepath = os.path.join(checkpoint_dir, ckpt)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"    - {ckpt}: {size_mb:.2f} MB")

if __name__ == "__main__":
    cli_main()