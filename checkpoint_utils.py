import os
import torch
import glob
from pathlib import Path
import json
from datetime import datetime


def list_checkpoints(checkpoint_dir="checkpoints", method=None):
    """
    チェックポイントディレクトリ内のすべてのチェックポイントをリスト表示
    
    Args:
        checkpoint_dir: チェックポイントのルートディレクトリ
        method: 特定の手法のみフィルタリング（例: "simclr", "byol"）
    """
    runs = []
    
    for run_dir in os.listdir(checkpoint_dir):
        run_path = os.path.join(checkpoint_dir, run_dir)
        if not os.path.isdir(run_path):
            continue
            
        # メソッドでフィルタリング
        if method and not run_dir.startswith(method):
            continue
            
        # チェックポイント情報を読み込み
        info_file = os.path.join(run_path, "checkpoint_info.txt")
        if os.path.exists(info_file):
            with open(info_file, "r") as f:
                info = {}
                for line in f:
                    if ":" in line:
                        key, value = line.strip().split(":", 1)
                        info[key.strip()] = value.strip()
                
                # チェックポイントファイルのリスト
                ckpt_files = glob.glob(os.path.join(run_path, "*.ckpt"))
                info["num_checkpoints"] = len(ckpt_files)
                info["run_path"] = run_path
                
                runs.append(info)
    
    # 表示
    print(f"\n{'='*80}")
    print(f"Available checkpoints in {checkpoint_dir}:")
    print(f"{'='*80}\n")
    
    for i, run in enumerate(runs, 1):
        print(f"{i}. {run.get('Run name', 'Unknown')}")
        print(f"   Method: {run.get('Method', 'Unknown')}")
        print(f"   Dataset: {run.get('Dataset', 'Unknown')}")
        print(f"   Epochs: {run.get('Epochs', 'Unknown')}")
        print(f"   Best loss: {run.get('Train loss at best', 'Unknown')}")
        print(f"   Checkpoints: {run.get('num_checkpoints', 0)} files")
        print(f"   Path: {run.get('run_path', '')}")
        print()
    
    return runs


def load_checkpoint(checkpoint_path, map_location=None):
    """
    チェックポイントを読み込んで情報を表示
    
    Args:
        checkpoint_path: チェックポイントファイルのパス
        map_location: デバイスマッピング（例: "cpu", "cuda:0"）
    
    Returns:
        checkpoint: ロードされたチェックポイント辞書
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=map_location or "cpu")
    
    print(f"\n{'='*60}")
    print(f"Checkpoint loaded: {os.path.basename(checkpoint_path)}")
    print(f"{'='*60}")
    
    # エポック情報
    if "epoch" in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    
    # ステップ情報
    if "global_step" in checkpoint:
        print(f"Global step: {checkpoint['global_step']}")
    
    # ハイパーパラメータ
    if "hyper_parameters" in checkpoint:
        print("\nHyperparameters:")
        hp = checkpoint["hyper_parameters"]
        for key, value in hp.items():
            print(f"  {key}: {value}")
    
    # モデル状態のキー
    if "state_dict" in checkpoint:
        print(f"\nModel state dict keys: {len(checkpoint['state_dict'])}")
        # エンコーダーの層数を推定
        encoder_keys = [k for k in checkpoint["state_dict"].keys() if "encoder" in k]
        print(f"Encoder-related keys: {len(encoder_keys)}")
    
    # 最適化器の状態
    if "optimizer_states" in checkpoint:
        print(f"\nOptimizer states: {len(checkpoint['optimizer_states'])}")
    
    # コールバックの状態
    if "callbacks" in checkpoint:
        print(f"\nCallbacks: {list(checkpoint['callbacks'].keys())}")
    
    return checkpoint


def extract_encoder_weights(checkpoint_path, output_path=None):
    """
    チェックポイントからエンコーダーの重みのみを抽出
    
    Args:
        checkpoint_path: 元のチェックポイントファイル
        output_path: 出力パス（Noneの場合は自動生成）
    
    Returns:
        encoder_path: エンコーダー重みの保存パス
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # エンコーダーの重みを抽出
    encoder_state_dict = {}
    for key, value in checkpoint["state_dict"].items():
        if key.startswith("encoder."):
            # "encoder."プレフィックスを削除
            new_key = key.replace("encoder.", "", 1)
            encoder_state_dict[new_key] = value
    
    # 出力パスの生成
    if output_path is None:
        base_path = os.path.splitext(checkpoint_path)[0]
        output_path = f"{base_path}_encoder_only.pth"
    
    # 保存
    torch.save({
        "encoder_state_dict": encoder_state_dict,
        "original_checkpoint": checkpoint_path,
        "extraction_date": datetime.now().isoformat(),
        "num_parameters": len(encoder_state_dict),
        "hyper_parameters": checkpoint.get("hyper_parameters", {}),
    }, output_path)
    
    print(f"Encoder weights extracted: {output_path}")
    print(f"Number of parameters: {len(encoder_state_dict)}")
    
    return output_path


def compare_checkpoints(checkpoint1_path, checkpoint2_path):
    """
    2つのチェックポイントを比較
    
    Args:
        checkpoint1_path: 最初のチェックポイント
        checkpoint2_path: 2番目のチェックポイント
    """
    ckpt1 = torch.load(checkpoint1_path, map_location="cpu")
    ckpt2 = torch.load(checkpoint2_path, map_location="cpu")
    
    print(f"\n{'='*60}")
    print("Checkpoint Comparison")
    print(f"{'='*60}")
    
    print(f"\nCheckpoint 1: {os.path.basename(checkpoint1_path)}")
    print(f"Checkpoint 2: {os.path.basename(checkpoint2_path)}")
    
    # エポックの比較
    epoch1 = ckpt1.get("epoch", "Unknown")
    epoch2 = ckpt2.get("epoch", "Unknown")
    print(f"\nEpochs: {epoch1} vs {epoch2}")
    
    # ハイパーパラメータの比較
    hp1 = ckpt1.get("hyper_parameters", {})
    hp2 = ckpt2.get("hyper_parameters", {})
    
    print("\nHyperparameter differences:")
    all_keys = set(hp1.keys()) | set(hp2.keys())
    for key in sorted(all_keys):
        val1 = hp1.get(key, "N/A")
        val2 = hp2.get(key, "N/A")
        if val1 != val2:
            print(f"  {key}: {val1} → {val2}")
    
    # モデルの重みの差分をチェック
    if "state_dict" in ckpt1 and "state_dict" in ckpt2:
        sd1 = ckpt1["state_dict"]
        sd2 = ckpt2["state_dict"]
        
        # 重みの差分の統計
        print("\nWeight differences:")
        common_keys = set(sd1.keys()) & set(sd2.keys())
        
        if common_keys:
            diffs = []
            for key in common_keys:
                if sd1[key].shape == sd2[key].shape:
                    diff = (sd1[key] - sd2[key]).abs().mean().item()
                    diffs.append(diff)
            
            if diffs:
                print(f"  Average absolute difference: {sum(diffs)/len(diffs):.6f}")
                print(f"  Max absolute difference: {max(diffs):.6f}")
                print(f"  Min absolute difference: {min(diffs):.6f}")


def cleanup_old_checkpoints(checkpoint_dir, keep_best_n=3, keep_last=True, dry_run=True):
    """
    古いチェックポイントをクリーンアップ
    
    Args:
        checkpoint_dir: チェックポイントディレクトリ
        keep_best_n: 保持するベストチェックポイントの数
        keep_last: 最後のチェックポイントを保持するか
        dry_run: 実際に削除せずに、削除対象を表示するだけ
    """
    print(f"\n{'='*60}")
    print(f"Cleanup checkpoints in: {checkpoint_dir}")
    print(f"{'='*60}")
    
    # すべての.ckptファイルを取得
    ckpt_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    
    if not ckpt_files:
        print("No checkpoint files found.")
        return
    
    # ファイルを分類
    best_files = []
    periodic_files = []
    last_files = []
    
    for f in ckpt_files:
        basename = os.path.basename(f)
        if basename.startswith("best-"):
            best_files.append(f)
        elif basename.startswith("periodic-"):
            periodic_files.append(f)
        elif "last" in basename:
            last_files.append(f)
    
    # 削除対象を決定
    to_delete = []
    
    # ベストチェックポイントの処理
    if len(best_files) > keep_best_n:
        # ファイル名から損失値を抽出してソート
        best_with_loss = []
        for f in best_files:
            try:
                # "best-{epoch:03d}-{train_loss:.4f}.ckpt" の形式を解析
                parts = os.path.basename(f).split("-")
                if len(parts) >= 3:
                    loss_str = parts[2].replace(".ckpt", "")
                    loss = float(loss_str)
                    best_with_loss.append((f, loss))
            except:
                pass
        
        # 損失でソート（昇順）
        best_with_loss.sort(key=lambda x: x[1])
        
        # 保持するファイル以外を削除対象に
        for f, _ in best_with_loss[keep_best_n:]:
            to_delete.append(f)
    
    # 最後のチェックポイントの処理
    if not keep_last:
        to_delete.extend(last_files)
    
    # 結果を表示
    print(f"\nTotal checkpoints: {len(ckpt_files)}")
    print(f"Best checkpoints: {len(best_files)} (keep {keep_best_n})")
    print(f"Periodic checkpoints: {len(periodic_files)}")
    print(f"Last checkpoints: {len(last_files)} (keep: {keep_last})")
    print(f"\nTo be deleted: {len(to_delete)} files")
    
    if to_delete:
        print("\nFiles to delete:")
        for f in to_delete:
            size_mb = os.path.getsize(f) / (1024 * 1024)
            print(f"  - {os.path.basename(f)} ({size_mb:.1f} MB)")
        
        total_size_mb = sum(os.path.getsize(f) for f in to_delete) / (1024 * 1024)
        print(f"\nTotal space to free: {total_size_mb:.1f} MB")
        
        if not dry_run:
            response = input("\nProceed with deletion? (y/N): ")
            if response.lower() == 'y':
                for f in to_delete:
                    os.remove(f)
                    print(f"Deleted: {os.path.basename(f)}")
                print("\nCleanup completed!")
            else:
                print("Cleanup cancelled.")
        else:
            print("\n(Dry run - no files were deleted)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Checkpoint management utilities")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all checkpoints")
    list_parser.add_argument("--dir", default="checkpoints", help="Checkpoint directory")
    list_parser.add_argument("--method", help="Filter by method")
    
    # Load command
    load_parser = subparsers.add_parser("load", help="Load and inspect checkpoint")
    load_parser.add_argument("path", help="Checkpoint file path")
    
    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract encoder weights")
    extract_parser.add_argument("path", help="Checkpoint file path")
    extract_parser.add_argument("--output", help="Output path")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two checkpoints")
    compare_parser.add_argument("path1", help="First checkpoint")
    compare_parser.add_argument("path2", help="Second checkpoint")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old checkpoints")
    cleanup_parser.add_argument("dir", help="Checkpoint directory")
    cleanup_parser.add_argument("--keep-best", type=int, default=3, help="Number of best checkpoints to keep")
    cleanup_parser.add_argument("--no-keep-last", action="store_true", help="Don't keep last checkpoint")
    cleanup_parser.add_argument("--execute", action="store_true", help="Actually delete files (not dry run)")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_checkpoints(args.dir, args.method)
    elif args.command == "load":
        load_checkpoint(args.path)
    elif args.command == "extract":
        extract_encoder_weights(args.path, args.output)
    elif args.command == "compare":
        compare_checkpoints(args.path1, args.path2)
    elif args.command == "cleanup":
        cleanup_old_checkpoints(
            args.dir, 
            keep_best_n=args.keep_best,
            keep_last=not args.no_keep_last,
            dry_run=not args.execute
        )
    else:
        parser.print_help()