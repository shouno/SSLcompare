#!/usr/bin/env python3
"""
複数のSSL手法を自動的に実行し、結果を比較するスクリプト
"""

import subprocess
import os
import time
import json
from datetime import datetime
import argparse


def run_experiment(method, dataset, data_dir, epochs, batch_size, extra_args=None):
    """
    単一の実験を実行
    
    Args:
        method: SSL手法
        dataset: データセット名
        data_dir: データディレクトリ
        epochs: エポック数
        batch_size: バッチサイズ
        extra_args: 追加の引数（辞書形式）
    
    Returns:
        実験の結果情報
    """
    cmd = [
        "python", "main.py",
        "--method", method,
        "--dataset", dataset,
        "--data_dir", data_dir,
        "--max_epochs", str(epochs),
        "--batch_size", str(batch_size),
    ]
    
    # 手法固有の引数を追加
    if extra_args:
        for key, value in extra_args.items():
            cmd.extend([f"--{key}", str(value)])
    
    print(f"\n{'='*60}")
    print(f"Starting experiment: {method} on {dataset}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    try:
        # 実験を実行
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # 実行時間
        elapsed_time = time.time() - start_time
        
        # 結果を解析
        output_lines = result.stdout.split('\n')
        checkpoint_dir = None
        best_checkpoint = None
        
        for line in output_lines:
            if "Checkpoints saved in:" in line:
                checkpoint_dir = line.split(":")[-1].strip()
            elif "Best checkpoint:" in line:
                best_checkpoint = line.split(":")[-1].strip()
        
        return {
            "method": method,
            "dataset": dataset,
            "status": "success",
            "elapsed_time": elapsed_time,
            "checkpoint_dir": checkpoint_dir,
            "best_checkpoint": best_checkpoint,
            "command": " ".join(cmd),
        }
        
    except subprocess.CalledProcessError as e:
        print(f"Error running {method}: {e}")
        return {
            "method": method,
            "dataset": dataset,
            "status": "failed",
            "error": str(e),
            "command": " ".join(cmd),
        }


def run_linear_evaluation(checkpoint_path, data_dir, epochs=100):
    """
    線形評価を実行
    
    Args:
        checkpoint_path: チェックポイントのパス
        data_dir: データディレクトリ
        epochs: 評価エポック数
    
    Returns:
        評価結果
    """
    cmd = [
        "python", "linear_eval.py",
        "--checkpoint", checkpoint_path,
        "--data_dir", data_dir,
        "--max_epochs", str(epochs),
        "--freeze_backbone",
    ]
    
    print(f"\nRunning linear evaluation for: {checkpoint_path}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # 結果から精度を抽出
        output_lines = result.stdout.split('\n')
        test_acc = None
        
        for line in output_lines:
            if "test_acc" in line:
                # PyTorch Lightningのログ形式から精度を抽出
                parts = line.split()
                for part in parts:
                    if "test_acc" in part:
                        try:
                            test_acc = float(part.split("=")[-1])
                        except:
                            pass
        
        return {
            "checkpoint": checkpoint_path,
            "test_accuracy": test_acc,
            "status": "success",
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "checkpoint": checkpoint_path,
            "status": "failed",
            "error": str(e),
        }


def run_all_experiments(config):
    """
    設定に基づいてすべての実験を実行
    
    Args:
        config: 実験設定
    
    Returns:
        すべての実験結果
    """
    results = {
        "start_time": datetime.now().isoformat(),
        "config": config,
        "experiments": [],
        "evaluations": [],
    }
    
    # 各手法で実験を実行
    for method_config in config["methods"]:
        method = method_config["name"]
        extra_args = method_config.get("extra_args", {})
        
        # SSL学習を実行
        exp_result = run_experiment(
            method=method,
            dataset=config["dataset"],
            data_dir=config["data_dir"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            extra_args=extra_args,
        )
        
        results["experiments"].append(exp_result)
        
        # 成功した場合は線形評価も実行
        if exp_result["status"] == "success" and exp_result["best_checkpoint"]:
            if config.get("run_linear_eval", True):
                eval_result = run_linear_evaluation(
                    checkpoint_path=exp_result["best_checkpoint"],
                    data_dir=config["data_dir"],
                    epochs=config.get("linear_eval_epochs", 100),
                )
                eval_result["method"] = method
                results["evaluations"].append(eval_result)
    
    results["end_time"] = datetime.now().isoformat()
    
    # 結果を保存
    output_file = f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    
    # サマリーを表示
    print("\nSummary:")
    print("-" * 40)
    
    for exp in results["experiments"]:
        status = "✓" if exp["status"] == "success" else "✗"
        print(f"{status} {exp['method']}: {exp['status']}")
        
        if exp["status"] == "success":
            # 対応する評価結果を探す
            eval_results = [e for e in results["evaluations"] 
                          if e.get("method") == exp["method"]]
            if eval_results and eval_results[0].get("test_accuracy"):
                print(f"  → Test accuracy: {eval_results[0]['test_accuracy']:.2%}")
    
    return results


def create_default_config():
    """デフォルトの実験設定を作成"""
    return {
        "dataset": "cifar10",
        "data_dir": "./data",
        "epochs": 200,  # デモ用に短めに設定
        "batch_size": 256,
        "linear_eval_epochs": 100,
        "run_linear_eval": True,
        "methods": [
            {
                "name": "simclr",
                "extra_args": {
                    "temperature": 0.5,
                    "lr": 0.5,
                }
            },
            {
                "name": "byol",
                "extra_args": {
                    "ema_decay": 0.996,
                    "lr": 0.2,
                }
            },
            {
                "name": "simsiam",
                "extra_args": {
                    "lr": 0.1,
                }
            },
            {
                "name": "barlow",
                "extra_args": {
                    "lambd": 0.005,
                    "lr": 0.2,
                }
            },
            {
                "name": "swav",
                "extra_args": {
                    "temperature": 0.1,
                    "n_prototypes": 300,
                    "lr": 0.2,
                }
            },
            {
                "name": "mae",
                "extra_args": {
                    "mask_ratio": 0.75,
                    "patch_size": 4,
                }
            },
        ]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple SSL experiments")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--create-config", action="store_true", 
                       help="Create a default config file")
    parser.add_argument("--methods", nargs="+", 
                       choices=["simclr", "byol", "simsiam", "barlow", "swav", "mae"],
                       help="Methods to run (default: all)")
    parser.add_argument("--dataset", type=str, default="cifar10",
                       choices=["cifar10", "imagenet"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--no-linear-eval", action="store_true",
                       help="Skip linear evaluation")
    
    args = parser.parse_args()
    
    if args.create_config:
        # デフォルト設定ファイルを作成
        config = create_default_config()
        with open("experiment_config.json", "w") as f:
            json.dump(config, f, indent=2)
        print("Created default config file: experiment_config.json")
        print("Edit this file and run again with --config experiment_config.json")
    else:
        # 設定を読み込むか作成
        if args.config:
            with open(args.config, "r") as f:
                config = json.load(f)
        else:
            config = create_default_config()
            
            # コマンドライン引数で上書き
            if args.methods:
                config["methods"] = [{"name": m, "extra_args": {}} 
                                   for m in args.methods]
            config["dataset"] = args.dataset
            config["data_dir"] = args.data_dir
            config["epochs"] = args.epochs
            config["batch_size"] = args.batch_size
            config["run_linear_eval"] = not args.no_linear_eval
        
        # 実験を実行
        run_all_experiments(config)