import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import argparse
from pathlib import Path
import yaml
import json


def read_tensorboard_logs(log_dir):
    """TensorBoardのログから学習曲線を読み込む"""
    try:
        from torch.utils.tensorboard import SummaryWriter
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("Please install tensorboard: pip install tensorboard")
        return None
    
    # イベントファイルを探す
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
        return None
    
    # 最新のイベントファイルを使用
    event_file = max(event_files, key=os.path.getctime)
    
    # EventAccumulatorでログを読み込む
    ea = EventAccumulator(event_file)
    ea.Reload()
    
    # 利用可能なスカラーメトリクスを取得
    scalar_tags = ea.Tags()['scalars']
    
    data = {}
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = pd.DataFrame({'step': steps, 'value': values})
    
    return data


def plot_training_curves(checkpoint_dirs, metrics=['train_loss', 'lr'], save_path=None):
    """
    複数の実験の学習曲線をプロット
    
    Args:
        checkpoint_dirs: チェックポイントディレクトリのリスト
        metrics: プロットするメトリクスのリスト
        save_path: 保存先のパス
    """
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 6*len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(range(len(checkpoint_dirs)))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        for j, ckpt_dir in enumerate(checkpoint_dirs):
            # 実験名を抽出
            exp_name = os.path.basename(ckpt_dir)
            method = exp_name.split('_')[0]
            
            # ログディレクトリを探す
            tb_dir = os.path.join(ckpt_dir, "tb_logs")
            if not os.path.exists(tb_dir):
                # PyTorch Lightningのデフォルトログ位置も確認
                tb_dir = os.path.join(ckpt_dir, "lightning_logs", "version_0")
            
            if os.path.exists(tb_dir):
                data = read_tensorboard_logs(tb_dir)
                if data and metric in data:
                    df = data[metric]
                    ax.plot(df['step'], df['value'], 
                           label=method, color=colors[j], linewidth=2)
        
        ax.set_xlabel('Step')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} During Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def plot_comparison_results(results_file, save_path=None):
    """
    実験結果の比較をプロット
    
    Args:
        results_file: run_experiments.pyで生成された結果ファイル
        save_path: 保存先のパス
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # 評価結果を抽出
    methods = []
    accuracies = []
    
    for eval_result in results['evaluations']:
        if eval_result['status'] == 'success' and eval_result.get('test_accuracy'):
            methods.append(eval_result['method'])
            accuracies.append(eval_result['test_accuracy'] * 100)  # パーセント表記
    
    if not methods:
        print("No evaluation results found")
        return
    
    # プロット
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(methods, accuracies, color=plt.cm.viridis(range(len(methods))))
    
    # 値をバーの上に表示
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom')
    
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title(f'Linear Evaluation Results on {results["config"]["dataset"].upper()}')
    ax.set_ylim(0, 100)
    ax.grid(True, axis='y', alpha=0.3)
    
    # 実験設定を表示
    config_text = f"Epochs: {results['config']['epochs']}, Batch Size: {results['config']['batch_size']}"
    ax.text(0.5, 0.02, config_text, transform=ax.transAxes, 
            ha='center', va='bottom', fontsize=10, style='italic')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def create_summary_table(checkpoint_dir):
    """
    チェックポイントディレクトリから要約テーブルを作成
    
    Args:
        checkpoint_dir: チェックポイントのルートディレクトリ
    
    Returns:
        pandas.DataFrame: 要約テーブル
    """
    summary_data = []
    
    for run_dir in os.listdir(checkpoint_dir):
        run_path = os.path.join(checkpoint_dir, run_dir)
        if not os.path.isdir(run_path):
            continue
        
        # checkpoint_info.txtを読み込む
        info_file = os.path.join(run_path, "checkpoint_info.txt")
        if os.path.exists(info_file):
            info = {}
            with open(info_file, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        info[key.strip()] = value.strip()
            
            # チェックポイントのサイズを計算
            ckpt_files = glob.glob(os.path.join(run_path, "*.ckpt"))
            total_size_mb = sum(os.path.getsize(f) for f in ckpt_files) / (1024 * 1024)
            
            summary_data.append({
                'Method': info.get('Method', 'Unknown'),
                'Dataset': info.get('Dataset', 'Unknown'),
                'Encoder': info.get('Base encoder', 'Unknown'),
                'Epochs': info.get('Epochs', 'Unknown'),
                'Best Loss': float(info.get('Train loss at best', 'NaN')) if info.get('Train loss at best') and info.get('Train loss at best') != 'None' else float('nan'),
                'Checkpoints': len(ckpt_files),
                'Size (MB)': round(total_size_mb, 1),
                'Directory': run_dir,
            })
    
    df = pd.DataFrame(summary_data)
    if not df.empty:
        df = df.sort_values(['Method', 'Dataset'])
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SSL training results")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Plot training curves
    plot_parser = subparsers.add_parser('plot', help='Plot training curves')
    plot_parser.add_argument('dirs', nargs='+', help='Checkpoint directories')
    plot_parser.add_argument('--metrics', nargs='+', default=['train_loss', 'lr'],
                            help='Metrics to plot')
    plot_parser.add_argument('--save', help='Save path for the plot')
    
    # Plot comparison results
    compare_parser = subparsers.add_parser('compare', help='Plot comparison results')
    compare_parser.add_argument('results_file', help='Results JSON file from run_experiments.py')
    compare_parser.add_argument('--save', help='Save path for the plot')
    
    # Create summary table
    summary_parser = subparsers.add_parser('summary', help='Create summary table')
    summary_parser.add_argument('--dir', default='checkpoints', help='Checkpoint directory')
    summary_parser.add_argument('--save', help='Save path for the CSV file')
    
    args = parser.parse_args()
    
    if args.command == 'plot':
        plot_training_curves(args.dirs, args.metrics, args.save)
    elif args.command == 'compare':
        plot_comparison_results(args.results_file, args.save)
    elif args.command == 'summary':
        df = create_summary_table(args.dir)
        print("\nCheckpoint Summary:")
        print("=" * 80)
        print(df.to_string(index=False))
        
        if args.save:
            df.to_csv(args.save, index=False)
            print(f"\nSummary saved to: {args.save}")
    else:
        parser.print_help()