#!/bin/bash

# SimCLR/BYOL/SimSiam
python3 main.py --method simclr  --dataset cifar10 --data_dir /workspace/data_dir --batch_size 256 --max_epochs 200
python3 main.py --method byol    --dataset cifar10 --data_dir /workspace/data_dir --batch_size 256 --max_epochs 200
python3 main.py --method simsiam --dataset cifar10 --data_dir /workspace/data_dir --batch_size 256 --max_epochs 200
python3 main.py --method barlow  --dataset cifar10 --data_dir /workspace/data_dir --batch_size 256 --max_epochs 200

# SwAV (メモリ使用量が多い)
python3 main.py --method swav    --dataset cifar10 --data_dir /workspace/data_dir --batch_size 256 --max_epochs 200

# MAE (メモリ使用量が多い)
python3 main.py --method mae     --dataset cifar10 --data_dir /workspace/data_dir --batch_size 256 --max_epochs 200