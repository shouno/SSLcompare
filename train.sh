#!/bin/bash

TRAIN_PY=scripts/train.py
DATASET=stl10
# DATAFOLDER=/workspace/data
DATAFOLDER=/datasets/$DATASET
MAXEPOCHS=200

# SimCLR/BYOL/SimSiam
python3 $TRAIN_PY --method simclr       --dataset $DATASET --data_dir $DATAFOLDER --max_epochs $MAXEPOCHS
python3 $TRAIN_PY --method byol         --dataset $DATASET --data_dir $DATAFOLDER --max_epochs $MAXEPOCHS
python3 $TRAIN_PY --method simsiam      --dataset $DATASET --data_dir $DATAFOLDER --max_epochs $MAXEPOCHS
python3 $TRAIN_PY --method barlowtwins  --dataset $DATASET --data_dir $DATAFOLDER --max_epochs $MAXEPOCHS

# SwAV (メモリ使用量が多い)
python3 $TRAIN_PY --method swav         --dataset $DATASET --data_dir $DATAFOLDER --max_epochs $MAXEPOCHS

# MAE (メモリ使用量が多い), エンコーダを ViT-MAE に固定, lr を低めに設定
python3 $TRAIN_PY --method mae          --dataset $DATASET --data_dir $DATAFOLDER --lr 1e-4 --base_encoder vit_mae --max_epochs $MAXEPOCHS --batch_size 256 