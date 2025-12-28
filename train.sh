#!/bin/bash

TRAIN_PY=scripts/train.py
DATASET=cifar10
DATAFOLDER=/workspace/data
MAXEPOCHS=200

# SimCLR/BYOL/SimSiam
python3 $TRAIN_PY --method simclr       --dataset $DATASET --data_dir $DATAFOLDER/$DATASET --max_epochs $MAXEPOCHS
python3 $TRAIN_PY --method byol         --dataset $DATASET --data_dir $DATAFOLDER/$DATASET --max_epochs $MAXEPOCHS
python3 $TRAIN_PY --method simsiam      --dataset $DATASET --data_dir $DATAFOLDER/$DATASET --max_epochs $MAXEPOCHS
python3 $TRAIN_PY --method barlowtwins  --dataset $DATASET --data_dir $DATAFOLDER/$DATASET --max_epochs $MAXEPOCHS

# SwAV (メモリ使用量が多い)
python3 $TRAIN_PY --method swav         --dataset $DATASET --data_dir $DATAFOLDER/$DATASET --max_epochs $MAXEPOCHS

# MAE (メモリ使用量が多い)
python3 $TRAIN_PY --method mae          --dataset $DATASET --data_dir $DATAFOLDER/$DATASET --lr 1e-4 --max_epochs $MAXEPOCHS --batch_size 256 