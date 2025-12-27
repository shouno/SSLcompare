#!/bin/bash

TRAIN_PY=scripts/train.py
DATASET=cifar10
DATAFOLDER=/workspace/data

# SimCLR/BYOL/SimSiam
python3 $(TRAIN_PY) --method simclr  --dataset $(DATASET) --data_dir $(DATAFOLDER)/$(DATASET) --max_epochs 10
#python3 main.py --method byol    --dataset $(DATASET) --data_dir $(DATAFOLDER)/$(DATASET)
#python3 main.py --method simsiam --dataset $(DATASET) --data_dir $(DATAFOLDER)/$(DATASET)
#python3 main.py --method barlow  --dataset $(DATASET) --data_dir $(DATAFOLDER)/$(DATASET)

# SwAV (メモリ使用量が多い)
#python3 main.py --method swav    --dataset $(DATASET) --data_dir $(DATAFOLDER)/$(DATASET)

# MAE (メモリ使用量が多い)
#python3 main.py --method mae     --dataset $(DATASET) --data_dir $(DATAFOLDER)/$(DATASET) --batch_size 256 --max_epochs 200