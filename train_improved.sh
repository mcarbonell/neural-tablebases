#!/bin/bash
# Training script with improved settings

echo "=== Training MLP with improved settings ==="
python src/train.py \
    --data_path data/KQvK.npz \
    --model mlp \
    --epochs 1000 \
    --batch_size 2048 \
    --lr 0.001 \
    --patience 100 \
    --hard_weight 2.0 \
    --hard_mining \
    --hard_mining_freq 50 \
    --hard_mining_epochs 3

echo ""
echo "=== Training SIREN with improved settings ==="
python src/train.py \
    --data_path data/KQvK.npz \
    --model siren \
    --epochs 1000 \
    --batch_size 2048 \
    --lr 0.0005 \
    --patience 100 \
    --hard_weight 2.0 \
    --hard_mining \
    --hard_mining_freq 50 \
    --hard_mining_epochs 3
