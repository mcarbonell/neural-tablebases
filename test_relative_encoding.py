"""Test the relative encoding"""
import numpy as np
import sys
sys.path.insert(0, 'src')
from train import TablebaseDataset
from models import get_model_for_endgame

print("=" * 70)
print("TESTING RELATIVE ENCODING")
print("=" * 70)

# Load dataset
dataset = TablebaseDataset('data/KQvK.npz')

print(f"\nOK Dataset loaded successfully")
print(f"  - Positions: {len(dataset)}")
print(f"  - Input dims: {dataset.input_size}")
print(f"  - Num pieces: {dataset.num_pieces}")
print(f"  - Relative encoding: {dataset.use_relative_encoding}")

# Create model
model = get_model_for_endgame('mlp', dataset.num_pieces, 3, dataset.use_relative_encoding)
print(f"\nOK Model created successfully")
print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  - Input size: {model.backbone[0].in_features}")

# Test forward pass
import torch
x, wdl, dtz = dataset[0]
x_batch = x.unsqueeze(0)

model.eval()  # Set to eval mode to avoid BatchNorm issues
with torch.no_grad():
    wdl_logits, dtz_pred = model(x_batch)
    print(f"\nOK Forward pass successful")
    print(f"  - Input shape: {x_batch.shape}")
    print(f"  - WDL output shape: {wdl_logits.shape}")
    print(f"  - DTZ output shape: {dtz_pred.shape}")
    print(f"  - WDL prediction: {wdl_logits.argmax(1).item()} (actual: {wdl.item()})")

print("\n" + "=" * 70)
print("ALL TESTS PASSED! Ready to train.")
print("=" * 70)

print("\nTo train:")
print("  python src/train.py --data_path data/KQvK.npz --model mlp --epochs 100 --batch_size 2048 --lr 0.001")
