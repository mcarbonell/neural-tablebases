"""Quick test script to verify the changes work"""
import torch
from src.models import get_model_for_endgame
from src.train import TablebaseDataset

print("=== Testing Model Changes ===\n")

# Test dataset loading
print("1. Loading dataset...")
dataset = TablebaseDataset("data/KQvK.npz")

print(f"\n2. Dataset info:")
print(f"   - Total positions: {len(dataset)}")
print(f"   - Input size: {dataset.input_size}")
print(f"   - Num pieces: {dataset.num_pieces}")
print(f"   - Class weights: {dataset.class_weights}")

# Test model creation
print(f"\n3. Creating MLP model...")
mlp_model = get_model_for_endgame("mlp", dataset.num_pieces, num_wdl_classes=3)
print(f"   - Parameters: {sum(p.numel() for p in mlp_model.parameters()):,}")
print(f"   - Architecture: {mlp_model}")

print(f"\n4. Creating SIREN model...")
siren_model = get_model_for_endgame("siren", dataset.num_pieces, num_wdl_classes=3)
print(f"   - Parameters: {sum(p.numel() for p in siren_model.parameters()):,}")

# Test forward pass
print(f"\n5. Testing forward pass...")
x, wdl, dtz = dataset[0]
x_batch = x.unsqueeze(0)

with torch.no_grad():
    wdl_logits, dtz_pred = mlp_model(x_batch)
    print(f"   - MLP output shapes: wdl={wdl_logits.shape}, dtz={dtz_pred.shape}")
    print(f"   - WDL classes: {wdl_logits.shape[1]} (expected 3)")
    
    wdl_logits, dtz_pred = siren_model(x_batch)
    print(f"   - SIREN output shapes: wdl={wdl_logits.shape}, dtz={dtz_pred.shape}")
    print(f"   - WDL classes: {wdl_logits.shape[1]} (expected 3)")

print("\n✓ All tests passed!")
