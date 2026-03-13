"""
Test different SIREN hyperparameters to find optimal configuration.
"""
import subprocess
import sys

# Test configurations
configs = [
    # (lr, omega_0, hidden_size, num_layers, description)
    (0.0001, 30, 128, 3, "low_lr"),
    (0.001, 10, 128, 3, "low_omega"),
    (0.001, 30, 256, 4, "larger_arch"),
    (0.0001, 10, 256, 4, "low_lr_low_omega_large"),
]

print("Testing SIREN hyperparameters on KQvK...")
print("=" * 60)

results = []

for lr, omega_0, hidden_size, num_layers, desc in configs:
    print(f"\nTesting: {desc}")
    print(f"  LR: {lr}, Omega_0: {omega_0}, Hidden: {hidden_size}, Layers: {num_layers}")
    
    # Note: This would require modifying train.py to accept these parameters
    # For now, we'll document the approach
    
    print(f"  Command: python src/train.py --data_path data/KQvK.npz --model siren")
    print(f"           --lr {lr} --siren_omega_0 {omega_0}")
    print(f"           --siren_hidden {hidden_size} --siren_layers {num_layers}")
    print(f"           --epochs 30 --batch_size 2048")
    
    results.append({
        'config': desc,
        'lr': lr,
        'omega_0': omega_0,
        'hidden_size': hidden_size,
        'num_layers': num_layers
    })

print("\n" + "=" * 60)
print("To run these tests, we need to add SIREN hyperparameters to train.py")
print("Recommended additions to argparse:")
print("  --siren_omega_0 (default: 30)")
print("  --siren_hidden (default: 128)")
print("  --siren_layers (default: 3)")
