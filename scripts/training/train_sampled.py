"""
Train on a sampled subset of a large dataset for faster validation.
Useful for 4+ piece endgames with millions of positions.
"""
import numpy as np
import argparse
import os
import sys

def sample_dataset(input_path, output_path, sample_rate=0.2, seed=42):
    """
    Create a sampled version of a dataset.
    
    Args:
        input_path: Path to original .npz file
        output_path: Path to save sampled .npz file
        sample_rate: Fraction of data to keep (0.0 to 1.0)
        seed: Random seed for reproducibility
    """
    print(f"Loading dataset from {input_path}...")
    data = np.load(input_path)
    
    total_positions = len(data['x'])
    sample_size = int(total_positions * sample_rate)
    
    print(f"Total positions: {total_positions:,}")
    print(f"Sample rate: {sample_rate:.1%}")
    print(f"Sample size: {sample_size:,}")
    
    # Random sampling with seed
    np.random.seed(seed)
    indices = np.random.choice(total_positions, sample_size, replace=False)
    indices.sort()  # Keep some locality for better cache performance
    
    print(f"Sampling {len(indices):,} positions...")
    
    # Sample data
    x_sampled = data['x'][indices]
    wdl_sampled = data['wdl'][indices]
    dtz_sampled = data['dtz'][indices]
    
    # Save sampled dataset
    print(f"Saving to {output_path}...")
    np.savez_compressed(output_path,
                        x=x_sampled,
                        wdl=wdl_sampled,
                        dtz=dtz_sampled)
    
    # Report sizes
    original_size = os.path.getsize(input_path) / 1024 / 1024
    sampled_size = os.path.getsize(output_path) / 1024 / 1024
    
    print(f"\nOriginal size: {original_size:.1f} MB")
    print(f"Sampled size: {sampled_size:.1f} MB")
    print(f"Compression: {original_size / sampled_size:.1f}x")
    print(f"\nSampled dataset saved to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample a large dataset for faster training")
    parser.add_argument("--input", type=str, required=True,
                       help="Input .npz file")
    parser.add_argument("--output", type=str, default=None,
                       help="Output .npz file (default: input_sampled.npz)")
    parser.add_argument("--rate", type=float, default=0.2,
                       help="Sample rate (default: 0.2 = 20%%)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--train", action="store_true",
                       help="Train immediately after sampling")
    parser.add_argument("--model", type=str, default="mlp",
                       choices=["mlp", "siren"],
                       help="Model architecture (default: mlp)")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Training epochs (default: 30)")
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = f"{base}_sampled_{int(args.rate*100)}pct.npz"
    
    # Sample dataset
    sampled_path = sample_dataset(args.input, args.output, args.rate, args.seed)
    
    # Train if requested
    if args.train:
        print(f"\n{'='*60}")
        print("Starting training on sampled dataset...")
        print(f"{'='*60}\n")
        
        import subprocess
        cmd = [
            "python", "src/train.py",
            "--data_path", sampled_path,
            "--model", args.model,
            "--epochs", str(args.epochs),
            "--batch_size", "2048"
        ]
        
        subprocess.run(cmd)
