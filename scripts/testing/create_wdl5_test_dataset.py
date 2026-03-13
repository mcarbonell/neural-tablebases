"""
Create a test dataset with WDL 5 classes by artificially adding cursed/blessed positions.
This is for testing purposes only - real cursed/blessed positions would come from Syzygy.
"""
import numpy as np
import sys
import os

def create_wdl5_test_dataset(input_path, output_path, cursed_ratio=0.05, blessed_ratio=0.05):
    """
    Create a WDL 5-class dataset from a WDL 3-class dataset.
    
    Args:
        input_path: Path to 3-class dataset
        output_path: Path to save 5-class dataset
        cursed_ratio: Fraction of wins to convert to cursed wins
        blessed_ratio: Fraction of losses to convert to blessed losses
    """
    print(f"Loading 3-class dataset from {input_path}...")
    data = np.load(input_path)
    
    x = data['x']
    wdl = data['wdl'].copy()
    dtz = data['dtz']
    
    print(f"Original WDL distribution:")
    unique, counts = np.unique(wdl, return_counts=True)
    for val, count in zip(unique, counts):
        print(f"  WDL {val:2d}: {count:7d} ({count/len(wdl)*100:.1f}%)")
    
    # Convert some wins (2) to cursed wins (1)
    win_indices = np.where(wdl == 2)[0]
    num_cursed = int(len(win_indices) * cursed_ratio)
    cursed_indices = np.random.choice(win_indices, num_cursed, replace=False)
    wdl[cursed_indices] = 1
    print(f"\nConverted {num_cursed} wins to cursed wins ({cursed_ratio*100:.1f}%)")
    
    # Convert some losses (-2) to blessed losses (-1)
    loss_indices = np.where(wdl == -2)[0]
    num_blessed = int(len(loss_indices) * blessed_ratio)
    blessed_indices = np.random.choice(loss_indices, num_blessed, replace=False)
    wdl[blessed_indices] = -1
    print(f"Converted {num_blessed} losses to blessed losses ({blessed_ratio*100:.1f}%)")
    
    print(f"\nNew WDL distribution (5 classes):")
    unique, counts = np.unique(wdl, return_counts=True)
    wdl_names = {-2: "Loss", -1: "Blessed loss", 0: "Draw", 1: "Cursed win", 2: "Win"}
    for val, count in zip(unique, counts):
        print(f"  WDL {val:2d} ({wdl_names[val]:13s}): {count:7d} ({count/len(wdl)*100:.1f}%)")
    
    # Save new dataset
    print(f"\nSaving 5-class dataset to {output_path}...")
    np.savez_compressed(output_path, x=x, wdl=wdl, dtz=dtz)
    
    print(f"Done! Dataset saved with {len(x)} positions.")
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create WDL 5-class test dataset")
    parser.add_argument("--input", type=str, default="data/KQvK.npz",
                       help="Input 3-class dataset")
    parser.add_argument("--output", type=str, default="data/KQvK_wdl5.npz",
                       help="Output 5-class dataset")
    parser.add_argument("--cursed-ratio", type=float, default=0.05,
                       help="Fraction of wins to convert to cursed (default: 0.05)")
    parser.add_argument("--blessed-ratio", type=float, default=0.05,
                       help="Fraction of losses to convert to blessed (default: 0.05)")
    
    args = parser.parse_args()
    
    create_wdl5_test_dataset(args.input, args.output, args.cursed_ratio, args.blessed_ratio)
