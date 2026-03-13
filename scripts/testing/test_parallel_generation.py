"""
Quick test of parallel dataset generation.
Tests with KQvK (small dataset) to verify functionality.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from generate_datasets_parallel import generate_dataset_parallel
import time

def test_parallel():
    print("Testing parallel generation with KQvK...")
    print("This should take ~30 seconds with 8 cores vs ~3 minutes single-threaded\n")
    
    start = time.time()
    
    generate_dataset_parallel(
        syzygy_path="syzygy",
        output_dir="data",
        config="KQvK",
        compact=True,
        relative=True,
        use_move_distance=False,
        num_workers=None,  # Use all cores
        chunk_size=5000
    )
    
    elapsed = time.time() - start
    print(f"\nTest completed in {elapsed:.1f} seconds")
    
    # Verify output
    import numpy as np
    data = np.load("data/KQvK.npz")
    print(f"\nVerification:")
    print(f"  Positions: {len(data['x']):,}")
    print(f"  Encoding dims: {data['x'].shape[1]}")
    print(f"  WDL labels: {len(data['wdl']):,}")
    print(f"  DTZ labels: {len(data['dtz']):,}")

if __name__ == "__main__":
    test_parallel()
