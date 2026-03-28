"""
Final test for canonical forms dataset generation.
"""
import subprocess
import time
import os
import sys
import numpy as np

def test_canonical_generation_small():
    """Test canonical forms with a very small dataset."""
    print("Testing canonical forms dataset generation...")
    print("="*60)
    
    # Create test directory
    test_dir = "data_canonical_test"
    os.makedirs(test_dir, exist_ok=True)
    
    # Run with very small chunk size for quick testing
    cmd = [
        sys.executable, "src/generate_datasets_parallel_canonical.py",
        "--syzygy", "syzygy",
        "--data", test_dir,
        "--config", "KQvK",
        "--relative",
        "--canonical",
        "--workers", "2",
        "--chunk-size", "5000"  # Small for testing
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run for a limited time
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✓ Generation completed in {elapsed:.1f}s")
            
            # Check if file was created
            output_file = os.path.join(test_dir, "KQvK_canonical.npz")
            if os.path.exists(output_file):
                print(f"\n✓ Dataset created: {output_file}")
                
                # Load and check dataset
                data = np.load(output_file)
                print(f"  Samples: {data['x'].shape[0]:,}")
                print(f"  Dimensions: {data['x'].shape[1]}")
                
                # Check metadata
                metadata_file = output_file.replace('.npz', '_metadata.pkl')
                if os.path.exists(metadata_file):
                    import pickle
                    with open(metadata_file, 'rb') as f:
                        metadata = pickle.load(f)
                    print(f"\nMetadata:")
                    print(f"  Total valid positions: {metadata.get('total_valid', 'N/A'):,}")
                    print(f"  Unique positions: {metadata.get('unique_positions', 'N/A'):,}")
                    print(f"  Reduction: {metadata.get('reduction', 0):.1%}")
                    print(f"  Canonical forms: {metadata.get('canonical_forms', False)}")
                
                # Compare with original KQvK size
                original_file = "data/KQvK_v2_fixed.npz"
                if os.path.exists(original_file):
                    original_data = np.load(original_file)
                    print(f"\nOriginal KQvK: {original_data['x'].shape[0]:,} samples")
                    
                    if 'reduction' in metadata:
                        expected_reduction = metadata['reduction']
                        print(f"Expected reduction: {expected_reduction:.1%}")
                    
                    # Check if reduction is reasonable (should be ~87.5% for 8 symmetries)
                    reduction = 1 - data['x'].shape[0] / original_data['x'].shape[0]
                    print(f"Actual reduction: {reduction:.1%}")
                    
                    if reduction > 0.5:  # Should be significant
                        print("✓ Significant reduction achieved")
                    else:
                        print("⚠ Reduction lower than expected")
                
                return True
            else:
                print(f"\n✗ Dataset file not created")
                return False
        else:
            print(f"\n✗ Generation failed in {elapsed:.1f}s")
            print(f"Return code: {result.returncode}")
            if result.stdout:
                print(f"Stdout (last 500 chars):\n{result.stdout[-500:]}")
            if result.stderr:
                print(f"Stderr (last 500 chars):\n{result.stderr[-500:]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("\n✗ Generation timed out after 120 seconds")
        return False
    except Exception as e:
        print(f"\n✗ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_canonical_duplicates():
    """Verify that there are no duplicate canonical forms in the dataset."""
    print("\n" + "="*60)
    print("VERIFYING CANONICAL FORMS DEDUPLICATION")
    print("="*60)
    
    test_file = "data_canonical_test/KQvK_canonical.npz"
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return False
    
    data = np.load(test_file)
    positions = data['x']
    
    # Check for duplicates by comparing positions
    print(f"Checking {positions.shape[0]:,} positions for duplicates...")
    
    # Convert to list of tuples for hashing
    position_tuples = [tuple(pos.tolist()) for pos in positions]
    
    # Check for duplicates
    unique_positions = set(position_tuples)
    
    print(f"Unique positions: {len(unique_positions):,}")
    print(f"Total positions: {len(position_tuples):,}")
    
    if len(unique_positions) == len(position_tuples):
        print("✓ No duplicates found in canonical dataset")
        return True
    else:
        duplicates = len(position_tuples) - len(unique_positions)
        print(f"✗ Found {duplicates} duplicates in canonical dataset")
        return False

def main():
    print("FINAL CANONICAL FORMS TEST")
    print("="*60)
    
    # Clean up previous test
    import shutil
    if os.path.exists("data_canonical_test"):
        shutil.rmtree("data_canonical_test")
    
    # Run generation test
    success = test_canonical_generation_small()
    
    if success:
        # Verify deduplication
        verify_success = verify_canonical_duplicates()
        
        if verify_success:
            print("\n" + "="*60)
            print("NEXT STEPS:")
            print("="*60)
            print("1. Generate full canonical datasets:")
            print("   python src/generate_datasets_parallel_canonical.py --config KQvK --relative --canonical")
            print("   python src/generate_datasets_parallel_canonical.py --config KRvK --relative --canonical")
            print("   python src/generate_datasets_parallel_canonical.py --config KPvK --relative --canonical")
            print("\n2. Train models on canonical datasets:")
            print("   python src/train.py --data_path data/KQvK_canonical.npz")
            print("   python src/train.py --data_path data/KRvK_canonical.npz")
            print("   python src/train.py --data_path data/KPvK_canonical.npz")
            print("\n3. Compare accuracy with original datasets")
    
    # Clean up
    if os.path.exists("data_canonical_test"):
        shutil.rmtree("data_canonical_test")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)