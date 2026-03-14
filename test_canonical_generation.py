"""
Test script for canonical forms dataset generation.
"""
import subprocess
import time
import os

def test_small_generation():
    """Test with a very small dataset."""
    print("Testing canonical forms dataset generation...")
    print("="*60)
    
    # Create test directory
    test_dir = "test_canonical"
    os.makedirs(test_dir, exist_ok=True)
    
    # Run with very small chunk size and limited combinations
    # We'll modify the script to only generate a few positions
    
    cmd = [
        "python", "src/generate_datasets_canonical.py",
        "--syzygy", "syzygy",
        "--data", test_dir,
        "--config", "KQvK",
        "--relative",
        "--move-distance",
        "--workers", "2",
        "--chunk-size", "100"  # Very small for testing
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run for a limited time
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✓ Generation completed in {elapsed:.1f}s")
            print(f"Output (first 500 chars):")
            print(result.stdout[:500])
            
            # Check if file was created
            output_file = os.path.join(test_dir, "KQvK_canonical.npz")
            if os.path.exists(output_file):
                print(f"\n✓ Dataset created: {output_file}")
                
                # Check size
                import numpy as np
                data = np.load(output_file)
                print(f"  Samples: {data['x'].shape[0]}")
                print(f"  Dimensions: {data['x'].shape[1]}")
                
                return True
            else:
                print(f"\n✗ Dataset file not created")
                return False
        else:
            print(f"\n✗ Generation failed in {elapsed:.1f}s")
            print(f"Error: {result.stderr[:500]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("\n✗ Generation timed out after 30 seconds")
        return False
    except Exception as e:
        print(f"\n✗ Exception: {e}")
        return False

def compare_with_original():
    """Compare canonical dataset with original."""
    print("\n" + "="*60)
    print("COMPARING WITH ORIGINAL DATASET")
    print("="*60)
    
    import numpy as np
    
    # Load original KQvK
    original_path = "data/KQvK_v2_fixed.npz"
    canonical_path = "test_canonical/KQvK_canonical.npz"
    
    if not os.path.exists(original_path):
        print(f"Original dataset not found: {original_path}")
        return
    
    if not os.path.exists(canonical_path):
        print(f"Canonical dataset not found: {canonical_path}")
        return
    
    # Load datasets
    original_data = np.load(original_path)
    canonical_data = np.load(canonical_path)
    
    print(f"Original KQvK: {original_data['x'].shape[0]:,} samples")
    print(f"Canonical KQvK: {canonical_data['x'].shape[0]:,} samples")
    
    if canonical_data['x'].shape[0] > 0:
        reduction = 1 - canonical_data['x'].shape[0] / original_data['x'].shape[0]
        print(f"Reduction: {reduction:.1%}")
        
        # Check dimensions
        if original_data['x'].shape[1] == canonical_data['x'].shape[1]:
            print(f"✓ Same dimensions: {original_data['x'].shape[1]}")
        else:
            print(f"✗ Different dimensions: {original_data['x'].shape[1]} vs {canonical_data['x'].shape[1]}")
        
        # Check WDL values
        original_wdl = np.unique(original_data['wdl'])
        canonical_wdl = np.unique(canonical_data['wdl'])
        
        if np.array_equal(original_wdl, canonical_wdl):
            print(f"✓ Same WDL values: {original_wdl}")
        else:
            print(f"✗ Different WDL values: {original_wdl} vs {canonical_wdl}")

def main():
    print("CANONICAL FORMS DATASET GENERATION TEST")
    print("="*60)
    
    # Clean up previous test
    import shutil
    if os.path.exists("test_canonical"):
        shutil.rmtree("test_canonical")
    
    # Run test
    success = test_small_generation()
    
    if success:
        compare_with_original()
        
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("1. Generate full KQvK canonical dataset:")
        print("   python src/generate_datasets_canonical.py --config KQvK --relative --move-distance")
        print("\n2. Train model on canonical dataset:")
        print("   python src/train.py --data_path data/KQvK_canonical.npz")
        print("\n3. Compare accuracy with original dataset")
    
    # Clean up
    if os.path.exists("test_canonical"):
        shutil.rmtree("test_canonical")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)