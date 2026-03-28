"""
Test script for canonical forms with parallel generator.
"""
import subprocess
import time
import os
import sys

def test_canonical_generation_small():
    """Test canonical forms with a very small dataset."""
    print("Testing canonical forms with parallel generator...")
    print("="*60)
    
    # Create test directory
    test_dir = "data_canonical_test"
    os.makedirs(test_dir, exist_ok=True)
    
    # Run with very small chunk size for quick testing
    cmd = [
        sys.executable, "src/generate_datasets_parallel.py",
        "--syzygy", "syzygy",
        "--data", test_dir,
        "--config", "KQvK",
        "--relative",
        "--canonical",
        "--workers", "2",
        "--chunk-size", "1000"  # Very small for testing
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run for a limited time
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✓ Generation completed in {elapsed:.1f}s")
            
            # Check if file was created
            output_file = os.path.join(test_dir, "KQvK_canonical.npz")
            if os.path.exists(output_file):
                print(f"\n✓ Dataset created: {output_file}")
                
                # Check size
                import numpy as np
                data = np.load(output_file)
                print(f"  Samples: {data['x'].shape[0]:,}")
                print(f"  Dimensions: {data['x'].shape[1]}")
                
                # Compare with original KQvK size
                original_file = "data/KQvK_v2_fixed.npz"
                if os.path.exists(original_file):
                    original_data = np.load(original_file)
                    print(f"\nOriginal KQvK: {original_data['x'].shape[0]:,} samples")
                    reduction = 1 - data['x'].shape[0] / original_data['x'].shape[0]
                    print(f"Reduction: {reduction:.1%}")
                
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
        print("\n✗ Generation timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"\n✗ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_canonical_forms_module():
    """Test that canonical_forms module works."""
    print("\n" + "="*60)
    print("TESTING CANONICAL FORMS MODULE")
    print("="*60)
    
    try:
        from src.canonical_forms import find_canonical_form, board_to_encoding_key
        
        # Create a test board
        import chess
        board = chess.Board()
        board.clear_board()
        board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.D1, chess.Piece(chess.QUEEN, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        board.turn = chess.WHITE
        
        # Simple encoding function for testing
        def simple_encoding(board):
            pieces = []
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece:
                    pieces.append((piece.symbol(), square))
            pieces.sort()
            return str(pieces)
        
        # Test canonical form
        canonical_board, transform_info = find_canonical_form(board, simple_encoding)
        canonical_key = board_to_encoding_key(canonical_board, simple_encoding)
        
        print(f"Original board: {board.fen()}")
        print(f"Canonical board: {canonical_board.fen()}")
        print(f"Transform info: {transform_info}")
        print(f"Canonical key: {hash(canonical_key) % 10000:04d}")
        
        print("\n✓ canonical_forms module works correctly")
        return True
        
    except Exception as e:
        print(f"\n✗ canonical_forms module error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("CANONICAL FORMS PARALLEL GENERATOR TEST")
    print("="*60)
    
    # Clean up previous test
    import shutil
    if os.path.exists("data_canonical_test"):
        shutil.rmtree("data_canonical_test")
    
    # Test canonical forms module first
    module_ok = test_canonical_forms_module()
    
    if not module_ok:
        print("\n✗ Cannot proceed without canonical_forms module")
        return False
    
    # Run small generation test
    success = test_canonical_generation_small()
    
    if success:
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("1. Generate full KQvK canonical dataset:")
        print("   python src/generate_datasets_parallel.py --config KQvK --relative --canonical")
        print("\n2. Train model on canonical dataset:")
        print("   python src/train.py --data_path data/KQvK_canonical.npz")
        print("\n3. Compare accuracy with original dataset")
    
    # Clean up
    if os.path.exists("data_canonical_test"):
        shutil.rmtree("data_canonical_test")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)