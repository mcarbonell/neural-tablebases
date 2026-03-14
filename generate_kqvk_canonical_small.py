"""
Generate a small canonical KQvK dataset for testing.
"""
import sys
import os
sys.path.append('src')

from generate_datasets_canonical import generate_dataset_canonical

def main():
    print("Generating small canonical KQvK dataset...")
    print("="*60)
    
    # Create output directory
    output_dir = "data_canonical_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate with very limited scope
    # We'll modify the function to only process a small number of combinations
    
    # First, let's create a simplified version
    import chess
    import chess.syzygy
    import numpy as np
    from canonical_forms import find_canonical_form, board_to_encoding_key
    from generate_datasets import encode_board_relative
    
    print("Opening Syzygy tablebase...")
    tablebase = chess.syzygy.open_tablebase("syzygy")
    
    # Generate a few positions manually
    positions = []
    encodings = []
    wdl_values = []
    dtz_values = []
    canonical_keys = set()
    
    # Encoding function
    def encoding_func(board):
        return encode_board_relative(board, use_move_distance=True)
    
    # Generate positions more spread out
    test_positions = 100
    generated = 0
    
    print(f"\nGenerating up to {test_positions} positions...")
    
    # Use squares that are more likely to be valid
    # Kings shouldn't be adjacent, etc.
    white_king_squares = [chess.E1, chess.D1, chess.F1, chess.E2, chess.D2]
    white_queen_squares = [chess.D1, chess.E2, chess.F3, chess.C2, chess.D3]
    black_king_squares = [chess.E8, chess.D8, chess.F8, chess.E7, chess.D7]
    
    for wk_sq in white_king_squares:
        for wq_sq in white_queen_squares:
            for bk_sq in black_king_squares:
                if len({wk_sq, wq_sq, bk_sq}) == 3:  # All squares different
                    # Check if kings are not adjacent (illegal position)
                    wk_rank, wk_file = chess.square_rank(wk_sq), chess.square_file(wk_sq)
                    bk_rank, bk_file = chess.square_rank(bk_sq), chess.square_file(bk_sq)
                    
                    rank_diff = abs(wk_rank - bk_rank)
                    file_diff = abs(wk_file - bk_file)
                    
                    # Kings cannot be adjacent or on same square
                    if rank_diff <= 1 and file_diff <= 1:
                        continue  # Skip illegal position
                    
                    board = chess.Board()
                    board.clear_board()
                    board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
                    board.set_piece_at(wq_sq, chess.Piece(chess.QUEEN, chess.WHITE))
                    board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))
                    
                    # Test both turns
                    for turn in [chess.WHITE, chess.BLACK]:
                        board.turn = turn
                        
                        # Check basic validity
                        if board.is_valid():
                            try:
                                wdl = tablebase.probe_wdl(board)
                                dtz = tablebase.probe_dtz(board)
                                
                                # Get canonical form
                                canonical_board, transform = find_canonical_form(board, encoding_func)
                                key = board_to_encoding_key(canonical_board, encoding_func)
                                
                                # Only add if unique canonical key
                                if key not in canonical_keys:
                                    canonical_keys.add(key)
                                    
                                    # Encode canonical board
                                    encoding = encode_board_relative(canonical_board, use_move_distance=True)
                                    
                                    positions.append(canonical_board)
                                    encodings.append(encoding)
                                    wdl_values.append(wdl)
                                    dtz_values.append(dtz)
                                    
                                    generated += 1
                                    print(f"  Generated {generated}: WDL={wdl}, DTZ={dtz}, key={hash(key)%10000:04d}")
                                    
                                    if generated >= test_positions:
                                        break
                            
                            except Exception as e:
                                # Position not in tablebase or other error
                                pass
                
                if generated >= test_positions:
                    break
            if generated >= test_positions:
                break
        if generated >= test_positions:
            break
    
    tablebase.close()
    
    print(f"\nGenerated {generated} unique canonical positions.")
    
    if generated > 0:
        # Convert to arrays
        x_array = np.array(encodings, dtype=np.float32)
        wdl_array = np.array(wdl_values, dtype=np.int8)
        dtz_array = np.array(dtz_values, dtype=np.int16)
        
        print(f"\nDataset shape: {x_array.shape}")
        print(f"  Samples: {x_array.shape[0]}")
        print(f"  Dimensions: {x_array.shape[1]} (should be 64 for v2 fixed)")
        
        # Save
        output_path = os.path.join(output_dir, "KQvK_canonical_small.npz")
        np.savez_compressed(output_path, x=x_array, wdl=wdl_array, dtz=dtz_array)
        
        print(f"\n✓ Dataset saved to {output_path}")
        
        # Also save the positions for inspection
        positions_path = os.path.join(output_dir, "positions.txt")
        with open(positions_path, 'w') as f:
            for i, board in enumerate(positions):
                f.write(f"Position {i} (WDL={wdl_values[i]}, DTZ={dtz_values[i]}):\n")
                f.write(str(board))
                f.write("\n" + "-"*40 + "\n")
        
        print(f"✓ Positions saved to {positions_path}")
        
        # Quick analysis
        print("\n" + "="*60)
        print("DATASET ANALYSIS")
        print("="*60)
        
        unique_wdl = np.unique(wdl_array)
        print(f"WDL values: {unique_wdl}")
        
        # Check reduction factor
        # For these test positions, we don't have the original count
        # but we can estimate
        
        print("\nExpected for full KQvK:")
        print("  Original: ~64,631 positions")
        print("  Canonical: ~12,000-16,000 positions (75-80% reduction)")
        print("  Training speedup: 4-5x")
    
    else:
        print("✗ No positions generated!")
    
    return generated > 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)