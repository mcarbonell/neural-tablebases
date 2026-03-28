"""
Test canonical forms with actual encoding.
"""
import chess
import sys
import os
sys.path.append('.')

# Import encoding function
from src.generate_datasets import encode_board_relative

# Import canonical forms
from src.canonical_forms import find_canonical_form, board_to_canonical_key

def test_with_actual_encoding():
    """Test canonical forms with the actual relative encoding."""
    print("Testing canonical forms with actual encoding...")
    
    # Create test boards
    board1 = chess.Board()
    board1.clear_board()
    board1.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board1.set_piece_at(chess.D1, chess.Piece(chess.QUEEN, chess.WHITE))
    board1.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board1.turn = chess.WHITE
    
    # Create a symmetric position (rotated 90°)
    board2 = chess.Board()
    board2.clear_board()
    board2.set_piece_at(chess.H1, chess.Piece(chess.KING, chess.WHITE))
    board2.set_piece_at(chess.G1, chess.Piece(chess.QUEEN, chess.WHITE))
    board2.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
    board2.turn = chess.WHITE
    
    # Encoding function
    def encoding_func(board):
        return encode_board_relative(board, use_move_distance=False)
    
    print("Board 1 (original):")
    print(board1)
    encoding1 = encoding_func(board1)
    print(f"Encoding shape: {encoding1.shape}")
    
    print("\nBoard 2 (rotated 90°):")
    print(board2)
    encoding2 = encoding_func(board2)
    print(f"Encoding shape: {encoding2.shape}")
    
    # Get canonical forms
    canonical1, transform1 = find_canonical_form(board1, encoding_func)
    canonical2, transform2 = find_canonical_form(board2, encoding_func)
    
    print(f"\nCanonical 1: {canonical1.fen()}")
    print(f"Transform 1: {transform1}")
    print(f"Canonical 2: {canonical2.fen()}")
    print(f"Transform 2: {transform2}")
    
    # Check if they map to the same canonical form
    key1 = board_to_canonical_key(canonical1)
    key2 = board_to_canonical_key(canonical2)
    
    print(f"\nCanonical key 1: {hash(key1) % 10000:04d}")
    print(f"Canonical key 2: {hash(key2) % 10000:04d}")
    print(f"Keys equal? {key1 == key2}")
    
    # Check if encodings are the same
    encoding_c1 = encoding_func(canonical1)
    encoding_c2 = encoding_func(canonical2)
    
    print(f"\nEncoding canonical 1 shape: {encoding_c1.shape}")
    print(f"Encoding canonical 2 shape: {encoding_c2.shape}")
    print(f"Encodings equal? {np.array_equal(encoding_c1, encoding_c2)}")
    
    return key1 == key2

def test_multiple_symmetries():
    """Test that all 8 symmetries map to the same canonical form."""
    print("\n" + "="*60)
    print("Testing all 8 symmetries...")
    
    # Create original board
    board = chess.Board()
    board.clear_board()
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.D1, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE
    
    # Encoding function
    def encoding_func(board):
        return encode_board_relative(board, use_move_distance=False)
    
    # Get all symmetries
    from src.canonical_forms import get_all_symmetries
    symmetries = get_all_symmetries(board)
    
    print(f"Testing {len(symmetries)} symmetries...")
    
    canonical_keys = set()
    for i, sym_board in enumerate(symmetries):
        canonical_board, transform = find_canonical_form(sym_board, encoding_func)
        key = board_to_canonical_key(canonical_board)
        canonical_keys.add(key)
        
        print(f"Symmetry {i}: rotation={transform['rotation']}, reflected={transform['reflected']} -> key hash: {hash(key) % 10000:04d}")
    
    print(f"\nAll {len(symmetries)} symmetries map to {len(canonical_keys)} canonical form(s).")
    
    if len(canonical_keys) == 1:
        print("✓ SUCCESS: All symmetries have the same canonical form!")
        return True
    else:
        print("✗ FAIL: Symmetries have different canonical forms")
        return False

if __name__ == "__main__":
    import numpy as np
    
    print("="*60)
    print("CANONICAL FORMS WITH ACTUAL ENCODING TEST")
    print("="*60)
    
    test1 = test_with_actual_encoding()
    test2 = test_multiple_symmetries()
    
    print("\n" + "="*60)
    if test1 and test2:
        print("ALL TESTS PASSED ✓")
        exit(0)
    else:
        print("SOME TESTS FAILED")
        exit(1)