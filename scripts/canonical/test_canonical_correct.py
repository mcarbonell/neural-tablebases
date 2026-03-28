"""
Test canonical forms with correct rotations.
"""
import chess
import sys
import os
sys.path.append('.')

# Import encoding function
from src.generate_datasets import encode_board_relative

# Import canonical forms
from src.canonical_forms import find_canonical_form, board_to_canonical_key, rotate_board

def test_correct_rotation():
    """Test with correct rotation calculation."""
    print("Testing with correct rotations...")
    
    # Create original board
    board1 = chess.Board()
    board1.clear_board()
    board1.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))  # (0,4)
    board1.set_piece_at(chess.D1, chess.Piece(chess.QUEEN, chess.WHITE))  # (0,3)
    board1.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))  # (7,4)
    board1.turn = chess.WHITE
    
    print("Board 1 (original):")
    print(board1)
    print(f"White K at E1: rank={chess.square_rank(chess.E1)}, file={chess.square_file(chess.E1)}")
    print(f"White Q at D1: rank={chess.square_rank(chess.D1)}, file={chess.square_file(chess.D1)}")
    print(f"Black k at E8: rank={chess.square_rank(chess.E8)}, file={chess.square_file(chess.E8)}")
    
    # Create proper 90° rotation
    board2 = rotate_board(board1, 1)  # 90° clockwise
    
    print("\nBoard 2 (rotated 90° clockwise):")
    print(board2)
    
    # Check piece positions after rotation
    # (r,c) → (c, 7-r)
    # E1 (0,4) → (4, 7) = H5
    # D1 (0,3) → (3, 7) = H4
    # E8 (7,4) → (4, 0) = A5
    
    # Find which squares have pieces
    for square in chess.SQUARES:
        piece = board2.piece_at(square)
        if piece:
            r = chess.square_rank(square)
            c = chess.square_file(square)
            print(f"{piece.symbol()} at {chess.square_name(square)}: rank={r}, file={c}")
    
    # Encoding function
    def encoding_func(board):
        return encode_board_relative(board, use_move_distance=False)
    
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
    
    print(f"\nCanonical key 1 hash: {hash(key1) % 10000:04d}")
    print(f"Canonical key 2 hash: {hash(key2) % 10000:04d}")
    print(f"Keys equal? {key1 == key2}")
    
    return key1 == key2

def test_simple_position():
    """Test with a simpler position that's easier to verify."""
    print("\n" + "="*60)
    print("Testing simpler position...")
    
    # Create a board with pieces in corners for easier rotation
    board1 = chess.Board()
    board1.clear_board()
    board1.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))  # (0,0)
    board1.set_piece_at(chess.H1, chess.Piece(chess.QUEEN, chess.WHITE))  # (0,7)
    board1.set_piece_at(chess.A8, chess.Piece(chess.KING, chess.BLACK))  # (7,0)
    board1.turn = chess.WHITE
    
    print("Board 1 (corners):")
    print(board1)
    
    # Rotate 90°
    board2 = rotate_board(board1, 1)
    
    print("\nBoard 2 (rotated 90°):")
    print(board2)
    
    # Encoding function
    def encoding_func(board):
        return encode_board_relative(board, use_move_distance=False)
    
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
    
    print(f"\nCanonical key 1 hash: {hash(key1) % 10000:04d}")
    print(f"Canonical key 2 hash: {hash(key2) % 10000:04d}")
    print(f"Keys equal? {key1 == key2}")
    
    return key1 == key2

if __name__ == "__main__":
    import numpy as np
    
    print("="*60)
    print("CANONICAL FORMS WITH CORRECT ROTATIONS")
    print("="*60)
    
    test1 = test_correct_rotation()
    test2 = test_simple_position()
    
    print("\n" + "="*60)
    if test1 and test2:
        print("ALL TESTS PASSED ✓")
        exit(0)
    else:
        print("SOME TESTS FAILED")
        exit(1)