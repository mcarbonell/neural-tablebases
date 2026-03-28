"""
Test script for canonical forms with proper deduplication.
"""
import numpy as np
import chess
import sys
import os
sys.path.append('src')

from canonical_forms import find_canonical_form, board_to_encoding_key

def test_canonical_symmetry():
    """Test that symmetric positions map to the same canonical form."""
    print("Testing canonical forms for symmetric positions...")
    
    # Create a board with a specific position
    board1 = chess.Board()
    board1.clear_board()
    board1.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board1.set_piece_at(chess.D1, chess.Piece(chess.QUEEN, chess.WHITE))
    board1.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board1.turn = chess.WHITE
    
    # Create symmetric positions
    positions = []
    
    # Original position
    positions.append(board1.copy())
    
    # Create rotated versions
    # Rotate 90 degrees
    board2 = chess.Board()
    board2.clear_board()
    board2.set_piece_at(chess.H1, chess.Piece(chess.KING, chess.WHITE))
    board2.set_piece_at(chess.G1, chess.Piece(chess.QUEEN, chess.WHITE))
    board2.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
    board2.turn = chess.WHITE
    positions.append(board2)
    
    # Create a simple encoding function for testing
    def simple_encoding(board):
        pieces = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                pieces.append((piece.symbol(), square))
        pieces.sort()
        return tuple(pieces)
    
    # Test that all symmetric positions have the same canonical form
    canonical_keys = set()
    for i, board in enumerate(positions):
        canonical_board, transform = find_canonical_form(board, lambda b: simple_encoding(b))
        key = board_to_encoding_key(canonical_board, lambda b: simple_encoding(b))
        canonical_keys.add(key)
        print(f"Position {i+1}: {key}")
    
    if len(canonical_keys) == 1:
        print("✓ All symmetric positions map to the same canonical form")
        return True
    else:
        print(f"✗ Symmetric positions have {len(canonical_keys)} different canonical forms")
        return False

def test_canonical_duplicates():
    """Test that duplicate positions are properly deduplicated."""
    print("\nTesting duplicate elimination...")
    
    # Create two identical positions
    board1 = chess.Board()
    board1.clear_board()
    board1.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board1.set_piece_at(chess.D1, chess.Piece(chess.QUEEN, chess.WHITE))
    board1.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board1.turn = chess.WHITE
    
    # Create a symmetric position (should be canonicalized to same form)
    board2 = chess.Board()
    board2.clear_board()
    board2.set_piece_at(chess.H1, chess.Piece(chess.KING, chess.WHITE))
    board2.set_piece_at(chess.G1, chess.Piece(chess.QUEEN, chess.WHITE))
    board2.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
    board2.turn = chess.WHITE
    
    def simple_encoding(board):
        return str(sorted([(board.piece_at(sq), sq) for sq in chess.SQUARES if board.piece_at(sq)]))
    
    # Get canonical forms
    canonical1, _ = find_canonical_form(board1, simple_encoding)
    canonical2, _ = find_canonical_form(board2, simple_encoding)
    
    key1 = board_to_encoding_key(canonical1, simple_encoding)
    key2 = board_to_encoding_key(canonical2, simple_encoding)
    
    if key1 == key2:
        print("✓ Duplicate positions correctly identified as identical")
        return True
    else:
        print("✗ Duplicate positions not identified as identical")
        return False

def test_canonical_forms_integration():
    """Test the full canonical forms pipeline."""
    print("\nTesting canonical forms integration...")
    
    # Import the actual encoding function
    from generate_datasets import encode_board_relative
    
    # Create a test board
    board = chess.Board()
    board.clear_board()
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.D1, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE
    
    # Test encoding function
    def encoding_func(board):
        # Use relative encoding for canonical forms
        from generate_datasets import encode_board_relative
        return encode_board_relative(board, use_move_distance=False)
    
    # Get canonical form
    canonical_board, transform = find_canonical_form(board, encoding_func=encoding_func)
    
    # Verify canonical form is valid
    if canonical_board and transform:
        print("✓ Canonical form found successfully")
        
        # Check that the canonical form is valid
        canonical_key = board_to_encoding_key(canonical_board, encoding_func)
        print(f"Canonical key: {hash(canonical_key) % 10000:04d}")
        print(f"Transformation: rotation={transform['rotation']}, reflected={transform['reflected']}")
        return True
    else:
        print("✗ Failed to find canonical form")
        return False

def main():
    print("=" * 60)
    print("CANONICAL FORMS TEST SUITE")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Symmetry handling
    print("\n1. Testing symmetry handling...")
    if test_canonical_symmetry():
        print("✓ Symmetry test passed")
    else:
        print("✗ Symmetry test failed")
        all_passed = False
    
    # Test 2: Duplicate detection
    print("\n2. Testing duplicate detection...")
    if test_canonical_duplicates():
        print("✓ Duplicate detection test passed")
    else:
        print("✗ Duplicate detection test failed")
        all_passed = False
    
    # Test 3: Integration test
    print("\n3. Testing integration with encoding...")
    if test_canonical_forms_integration():
        print("✓ Integration test passed")
    else:
        print("✗ Integration test failed")
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)