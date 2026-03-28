"""
Debug script for canonical forms.
"""
import chess
import sys
sys.path.append('src')

# Create a simple test
board1 = chess.Board()
board1.clear_board()
board1.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
board1.set_piece_at(chess.D1, chess.Piece(chess.QUEEN, chess.WHITE))
board1.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
board1.turn = chess.WHITE

print("Board 1 (original):")
print(board1)
print(f"FEN: {board1.fen()}")

# Create a rotated version (90 degrees clockwise)
board2 = chess.Board()
board2.clear_board()
# E1 (4,4) -> H1 (0,7) after 90° rotation: (r,c) -> (c,7-r)
board2.set_piece_at(chess.H1, chess.Piece(chess.KING, chess.WHITE))  # (0,7)
board2.set_piece_at(chess.G1, chess.Piece(chess.QUEEN, chess.WHITE))  # (0,6)
board2.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))  # (7,7)
board2.turn = chess.WHITE

print("\nBoard 2 (rotated 90°):")
print(board2)
print(f"FEN: {board2.fen()}")

# Test canonical forms
from canonical_forms import rotate_board, reflect_board_horizontal, get_all_symmetries

print("\nTesting rotation...")
rotated = rotate_board(board1, 1)  # 90°
print("Rotated 90°:")
print(rotated)
print(f"FEN: {rotated.fen()}")

print("\nAll symmetries:")
symmetries = get_all_symmetries(board1)
for i, sym in enumerate(symmetries):
    print(f"\nSymmetry {i}:")
    print(sym)

# Test encoding function
def simple_encoding(board):
    """Simple encoding for debugging."""
    pieces = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            pieces.append((piece.color, piece.piece_type, chess.square_rank(square), chess.square_file(square)))
    pieces.sort()
    return tuple(pieces)

print("\nTesting board_to_encoding_key...")
from canonical_forms import board_to_encoding_key

key1 = board_to_encoding_key(board1, simple_encoding)
key2 = board_to_encoding_key(board2, simple_encoding)

print(f"Key 1: {key1}")
print(f"Key 2: {key2}")
print(f"Keys equal? {key1 == key2}")

# Test find_canonical_form
from canonical_forms import find_canonical_form

print("\nTesting find_canonical_form...")
canonical1, transform1 = find_canonical_form(board1, simple_encoding)
canonical2, transform2 = find_canonical_form(board2, simple_encoding)

print(f"Canonical 1: {canonical1.fen()}")
print(f"Transform 1: {transform1}")
print(f"Canonical 2: {canonical2.fen()}")
print(f"Transform 2: {transform2}")
print(f"Canonicals equal? {canonical1.fen() == canonical2.fen()}")

# Check if they map to the same canonical form
key_c1 = board_to_encoding_key(canonical1, simple_encoding)
key_c2 = board_to_encoding_key(canonical2, simple_encoding)
print(f"Canonical keys equal? {key_c1 == key_c2}")