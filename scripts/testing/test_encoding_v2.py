"""Test encoding v2 with move distance"""
import chess
import numpy as np
import sys
sys.path.append('src')
from generate_datasets import encode_board_relative, piece_move_distance

# Test piece_move_distance function
print("=" * 70)
print("TESTING PIECE MOVE DISTANCE")
print("=" * 70)

test_cases = [
    # (piece_type, from_sq, to_sq, expected_moves, description)
    (chess.KING, chess.A1, chess.H8, 7, "King a1-h8 (diagonal)"),
    (chess.QUEEN, chess.A1, chess.H8, 1, "Queen a1-h8 (diagonal)"),
    (chess.QUEEN, chess.A1, chess.B3, 2, "Queen a1-b3 (knight move)"),
    (chess.ROOK, chess.A1, chess.H8, 2, "Rook a1-h8 (corner to corner)"),
    (chess.ROOK, chess.A1, chess.A8, 1, "Rook a1-a8 (same file)"),
    (chess.ROOK, chess.A1, chess.H1, 1, "Rook a1-h1 (same rank)"),
    (chess.BISHOP, chess.A1, chess.H8, 1, "Bishop a1-h8 (diagonal)"),
    (chess.BISHOP, chess.A1, chess.H7, 10, "Bishop a1-h7 (different color)"),
    (chess.BISHOP, chess.A1, chess.C3, 1, "Bishop a1-c3 (same diagonal)"),
    (chess.KNIGHT, chess.A1, chess.B3, 1, "Knight a1-b3 (L-shape)"),
    (chess.KNIGHT, chess.A1, chess.C2, 1, "Knight a1-c2 (L-shape)"),
    (chess.KNIGHT, chess.A1, chess.A2, 3, "Knight a1-a2 (adjacent)"),
]

print("\nPiece Move Distance Tests:")
for piece_type, from_sq, to_sq, expected, desc in test_cases:
    result = piece_move_distance(piece_type, from_sq, to_sq)
    status = "OK" if abs(result - expected) < 0.1 else "FAIL"
    piece_name = chess.piece_name(piece_type).upper()
    print(f"{status:4s} {desc:40s} Expected: {expected:2.0f}, Got: {result:2.0f}")

# Test encoding dimensions
print("\n" + "=" * 70)
print("TESTING ENCODING DIMENSIONS")
print("=" * 70)

# Create test position: KQvK
board = chess.Board(None)
board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
board.set_piece_at(chess.D1, chess.Piece(chess.QUEEN, chess.WHITE))
board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
board.turn = chess.WHITE

# Encode with v1 (without move distance)
encoding_v1 = encode_board_relative(board, use_move_distance=False)
print(f"\nEncoding v1 (without move distance):")
print(f"  Dimensions: {len(encoding_v1)}")
print(f"  Expected: 43 (3 pieces)")
print(f"  Status: {'OK' if len(encoding_v1) == 43 else 'FAIL'}")

# Encode with v2 (with move distance)
encoding_v2 = encode_board_relative(board, use_move_distance=True)
print(f"\nEncoding v2 (with move distance):")
print(f"  Dimensions: {len(encoding_v2)}")
print(f"  Expected: 46 (3 pieces)")
print(f"  Status: {'OK' if len(encoding_v2) == 46 else 'FAIL'}")

# Show difference
print(f"\nDifference: {len(encoding_v2) - len(encoding_v1)} dims (move distances for 3 pairs)")

# Detailed breakdown
print("\n" + "=" * 70)
print("ENCODING BREAKDOWN")
print("=" * 70)

print("\nPer-piece features (30 dims, same in both versions):")
print("  Piece 1 (WK): coords(2) + type(6) + color(2) = 10 dims")
print("  Piece 2 (WQ): coords(2) + type(6) + color(2) = 10 dims")
print("  Piece 3 (BK): coords(2) + type(6) + color(2) = 10 dims")

print("\nPairwise features:")
print("  v1 (12 dims): 3 pairs × 4 dims (manhattan, chebyshev, dx, dy)")
print("  v2 (15 dims): 3 pairs × 5 dims (manhattan, chebyshev, move_dist, dx, dy)")

print("\nGlobal features (1 dim, same in both versions):")
print("  Side to move: 1 dim")

print("\nTotal:")
print(f"  v1: 30 + 12 + 1 = 43 dims")
print(f"  v2: 30 + 15 + 1 = 46 dims")

# Compare specific values
print("\n" + "=" * 70)
print("COMPARING SPECIFIC VALUES")
print("=" * 70)

print("\nPosition: White King e1, White Queen d1, Black King e8")
print("\nPair: White Queen (d1) <-> Black King (e8)")

# Extract pairwise features for WQ-BK pair (pair index 1: piece 1 ↔ piece 2)
# In v1: starts at index 30 + 4 = 34 (skip first pair)
# In v2: starts at index 30 + 5 = 35 (skip first pair)

# Actually, let's recalculate: pairs are (0,1), (0,2), (1,2)
# Pair (1,2) = WQ ↔ BK is the 3rd pair
# v1: index 30 + 2*4 = 38
# v2: index 30 + 2*5 = 40

idx_v1 = 30 + 2 * 4  # Third pair in v1
idx_v2 = 30 + 2 * 5  # Third pair in v2

print(f"\nv1 features (pair WQ-BK):")
print(f"  Manhattan: {encoding_v1[idx_v1]:.3f}")
print(f"  Chebyshev: {encoding_v1[idx_v1+1]:.3f}")
print(f"  dx: {encoding_v1[idx_v1+2]:.3f}")
print(f"  dy: {encoding_v1[idx_v1+3]:.3f}")

print(f"\nv2 features (pair WQ-BK):")
print(f"  Manhattan: {encoding_v2[idx_v2]:.3f}")
print(f"  Chebyshev: {encoding_v2[idx_v2+1]:.3f}")
print(f"  Move distance: {encoding_v2[idx_v2+2]:.3f} ← NEW")
print(f"  dx: {encoding_v2[idx_v2+3]:.3f}")
print(f"  dy: {encoding_v2[idx_v2+4]:.3f}")

# Calculate expected move distance for Queen d1 → e8
queen_moves = piece_move_distance(chess.QUEEN, chess.D1, chess.E8)
print(f"\nExpected Queen move distance d1-e8: {queen_moves:.0f} moves")
print(f"Normalized (÷10): {queen_moves/10:.3f}")
print(f"Encoded value: {encoding_v2[idx_v2+2]:.3f}")
print(f"Match: {'OK' if abs(encoding_v2[idx_v2+2] - queen_moves/10) < 0.01 else 'FAIL'}")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
