"""Debug KPvK encoding"""
import numpy as np
import chess

data = np.load('data/KPvK.npz')

print("=" * 70)
print("DEBUG KPvK ENCODING")
print("=" * 70)

# Analyze first few positions
for idx in range(5):
    x = data['x'][idx]
    wdl = data['wdl'][idx]
    
    print(f"\n📍 Position {idx}: WDL={wdl}")
    
    # Decode pieces
    for piece_idx in range(3):
        offset = piece_idx * 10
        row = x[offset]
        col = x[offset + 1]
        type_vec = x[offset + 2:offset + 8]
        color_vec = x[offset + 8:offset + 10]
        
        piece_types = ['K', 'Q', 'R', 'B', 'N', 'P']
        piece_type = piece_types[np.argmax(type_vec)] if type_vec.sum() > 0 else '?'
        color = 'White' if color_vec[0] > 0.5 else 'Black'
        
        rank = int(row * 7)
        file = int(col * 7)
        
        print(f"   Piece {piece_idx+1}: {color} {piece_type} at ({row:.3f}, {col:.3f}) = rank {rank}, file {file}")
        
        # Check if pawn on invalid rank
        if piece_type == 'P' and (rank == 0 or rank == 7):
            print(f"      ⚠️  WARNING: Pawn on rank {rank}!")

print("\n" + "=" * 70)
print("CHECKING ACTUAL BOARD POSITIONS")
print("=" * 70)

# The issue might be in how we're interpreting the encoding
# Let's check what the actual configuration is
print("\nKPvK should be: White King + White Pawn vs Black King")
print("Let's verify the piece order...")

# Count piece types across dataset
white_kings = 0
white_pawns = 0
black_kings = 0

for idx in range(min(100, len(data['x']))):
    x = data['x'][idx]
    
    for piece_idx in range(3):
        offset = piece_idx * 10
        type_vec = x[offset + 2:offset + 8]
        color_vec = x[offset + 8:offset + 10]
        
        piece_type_idx = np.argmax(type_vec)
        is_white = color_vec[0] > 0.5
        
        if piece_type_idx == 0 and is_white:  # White King
            white_kings += 1
        elif piece_type_idx == 5 and is_white:  # White Pawn
            white_pawns += 1
        elif piece_type_idx == 0 and not is_white:  # Black King
            black_kings += 1

print(f"\nIn first 100 positions:")
print(f"   White Kings: {white_kings}")
print(f"   White Pawns: {white_pawns}")
print(f"   Black Kings: {black_kings}")
print(f"   Expected: 100 of each")

print("\n" + "=" * 70)
