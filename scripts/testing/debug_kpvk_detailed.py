"""Debug KPvK dataset in detail"""
import numpy as np

data = np.load('data/KPvK.npz')
x = data['x']

print("Analyzing first 10 positions...")
for idx in range(min(10, len(x))):
    pos = x[idx]
    print(f"\nPosition {idx}:")
    
    for i in range(3):
        offset = i * 10
        row = pos[offset]
        col = pos[offset+1]
        piece_type_vec = pos[offset+2:offset+8]
        color_vec = pos[offset+8:offset+10]
        
        piece_idx = np.argmax(piece_type_vec)
        piece_names = ['K', 'Q', 'R', 'B', 'N', 'P']
        color_name = 'White' if color_vec[0] == 1.0 else 'Black'
        
        rank = int(row * 7)
        file = int(col * 7)
        
        print(f"  Piece {i+1}: {color_name} {piece_names[piece_idx]} at ({file}, {rank})")
        print(f"    Type vector: {piece_type_vec}")
        print(f"    Color vector: {color_vec}")

# Count unique piece combinations
print("\n" + "=" * 70)
print("Counting piece combinations...")

piece_combos = {}
for pos in x[:1000]:  # Sample first 1000
    pieces = []
    for i in range(3):
        offset = i * 10
        piece_type_vec = pos[offset+2:offset+8]
        color_vec = pos[offset+8:offset+10]
        
        piece_idx = np.argmax(piece_type_vec)
        piece_names = ['K', 'Q', 'R', 'B', 'N', 'P']
        color_name = 'W' if color_vec[0] == 1.0 else 'B'
        
        pieces.append(f"{color_name}{piece_names[piece_idx]}")
    
    combo = tuple(sorted(pieces))
    piece_combos[combo] = piece_combos.get(combo, 0) + 1

print("\nPiece combinations found:")
for combo, count in sorted(piece_combos.items(), key=lambda x: -x[1]):
    print(f"  {combo}: {count}")
