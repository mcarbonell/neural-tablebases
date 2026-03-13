"""Analyze KPvK dataset"""
import numpy as np

data = np.load('data/KPvK.npz')

print("=" * 70)
print("ANÁLISIS DE KPvK (Rey+Peón vs Rey)")
print("=" * 70)

print(f"\n📊 DATASET")
print(f"   Total posiciones: {len(data['x']):,}")
print(f"   Input dims: {data['x'].shape[1]}")

# WDL distribution
wdl_unique, wdl_counts = np.unique(data['wdl'], return_counts=True)
print(f"\n🎯 WDL DISTRIBUTION")
for val, count in zip(wdl_unique, wdl_counts):
    pct = 100 * count / len(data['wdl'])
    label = {-2: "Loss", 0: "Draw", 2: "Win"}[val]
    print(f"   {label:5s} (WDL={val:2d}): {count:7d} ({pct:5.2f}%)")

# DTZ stats
print(f"\n⏱️  DTZ STATISTICS")
print(f"   Range: [{data['dtz'].min()}, {data['dtz'].max()}]")
print(f"   Mean: {data['dtz'].mean():.2f}")

# Analyze piece types
print(f"\n🔍 PIECE TYPE ANALYSIS")
x = data['x'][0]
print(f"   First position:")
print(f"   Piece 1: coords=({x[0]:.3f}, {x[1]:.3f}), type={x[2:8]}")
print(f"   Piece 2: coords=({x[10]:.3f}, {x[11]:.3f}), type={x[12:18]}")
print(f"   Piece 3: coords=({x[20]:.3f}, {x[21]:.3f}), type={x[22:28]}")

# Check if pawn is encoded correctly
piece_types = ['K', 'Q', 'R', 'B', 'N', 'P']
print(f"\n   Piece type mapping:")
for i in range(3):
    offset = i * 10
    type_vec = data['x'][0][offset+2:offset+8]
    if type_vec.sum() > 0:
        piece_idx = np.argmax(type_vec)
        print(f"   Piece {i+1}: {piece_types[piece_idx]}")

# Check pawn positions (should not be on rank 0 or 7)
print(f"\n🎲 PAWN POSITION ANALYSIS")
pawn_positions = []
for i in range(min(1000, len(data['x']))):
    x = data['x'][i]
    # Check each piece to find the pawn
    for piece_idx in range(3):
        offset = piece_idx * 10
        type_vec = x[offset+2:offset+8]
        if type_vec[5] == 1.0:  # Pawn is index 5
            row = x[offset]
            pawn_positions.append(row)
            break

if pawn_positions:
    pawn_positions = np.array(pawn_positions)
    print(f"   Pawn rows (normalized 0-1):")
    print(f"   Min: {pawn_positions.min():.3f} (rank {int(pawn_positions.min()*7)})")
    print(f"   Max: {pawn_positions.max():.3f} (rank {int(pawn_positions.max()*7)})")
    print(f"   Mean: {pawn_positions.mean():.3f}")
    
    # Check if any pawns on rank 0 or 7
    rank_0 = np.sum(pawn_positions < 0.1)
    rank_7 = np.sum(pawn_positions > 0.9)
    print(f"   Pawns on rank 0: {rank_0}")
    print(f"   Pawns on rank 7: {rank_7}")
    print(f"   ✓ Correctly filtered" if rank_0 == 0 and rank_7 == 0 else "   ✗ ERROR: Invalid pawns!")

print(f"\n💡 CARACTERÍSTICAS ESPECIALES DEL PEÓN")
print(f"   1. No puede estar en filas 1 u 8 (promoción)")
print(f"   2. Avanza solo hacia adelante")
print(f"   3. Puede promocionar → cambio de material")
print(f"   4. Posición relativa a fila 8 es crítica")

print("\n" + "=" * 70)
