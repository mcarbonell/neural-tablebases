"""Verify KPvK dataset is correct"""
import numpy as np

data = np.load('data/KPvK.npz')
x = data['x']
wdl = data['wdl']
dtz = data['dtz']

print("=" * 70)
print("VERIFICACIÓN DATASET KPvK")
print("=" * 70)

print(f"\nTotal posiciones: {len(x):,}")
print(f"Input dimensions: {x.shape[1]}")

# Check encoding
if x.shape[1] == 43:
    print("✓ Encoding: Relativo (3 piezas)")
    
    # Analyze first position
    pos = x[0]
    print(f"\nEjemplo de encoding (primera posición):")
    print(f"  Pieza 1 (coords): [{pos[0]:.3f}, {pos[1]:.3f}]")
    print(f"  Pieza 1 (tipo): {pos[2:8]}")
    print(f"  Pieza 1 (color): {pos[8:10]}")
    print(f"  Pieza 2 (coords): [{pos[10]:.3f}, {pos[11]:.3f}]")
    print(f"  Pieza 2 (tipo): {pos[12:18]}")
    print(f"  Pieza 2 (color): {pos[18:20]}")
    print(f"  Pieza 3 (coords): [{pos[20]:.3f}, {pos[21]:.3f}]")
    print(f"  Pieza 3 (tipo): {pos[22:28]}")
    print(f"  Pieza 3 (color): {pos[28:30]}")
    
    # Count piece types
    piece_types = []
    colors = []
    for i in range(3):
        offset = i * 10
        piece_type_vec = pos[offset+2:offset+8]
        color_vec = pos[offset+8:offset+10]
        piece_idx = np.argmax(piece_type_vec)
        piece_names = ['K', 'Q', 'R', 'B', 'N', 'P']
        color_name = 'W' if color_vec[0] == 1.0 else 'B'
        piece_types.append(f"{color_name}{piece_names[piece_idx]}")
    
    print(f"\nPiezas detectadas: {piece_types}")
    
    # Count pawns across all positions
    pawn_count = 0
    for pos in x:
        for i in range(3):
            offset = i * 10
            piece_type_vec = pos[offset+2:offset+8]
            if piece_type_vec[5] == 1.0:  # Pawn
                pawn_count += 1
    
    print(f"\nTotal peones en dataset: {pawn_count:,}")
    print(f"Peones por posición: {pawn_count / len(x):.2f}")
    
    # Check pawn ranks
    pawn_ranks = []
    for pos in x:
        for i in range(3):
            offset = i * 10
            piece_type_vec = pos[offset+2:offset+8]
            if piece_type_vec[5] == 1.0:  # Pawn
                rank_normalized = pos[offset]
                rank = int(rank_normalized * 7)
                pawn_ranks.append(rank)
    
    print(f"\nDistribución de ranks de peones:")
    for rank in range(8):
        count = pawn_ranks.count(rank)
        if count > 0:
            print(f"  Rank {rank}: {count:,} ({count/len(pawn_ranks)*100:.1f}%)")
    
    if 0 in pawn_ranks or 7 in pawn_ranks:
        print("  ⚠️ ADVERTENCIA: Peones en rank 0 o 7 (inválido)")
    else:
        print("  ✓ Todos los peones en ranks válidos (1-6)")

else:
    print(f"✗ Encoding desconocido: {x.shape[1]} dims")

# WDL distribution
print(f"\nDistribución WDL:")
unique, counts = np.unique(wdl, return_counts=True)
for val, count in zip(unique, counts):
    wdl_name = {-2: "Loss", 0: "Draw", 2: "Win"}[val]
    print(f"  {wdl_name:5s} ({val:2d}): {count:6,} ({count/len(wdl)*100:.2f}%)")

# DTZ statistics
print(f"\nEstadísticas DTZ:")
print(f"  Min: {dtz.min()}")
print(f"  Max: {dtz.max()}")
print(f"  Mean: {dtz.mean():.2f}")
print(f"  Median: {np.median(dtz):.2f}")

# DTZ distribution by WDL
print(f"\nDTZ por WDL:")
for val in unique:
    mask = wdl == val
    dtz_subset = dtz[mask]
    wdl_name = {-2: "Loss", 0: "Draw", 2: "Win"}[val]
    print(f"  {wdl_name}: DTZ range [{dtz_subset.min()}, {dtz_subset.max()}], mean={dtz_subset.mean():.2f}")

print("\n" + "=" * 70)
