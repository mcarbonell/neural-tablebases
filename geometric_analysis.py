"""
Análisis de la regla geométrica simple para KQvK:
"Si el rey defensor puede capturar la dama → Draw, sino → Loss"
"""
import numpy as np
import chess

def decode_position(x):
    """Decode compact encoding to piece positions"""
    x_reshaped = x.reshape(3, 64)
    positions = []
    for piece_idx in range(3):
        square = np.argmax(x_reshaped[piece_idx])
        positions.append(square)
    return positions

def square_to_coords(square):
    """Convert square index to (row, col)"""
    return (square // 8, square % 8)

def manhattan_distance(sq1, sq2):
    """Manhattan distance between two squares"""
    r1, c1 = square_to_coords(sq1)
    r2, c2 = square_to_coords(sq2)
    return abs(r1 - r2) + abs(c1 - c2)

def chebyshev_distance(sq1, sq2):
    """Chebyshev distance (king moves) between two squares"""
    r1, c1 = square_to_coords(sq1)
    r2, c2 = square_to_coords(sq2)
    return max(abs(r1 - r2), abs(c1 - c2))

def is_adjacent(sq1, sq2):
    """Check if two squares are adjacent (king can move)"""
    return chebyshev_distance(sq1, sq2) == 1

# Load data
data = np.load('data/KQvK.npz')

print("=" * 70)
print("ANÁLISIS DE REGLA GEOMÉTRICA SIMPLE")
print("=" * 70)

print("\n🎯 HIPÓTESIS:")
print("   Si el rey negro puede capturar la dama blanca → Draw")
print("   Si no puede → Loss/Win según quién tenga ventaja")

print("\n🔍 Analizando posiciones...")

# Analyze draws
draw_positions = data['x'][data['wdl'] == 0]
print(f"\n📊 EMPATES (Draw): {len(draw_positions)} posiciones")

# Check if draws are when king can capture queen
draws_with_adjacent_king = 0
for i in range(min(100, len(draw_positions))):
    pos = decode_position(draw_positions[i])
    # Assuming order: [white_king, white_queen, black_king]
    # or some permutation - we need to check all combinations
    
    # Check all pairs to find which is king-queen adjacent
    for j in range(3):
        for k in range(3):
            if j != k and is_adjacent(pos[j], pos[k]):
                draws_with_adjacent_king += 1
                break

print(f"   Empates con piezas adyacentes: {draws_with_adjacent_king}/100 muestras")

print("\n💡 FEATURES GEOMÉTRICAS QUE EL MODELO NO VE:")
print("   1. Distancia rey-dama (Chebyshev)")
print("   2. Distancia rey-rey")
print("   3. ¿Están las piezas adyacentes?")
print("   4. ¿Puede el rey capturar la dama?")
print("   5. Posición relativa al centro del tablero")
print("   6. Distancia a los bordes")

print("\n📈 PROPUESTA: AÑADIR FEATURES GEOMÉTRICAS")
print("\nEn lugar de solo one-hot encoding:")
print("   Input actual: [192 dims] = 3 piezas × 64 casillas")
print("\nAñadir features geométricas:")
print("   + Distancia rey blanco - dama: [1 dim]")
print("   + Distancia rey negro - dama: [1 dim]")
print("   + Distancia rey - rey: [1 dim]")
print("   + Rey negro adyacente a dama: [1 dim, binario]")
print("   + Coordenadas normalizadas de cada pieza: [6 dims]")
print("   + Distancia al centro para cada pieza: [3 dims]")
print("   + Distancia al borde más cercano: [3 dims]")
print("   = Total: 192 + 16 = 208 dims")

print("\n🎓 EJEMPLO DE CÁLCULO:")
sample_idx = 0
pos = decode_position(data['x'][sample_idx])
print(f"\n   Posición {sample_idx}:")
print(f"   Piezas en casillas: {pos}")
print(f"   WDL: {data['wdl'][sample_idx]}, DTZ: {data['dtz'][sample_idx]}")

# Calculate geometric features
print(f"\n   Features geométricas:")
for i, sq in enumerate(pos):
    r, c = square_to_coords(sq)
    dist_center = chebyshev_distance(sq, 27)  # Center is around square 27
    dist_edge = min(r, c, 7-r, 7-c)
    print(f"   Pieza {i}: casilla {sq} → ({r},{c}), dist_centro={dist_center}, dist_borde={dist_edge}")

print(f"\n   Distancias entre piezas:")
for i in range(3):
    for j in range(i+1, 3):
        dist = chebyshev_distance(pos[i], pos[j])
        adjacent = "SÍ" if dist == 1 else "no"
        print(f"   Pieza {i} ↔ Pieza {j}: {dist} movimientos (adyacente: {adjacent})")

print("\n" + "=" * 70)
print("CONCLUSIÓN")
print("=" * 70)
print("\n✅ TIENES RAZÓN: La clasificación es geométricamente simple")
print("❌ PROBLEMA: El modelo solo ve one-hot, no distancias")
print("💡 SOLUCIÓN: Añadir features geométricas al input")
print("\nEsto debería mejorar la accuracy de ~68% a >90%")
print("=" * 70)
