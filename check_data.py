import numpy as np
import chess
import chess.syzygy

# Load dataset
data = np.load('data/KQvK.npz')

print("=" * 70)
print("ANÁLISIS DE DATOS DISPONIBLES")
print("=" * 70)

print("\n📦 Datos en el archivo:")
print(f"   Keys: {list(data.keys())}")
print(f"   x shape: {data['x'].shape}")
print(f"   wdl shape: {data['wdl'].shape}")
print(f"   dtz shape: {data['dtz'].shape}")

print("\n🎯 WDL (Win/Draw/Loss):")
wdl_unique, wdl_counts = np.unique(data['wdl'], return_counts=True)
for val, count in zip(wdl_unique, wdl_counts):
    pct = 100 * count / len(data['wdl'])
    label = {-2: "Loss (mate)", 0: "Draw", 2: "Win (mate)"}[val]
    print(f"   WDL {val:2d} ({label:15s}): {count:7d} ({pct:5.2f}%)")

print("\n⏱️  DTZ (Distance to Zero):")
print(f"   Range: [{data['dtz'].min()}, {data['dtz'].max()}]")
print(f"   Mean: {data['dtz'].mean():.2f}")
print(f"   Std: {data['dtz'].std():.2f}")

# Analyze DTZ distribution
dtz_by_wdl = {}
for wdl_val in wdl_unique:
    mask = data['wdl'] == wdl_val
    dtz_by_wdl[wdl_val] = data['dtz'][mask]
    print(f"\n   DTZ para WDL={wdl_val}:")
    print(f"      Range: [{dtz_by_wdl[wdl_val].min()}, {dtz_by_wdl[wdl_val].max()}]")
    print(f"      Mean: {dtz_by_wdl[wdl_val].mean():.2f}")

print("\n🔍 Ejemplos de posiciones:")
for i in range(10):
    print(f"   Pos {i}: WDL={data['wdl'][i]:2d}, DTZ={data['dtz'][i]:3d}")

print("\n💡 INTERPRETACIÓN:")
print("   - WDL: Win/Draw/Loss (resultado del juego con juego perfecto)")
print("   - DTZ: Distance to Zero (movimientos hasta captura/empate/mate)")
print("   - DTZ negativo: el oponente puede forzar el resultado")
print("   - DTZ positivo: nosotros podemos forzar el resultado")

print("\n🤔 TU OBSERVACIÓN:")
print("   'Si el rey puede comer la dama → Draw'")
print("   'Si no puede → Loss'")
print("   Esto es una regla GEOMÉTRICA simple que el modelo no ve!")

print("\n" + "=" * 70)
print("VERIFICACIÓN CON SYZYGY")
print("=" * 70)

# Check a few positions with Syzygy
try:
    tablebase = chess.syzygy.open_tablebase("syzygy")
    
    print("\nVerificando posiciones específicas...")
    
    # Decode a few positions and check
    for i in [0, 100, 1000]:
        x = data['x'][i]
        wdl_stored = data['wdl'][i]
        dtz_stored = data['dtz'][i]
        
        # Find piece positions
        x_reshaped = x.reshape(3, 64)
        pieces_squares = []
        for piece_idx in range(3):
            square = np.argmax(x_reshaped[piece_idx])
            pieces_squares.append(square)
        
        print(f"\nPosición {i}:")
        print(f"   Piezas en casillas: {pieces_squares}")
        print(f"   WDL almacenado: {wdl_stored}, DTZ almacenado: {dtz_stored}")
    
    tablebase.close()
    
except Exception as e:
    print(f"\nNo se pudo verificar con Syzygy: {e}")

print("\n" + "=" * 70)
