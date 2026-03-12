import numpy as np

# Load dataset
data = np.load('data/KQvK.npz')
wdl = data['wdl']

print("=== PROBLEMA IDENTIFICADO ===\n")

# WDL distribution
unique_wdl, counts = np.unique(wdl, return_counts=True)
print("Distribución WDL en el dataset:")
for val, count in zip(unique_wdl, counts):
    percentage = (count / len(wdl)) * 100
    print(f"  WDL {val:2d}: {count:7d} ({percentage:5.2f}%)")

print("\n¡PROBLEMA CRÍTICO!")
print("=" * 50)
print("El dataset está EXTREMADAMENTE DESBALANCEADO:")
print(f"  - Pérdidas (WDL=-2): 54.52%")
print(f"  - Empates  (WDL= 0):  6.26%")
print(f"  - Victorias(WDL= 2): 39.22%")
print(f"  - FALTAN: WDL=-1 y WDL=1 (0%)")
print("\nEsto significa que el dataset solo tiene 3 clases en lugar de 5.")
print("El modelo está intentando predecir 5 clases pero solo ve 3.")
print("\nPara KQvK (Rey+Dama vs Rey), esto tiene sentido:")
print("  - La mayoría son victorias o derrotas")
print("  - Pocos empates (posiciones de ahogado)")
print("  - Casi no hay posiciones con WDL=±1")

print("\n=== SOLUCIÓN ===")
print("Opciones:")
print("1. Cambiar el modelo a 3 clases (WDL: -2, 0, 2)")
print("2. Usar un endgame más complejo con todas las clases")
print("3. Ajustar el mapeo de clases en el dataset")
