"""Analyze 4-piece endgames and recommend which to test"""
import os

print("=" * 70)
print("ANÁLISIS DE ENDGAMES DE 4 PIEZAS")
print("=" * 70)

# Check which 4-piece endgames exist in Syzygy
syzygy_path = "syzygy"
endgames_4piece = []

# List all .rtbz files with 4 pieces
for file in os.listdir(syzygy_path):
    if file.endswith('.rtbz'):
        name = file.replace('.rtbz', '')
        # Count pieces (excluding 'v')
        pieces = name.replace('v', '')
        if len(pieces) == 4:
            endgames_4piece.append(name)

# Sort by complexity
endgames_4piece.sort()

print(f"\n📊 Total endgames de 4 piezas: {len(endgames_4piece)}")

# Categorize by type
categories = {
    "Fáciles (siempre ganan)": [],
    "Interesantes (promoción)": [],
    "Complejos (material igual)": [],
    "Triviales (siempre tablas)": []
}

for eg in endgames_4piece:
    white, black = eg.split('v')
    
    # Trivial draws (insufficient material)
    if (white in ['KB', 'KN'] and black in ['KB', 'KN']) or \
       (white == 'KBB' and black == 'K') or \
       (white == 'KNN' and black == 'K'):
        categories["Triviales (siempre tablas)"].append(eg)
    
    # Easy wins (overwhelming material)
    elif (('QQ' in white or 'RR' in white or 'QR' in white) and black == 'K'):
        categories["Fáciles (siempre ganan)"].append(eg)
    
    # Interesting (pawn promotion)
    elif 'P' in white or 'P' in black:
        categories["Interesantes (promoción)"].append(eg)
    
    # Complex (equal or near-equal material)
    elif len(white) == len(black):
        categories["Complejos (material igual)"].append(eg)
    
    # Default to interesting
    else:
        categories["Interesantes (promoción)"].append(eg)

print("\n" + "=" * 70)
print("CATEGORÍAS")
print("=" * 70)

for category, endgames in categories.items():
    if endgames:
        print(f"\n{category}: {len(endgames)}")
        for eg in endgames[:10]:  # Show first 10
            print(f"  - {eg}")
        if len(endgames) > 10:
            print(f"  ... y {len(endgames) - 10} más")

print("\n" + "=" * 70)
print("RECOMENDACIONES PARA PROBAR")
print("=" * 70)

recommendations = [
    ("KQQvK", "Fácil", "Rey+2 Damas vs Rey - Siempre gana, muy fácil"),
    ("KRRvK", "Fácil", "Rey+2 Torres vs Rey - Siempre gana, fácil"),
    ("KQRvK", "Fácil", "Rey+Dama+Torre vs Rey - Siempre gana, fácil"),
    ("KQPvK", "Interesante", "Rey+Dama+Peón vs Rey - Promoción del peón"),
    ("KRPvK", "Interesante", "Rey+Torre+Peón vs Rey - Promoción del peón"),
    ("KPPvK", "Interesante", "Rey+2 Peones vs Rey - Doble promoción"),
    ("KQvKQ", "Complejo", "Rey+Dama vs Rey+Dama - Material igual, difícil"),
    ("KRvKR", "Complejo", "Rey+Torre vs Rey+Torre - Material igual, difícil"),
    ("KQvKP", "Interesante", "Rey+Dama vs Rey+Peón - Dama debe parar peón"),
]

print("\n🎯 Top 5 Recomendados (de fácil a difícil):")
print("\n1. KQQvK - Rey+2 Damas vs Rey")
print("   Dificultad: ⭐ Muy fácil")
print("   Razón: Material abrumador, siempre gana")
print("   Accuracy esperado: >99.9%")

print("\n2. KRRvK - Rey+2 Torres vs Rey")
print("   Dificultad: ⭐ Fácil")
print("   Razón: Material fuerte, siempre gana")
print("   Accuracy esperado: >99.9%")

print("\n3. KQPvK - Rey+Dama+Peón vs Rey")
print("   Dificultad: ⭐⭐ Media")
print("   Razón: Promoción del peón añade complejidad")
print("   Accuracy esperado: >99.5%")

print("\n4. KPPvK - Rey+2 Peones vs Rey")
print("   Dificultad: ⭐⭐⭐ Media-Alta")
print("   Razón: Doble promoción, bloqueos complejos")
print("   Accuracy esperado: >99%")

print("\n5. KQvKQ - Rey+Dama vs Rey+Dama")
print("   Dificultad: ⭐⭐⭐⭐ Difícil")
print("   Razón: Material igual, muchas tablas")
print("   Accuracy esperado: >98%")

print("\n" + "=" * 70)
print("SUGERENCIA")
print("=" * 70)

print("\n💡 Empezar con KQQvK o KRRvK:")
print("   - Valida que el encoding escala a 4 piezas")
print("   - Debería ser muy rápido (1-2 épocas)")
print("   - Confirma que el modelo funciona antes de casos difíciles")

print("\n💡 Luego probar KQPvK:")
print("   - Más interesante (promoción)")
print("   - Valida que maneja peones en 4 piezas")
print("   - Complejidad media")

print("\n💡 Si queremos un reto: KQvKQ")
print("   - Material igual, muchas tablas")
print("   - Prueba real de la capacidad del modelo")
print("   - Más épocas necesarias")

print("\n" + "=" * 70)

# Check file sizes
print("\nTAMAÑOS DE ARCHIVOS SYZYGY:")
print("(Para estimar complejidad del dataset)")
print()

import os
for eg in ["KQQvK", "KRRvK", "KQPvK", "KPPvK", "KQvKQ"]:
    rtbz = f"syzygy/{eg}.rtbz"
    if os.path.exists(rtbz):
        size = os.path.getsize(rtbz)
        print(f"  {eg:8s}: {size:8,} bytes ({size/1024:.1f} KB)")

print("\n" + "=" * 70)
