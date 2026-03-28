"""
Create canonical dataset from existing KQvK dataset.
"""
import numpy as np
import chess
import sys
import os
sys.path.append('src')

from canonical_forms import find_canonical_form, board_to_encoding_key
from generate_datasets import encode_board_relative, encode_board

def main():
    print("Creating canonical dataset from existing KQvK...")
    print("="*60)
    
    # Load existing KQvK dataset
    data_path = "data/KQvK_v2_fixed.npz"
    if not os.path.exists(data_path):
        print(f"Dataset not found: {data_path}")
        return False
    
    print(f"Loading {data_path}...")
    data = np.load(data_path)
    x_data = data['x']
    wdl_data = data['wdl']
    dtz_data = data['dtz']
    
    print(f"Original dataset: {x_data.shape[0]:,} positions")
    
    # Take a small sample for testing
    sample_size = min(1000, x_data.shape[0])
    indices = np.random.choice(x_data.shape[0], sample_size, replace=False)
    
    x_sample = x_data[indices]
    wdl_sample = wdl_data[indices]
    dtz_sample = dtz_data[indices]
    
    print(f"Sampling {sample_size} positions for testing...")
    
    # Problem: We can't reconstruct boards from encoding easily
    # The encoding is lossy - we lose exact piece positions
    
    print("\nPROBLEM: Cannot reconstruct boards from encoding.")
    print("The encoding is lossy - we don't know exact piece positions.")
    
    print("\nSOLUTION: We need to modify the dataset generator to:")
    print("1. Save positions (not just encoding)")
    print("2. Apply canonical forms during generation")
    print("3. Only save canonical positions")
    
    print("\n" + "="*60)
    print("PLAN DE IMPLEMENTACIÓN")
    print("="*60)
    
    print("\n1. Modificar generate_datasets_parallel.py:")
    print("   - Añadir opción --canonical")
    print("   - Para cada posición válida:")
    print("     a. Calcular forma canónica")
    print("     b. Usar encoding de la forma canónica")
    print("     c. Solo guardar si es única")
    
    print("\n2. Generar KQvK canónico completo:")
    print("   python src/generate_datasets_parallel.py --config KQvK --relative --move-distance --canonical")
    
    print("\n3. Entrenar y comparar:")
    print("   - Accuracy vs dataset original")
    print("   - Tiempo de entrenamiento")
    print("   - Tamaño del modelo")
    
    print("\n4. Extender a otros endgames:")
    print("   - KRvK, KPvK, KRRvK")
    
    print("\n" + "="*60)
    print("ESTIMACIÓN DE BENEFICIOS")
    print("="*60)
    
    print("\nPara KQvK (64,631 posiciones):")
    print("  • Reducción esperada: 75-80%")
    print("  • Posiciones canónicas: ~12,000-16,000")
    print("  • Entrenamiento 4-5x más rápido")
    print("  • Modelo más pequeño (menos sobreajuste)")
    
    print("\nPara KRRvK (21.89M posiciones):")
    print("  • Reducción esperada: 75-80%")
    print("  • Posiciones canónicas: ~4.4-5.5M")
    print("  • Entrenamiento factible en GPU")
    print("  • Escalabilidad a 5-6 piezas")
    
    print("\n" + "="*60)
    print("PRÓXIMOS PASOS RECOMENDADOS")
    print("="*60)
    
    print("\n1. Implementar --canonical flag en generador")
    print("2. Generar KQvK canónico (proof-of-concept)")
    print("3. Entrenar y validar accuracy")
    print("4. Generar todos los endgames 3-piece")
    print("5. Extender a KRRvK (4-piece)")
    
    return True

if __name__ == "__main__":
    main()