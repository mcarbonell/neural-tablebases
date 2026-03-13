import os
import torch

print("=== Análisis de Modelos ===")
print()

# Listar archivos .pth
files = [f for f in os.listdir('data') if f.endswith('.pth')]
print(f"Archivos de modelo encontrados: {len(files)}")
print()

for f in files:
    path = os.path.join('data', f)
    size_mb = os.path.getsize(path) / (1024*1024)
    print(f"{f}: {size_mb:.2f} MB")
    
    # Intentar cargar y contar parámetros
    try:
        checkpoint = torch.load(path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else None
        
        if state_dict:
            total_params = sum(p.numel() for p in state_dict.values() if hasattr(p, 'numel'))
            print(f"  Parámetros: {total_params:,}")
    except Exception as e:
        print(f"  Error al cargar: {e}")
    print()

print("=== Análisis del Dataset ===")
import numpy as np
data = np.load('data/KQvK.npz')
print(f"Dataset KQvK.npz:")
print(f"  Posiciones: {len(data['x']):,}")
print(f"  Input shape: {data['x'].shape}")
print(f"  Input size: {data['x'].shape[1]} (equivale a {data['x'].shape[1]//64} piezas)")
print(f"  WDL range: [{data['wdl'].min()}, {data['wdl'].max()}]")
print(f"  DTZ range: [{data['dtz'].min()}, {data['dtz'].max()}]")
print()

# Distribución de WDL
unique, counts = np.unique(data['wdl'], return_counts=True)
print("Distribución de WDL:")
for u, c in zip(unique, counts):
    pct = c / len(data['wdl']) * 100
    print(f"  {u}: {c:,} ({pct:.1f}%)")
