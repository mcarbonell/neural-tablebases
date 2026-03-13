"""Analyze training details and model size"""
import numpy as np
import torch
import sys
sys.path.insert(0, 'src')
from models import get_model_for_endgame

print("=" * 70)
print("ANÁLISIS DETALLADO DEL ENTRENAMIENTO")
print("=" * 70)

# Load dataset
data = np.load('data/KQvK.npz')
total_positions = len(data['x'])
train_size = int(0.9 * total_positions)
val_size = total_positions - train_size

print(f"\nDATOS DEL DATASET")
print(f"   Total posiciones: {total_positions:,}")
print(f"   Train: {train_size:,} (90%)")
print(f"   Val: {val_size:,} (10%)")

# Training details
batch_size = 2048
batches_per_epoch = train_size // batch_size
positions_per_epoch = batches_per_epoch * batch_size

print(f"\nENTRENAMIENTO POR EPOCA")
print(f"   Batch size: {batch_size:,}")
print(f"   Batches por epoca: {batches_per_epoch}")
print(f"   Posiciones vistas por epoca: {positions_per_epoch:,}")
print(f"   Posiciones no vistas: {train_size - positions_per_epoch:,}")

# Accuracy before training (random)
num_classes = 3
random_accuracy = 1.0 / num_classes
print(f"\nACCURACY ANTES DE ENTRENAR")
print(f"   Random guess: {random_accuracy:.4f} ({random_accuracy*100:.2f}%)")
print(f"   Baseline (siempre Loss): 0.5452 (54.52%)")
print(f"   Despues de epoca 1: 0.9807 (98.07%)")
print(f"   Mejora en 1 epoca: +{(0.9807 - random_accuracy)*100:.2f}%")

# Model size comparison
print(f"\nTAMANO DEL MODELO")

# One-hot model
model_onehot = get_model_for_endgame('mlp', 3, 3, use_relative_encoding=False)
params_onehot = sum(p.numel() for p in model_onehot.parameters())
size_onehot_fp32 = params_onehot * 4  # 4 bytes per float32
size_onehot_fp16 = params_onehot * 2  # 2 bytes per float16
size_onehot_int8 = params_onehot * 1  # 1 byte per int8

# Relative model
model_relative = get_model_for_endgame('mlp', 3, 3, use_relative_encoding=True)
params_relative = sum(p.numel() for p in model_relative.parameters())
size_relative_fp32 = params_relative * 4
size_relative_fp16 = params_relative * 2
size_relative_int8 = params_relative * 1

print(f"\n   ONE-HOT ENCODING (192 dims):")
print(f"      Parametros: {params_onehot:,}")
print(f"      FP32: {size_onehot_fp32/1024/1024:.2f} MB")
print(f"      FP16: {size_onehot_fp16/1024:.2f} KB")
print(f"      INT8: {size_onehot_int8/1024:.2f} KB")

print(f"\n   RELATIVE ENCODING (43 dims):")
print(f"      Parametros: {params_relative:,}")
print(f"      FP32: {size_relative_fp32/1024/1024:.2f} MB")
print(f"      FP16: {size_relative_fp16/1024:.2f} KB")
print(f"      INT8: {size_relative_int8/1024:.2f} KB")

print(f"\n   REDUCCION:")
print(f"      Parametros: -{(params_onehot - params_relative):,} ({100*(params_onehot - params_relative)/params_onehot:.1f}%)")
print(f"      Tamano FP32: -{(size_onehot_fp32 - size_relative_fp32)/1024:.2f} KB")

# Syzygy comparison
syzygy_size = 10.4 * 1024 * 1024  # 10.4 MB in bytes
print(f"\nCOMPARACION CON SYZYGY")
print(f"   Syzygy KQvK: 10.4 MB")
print(f"   Neural (FP32): {size_relative_fp32/1024/1024:.2f} MB (compresion: {syzygy_size/size_relative_fp32:.1f}x)")
print(f"   Neural (FP16): {size_relative_fp16/1024:.2f} KB (compresion: {syzygy_size/size_relative_fp16:.1f}x)")
print(f"   Neural (INT8): {size_relative_int8/1024:.2f} KB (compresion: {syzygy_size/size_relative_int8:.1f}x)")

# Target
target_size = 250 * 1024  # 250 KB
print(f"\nOBJETIVO DEL PROYECTO")
print(f"   Target: < 250 KB")
print(f"   Actual (FP32): {size_relative_fp32/1024:.2f} KB {'OK' if size_relative_fp32 < target_size else 'X'}")
print(f"   Actual (FP16): {size_relative_fp16/1024:.2f} KB {'OK' if size_relative_fp16 < target_size else 'X'}")
print(f"   Actual (INT8): {size_relative_int8/1024:.2f} KB {'OK' if size_relative_int8 < target_size else 'X'}")

# Learning efficiency
print(f"\nEFICIENCIA DE APRENDIZAJE")
print(f"   Posiciones por epoca: {positions_per_epoch:,}")
print(f"   Accuracy despues de 1 epoca: 98.07%")
print(f"   Posiciones para 98%: {positions_per_epoch:,}")
print(f"   Accuracy por posicion: {98.07 / positions_per_epoch * 100:.6f}%")
print(f"   ")
print(f"   Con one-hot:")
print(f"      Epocas para 60%: 10")
print(f"      Posiciones vistas: {10 * positions_per_epoch:,}")
print(f"   ")
print(f"   Con relative:")
print(f"      Epocas para 98%: 1")
print(f"      Posiciones vistas: {positions_per_epoch:,}")
print(f"      Eficiencia: {10 * positions_per_epoch / positions_per_epoch:.0f}x mejor")

print("\n" + "=" * 70)
