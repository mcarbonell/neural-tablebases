# Tablebase Neural Network Versions (NET_VERSIONS)

Este documento registra la evolución de las arquitecturas neuronales del proyecto `neural-tablebases`. Su objetivo es evitar la pérdida de conocimiento técnico y asegurar la reproducibilidad de los resultados.

---

## [ARCHIVED] Vanguard V8 (35M Parameters)
- **Status**: Perdida / Restauración Fallida (Marzo 2026).
- **Archivo**: `data/v8_universal_35M_latest.pth`
- **Arquitectura**: GNN Relacional (4 capas).
- **Dimensiones**: Embed(32) + Coords(16) + Tac(4) = 52 Input. Hidden 128.
- **Configuración de Entrada (Teórica)**:
  - Coordenadas: `[-1, 0]`.
  - Orden: `[Embed, Coord, Tac]`.
  - Perspectiva: STM-Relative (Verified diagonal).
- **Resultados Post-Restauración**: ~40% Accuracy (Corrupción de pesos o incompatibilidad de aristas tácticas).
- **Lección**: No sobreescribir definiciones de modelos. Mantener `models_v{N}.py` inmutables.

---

## [FINALIZED] Vanguard V9 (Relational Sparse Fusion)
- **Status**: Congelada (Marzo 2026). Arquitectura de referencia en GNN pura.
- **Archivo Model**: `src/model/models_v9.py`
- **Resultados**:
    - **KRvK (Rook)**: data\logs\train_v9_krvk_max_throttle_v1.log
2026-04-01 00:02:33 | Epoch 5 | Batch 20 | Loss: 0.0003 | Acc: 1.0000 | MAE: 3.34 | Speed: 337 pos/s | LR: 1.00e-03    
    - **KQvK (Queen)**: data\logs\train_v9_kqvk_full_v1.log
00:05:38 | Epoch 2 | Batch 80 | Loss: 0.0001 | Acc: 1.0000 | MAE: 2.36 | Speed: 404 pos/s | LR: 1.00e-03    
    - **KPvK (Pawn)**: (Encuentra el "techo de cristal" de la GNN pura). data\logs\train_v9_kpvk_full_v2.log
00:30:31 | Epoch 3 | Batch 1290 | Loss: 0.0646 (+0.0098) | Acc: 0.9681 (+0.0001) | MAE: 0.92 (-0.00) | Speed: 369 pos/s | LR: 1.00e-03
00:30:33 | --- EPOCA 3 FINALIZADA ---
00:30:33 |   Accuracy: 0.9681
00:30:33 |   DTZ-MAE: 0.92 moves    
- **Benchmark vs Stockfish 16.1**: scripts\benchmark_sf_kpvk_intuition.py
    - SF Intuition (Depth 0): **62.67% Acc**.
    - V9 Advantage: +31.45% de precisión estática que el mejor motor del mundo en finales de peones.
- **Lección**: La GNN pura es 150 veces más lenta de entrenar que una MLP, pero su "intuición" táctica inicial es masivamente superior a los métodos clásicos.

---

## [ACTIVE] Vanguard V10 (Geometric Fusion GNN)
- **Status**: En Desarrollo (Hibridación GNN + Geometría).
- **Archivo Model**: `src/model/models_v10.py`
- **Innovación**: Integra las lecciones de la V4 (MLP Geométrica) en la arquitectura V9.
- **Diferenciadores**:
    1. **Perspective Normalization**: `flip_board` implementado en el pipeline de datos. La red siempre ve el tablero desde la perspectiva del bando que mueve (White-to-move). Eficiencia de entrenamiento 2x.
    2. **Pawn Urgency Feature**: Inyección de `pawn_rank / 7.0` como feature de nodo. Resuelve la ceguera de la GNN ante la urgencia de la coronación.
    3. **3D Batch Matmul**: Optimización de tensores `[B*16, 64, 64]` para máxima saturación de GPUs AMD (DirectML).

---

## [IDEAS] Future Versions (V11+)
- **Attention Modules**: Transformer Blocks para pesar la importancia de cada pieza en el grafo.
- **Sparse Encoding**: Integrar técnicas de Stockfish NNUE para mayor velocidad de inferencia.
