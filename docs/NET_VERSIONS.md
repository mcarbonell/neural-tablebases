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

## [ACTIVE] Vanguard V10: Geometric Fusion
*   **Architecture**: Relational GNN (V9) + **Perspective Normalization** + **Pawn Progress Features**.
*   **Key Innovation**: Normalizes all positions to "White-to-move" perspective and injects explicit rank-based urgency for pawns.
*   **Results (KPvK)**:
    *   **Accuracy**: **96.50%** (Significant jump from V9's 94.1%).
    *   **DTZ MAE**: **0.92 moves**.
    *   **Convergence**: Reached 94% in less than 1 epoch.
*   **Status**: Frozen. Architecture validated as superior for spatial urgency.

---

### [NEW] Vanguard V10.1: Weighted Relational Graphs
*   **Goal**: Bridge the cap to 99.9% accuracy in KPvK by matching the information density of the classic MLP.
*   **Architecture**: Vanguard V10 + **Weighted Adjacency**.
*   **Key Innovation**: Instead of binary connections (0 or 1), edges will carry **Geometric Weights** representing relative distances (dx, dy, Manhattan, Chebyshev).
*   **Hypothesis**: By providing the exact relative distance between pieces on the edges, the GNN can learn precise thresholding rules for promotion/opposition without relying on square-coordinate inference.

---

## [IDEAS] Future Versions (V11+)
- **Attention Modules**: Transformer Blocks para pesar la importancia de cada pieza en el grafo.
- **Sparse Encoding**: Integrar técnicas de Stockfish NNUE para mayor velocidad de inferencia.
