# Vanguard V10.1 Session Handover (2026-04-01)

## 1. Current State: Vanguard V10.1
- **Model**: `src/model/models_v10_1.py`
- **Train Script**: `src/train_v10_1.py`
- **Innovation**: **Weighted Relational Graphs**. 
  - Adjacency tensor expanded to 20 channels (16 logical + 4 geometric).
  - `VanguardWeightedRelLayerV10_1` uses a `Spatial Gate` to modulate node messages using `dx, dy, Manhattan, Chebyshev` distances.
- **Current Run**: `v10_1_kpvk_weighted_v1.log`
  - Epoca 0 Acc: 91.36%
  - Epoca 1 Acc (Batch 100): **94.43%** and rising.

## 2. Baseline Benchmarks (Stockfish Intuition - Depth 0)
Results documented in `docs/analysis_sf_intuition_4p.md`:
- **KPvKP**: 84.20% (Struggles with deep pawn opposition).
- **KPvKR**: 73.31% (Fatal errors in Torre vs Peón).
- **KRvKN**: 64.51% (Major blind spot in technical victories).
- **KBvKP**: 98.92% (High due to drawish nature of minor pieces).

## 3. Data Pipeline Improvements
- **Vectorized Data Loading**: `src/train_v10_1.py` uses NumPy to calculate geometric weights for 10k+ edges per batch without Python loops.
- **Architecture Logging**: Logs now mandatory show the full PyTorch topology and parameter count (1.18M params for V10.1).

## 4. Next Steps
1. **Validate V10.1 Final**: Check if it reaches the 99.9% Acc mark on KPvK.
2. **Transition to 4-Pieces**: Using the V10.1 architecture to train on `KPvKP` and `KPvKR` to solve the blind spots identified in Stockfish.
3. **Search Integration**: Test if the V10.1 + Minimax (1-2 ply) results in 100% Accuracy.
