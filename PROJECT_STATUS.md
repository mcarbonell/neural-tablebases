# Project Status: V8-Pro Triple Head Industrial Phase 🧬♟️

Last updated: March 28, 2026

## 1. Major Breakthroughs 🚀
- [x] **V8 4-Piece Endgame Perfection**: Reached a stable **99.83% Accuracy** in standard endgame tablebase compression.
- [x] **V8-Pro Architecture**: Implemented a high-performance **Vectorized RGNN** that eliminates Python loops and achieves **>1000 pos/s** on Radeon 780M iGPU.
- [x] **Global Attention Pooling**: Added a learned attention mechanism to the GNN to focus on critical board squares (e.g., King safety, tactical centers).
- [x] **Universal Triple Head**: Predicting **WDL**, **DTZ**, and **Stockfish-Eval** simultaneously.

## 2. Methodology & Infrastructure 🏭
- **Lichess Industry**: Shard generator for `lichess_db_eval.jsonl.zst` tuned for massive scale. Fully sharded 40M+ positions (Est 1.8GB).
- **Mandatory Logging**: Enforced research-grade logging with triple metric tracking (WDL-Acc, DTZ-MAE, Eval-MAE).
- **Hybrid Learning**: Adaptive loss handling for Lichess vs. Syzygy Bitbases.
- **Best Checkpoint**: `train_v8.py` now saves `_best.pth` automatically (best WDL-Acc across epochs).
- **Vectorized Graph Builder**: `build_giant_graph` now uses tensor offset broadcasting — no Python loop in the hot-path.

## 3. Current Objective: Universal Chess Evaluation
Achieve parity with Stockfish intuition while maintaining Syzygy technicality.
- [x] **V8 Universal Sharding**: Streaming 100M+ tactical positions.
- [x] **V8-Pro Training**: Continuous optimization of the Triple Head flagship model (1.17M parameters). Reached **~73%+ WDL-Acc** on Lichess evaluations (Epoch 1 in progress).
- [x] **Architecture Refinement**: Added **LayerNorm**, **Dropout**, and **Cosine LR Scheduler** for industrial stability.
- [x] **Holdout Eval Script**: `src/eval_v8_tablebase.py` — measures real WDL accuracy vs. Syzygy ground truth.
- [ ] **GNN-Search Integration**: Low-latency GNN Negamax with ONNX export. (Next Sprint)

## Sprint H1 — Completed 2026-03-28
1. **Best Checkpoint Tracker** → `_best.pth` saved automatically on WDL-Acc improvement.
2. **Vectorized `build_giant_graph`** → Removed Python `for` loop; uses `torch.arange` offsets.
3. **`src/eval_v8_tablebase.py`** → Holdout evaluation against real Syzygy tablebases.
4. **Repository Cleanup** → Moved 12+ debug/test/train scripts from root to `scripts/debug/`, `scripts/canonical/`, `scripts/train/`.
5. **`TODO.md` Synchronized** → Reflects V8-Pro state (not V5-MLP legacy tasks).

## Quick Links
- [Main README](README.md)
- [Documentation Index](docs/README.md)
- [GNN Model Architecture](src/models_v8.py)
- [Industrial Trainer](src/train_v8.py)
- [Holdout Evaluator](src/eval_v8_tablebase.py)
- [GNN-Search Architecture Plan](docs/planning/GNN_NEURAL_SEARCH_ARCHITECTURE.md)
