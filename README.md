# Neural Chess Tablebase Compression

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> Compressing chess endgame tablebases and universal evaluation with **Relational Graph Neural Networks (RGNN)**, attention mechanisms, and triple-head outputs.

Neural Tablebases explores whether chess knowledge (both endgame technicality and middle-game intuition) can be represented with extreme efficiency using neural models.

## Current Snapshot (V8-Pro Era) 🧪🧬

- **Architecture**: Transitioned to **V8 Relational GNN** (RGNN) with Vectorized Message Passing and Global Attention Pooling.
- **Milestone**: Reached **99.83% Accuracy** on complex 4-piece endgames using the V8 architecture.
- **Universal Evaluation**: Implementation of the **Triple-Head V8-Pro**, predicting WDL, DTZ, and Stockfish-Eval (Centipawns) simultaneously.
- **Industry Pipeline**: High-speed sharding of the **Lichess Evaluation Dataset** (100M+ positions) using a Rust-backed tactical engine.

## Current Results

| Phase | Architecture | Targets | Status | Notes |
|-------|--------------|---------|--------|-------|
| **3-Piece** | V8 GNN | WDL + DTZ | **100.00%** | Solved with zero errors |
| **4-Piece** | V8 GNN | WDL + DTZ | **99.83%** | Competitive with Syzygy |
| **Universal**| **V8-Pro** | **WDL + DTZ + Eval** | **Active Learning** | Training on 100M+ Lichess positions |
| **Legacy** | V5 MLP | WDL | Stable | Kept for comparative analysis |

The repo still contains historical V1-V4 artifacts and reports. The current direction is V5-style relative encoding plus canonical datasets and search-based correction.

## Quick Start

### 1. Industry-Scale Generation (Lichess eval dataset)

Generate topological shards from Lichess ZST backups using the high-performance Rust engine:

```bash
# Process 100M+ positions with 12 workers
python src/generate_gnn_lichess.py --input data/lichess_db_eval.jsonl.zst --output_dir data/v8_shards --workers 12
```

### 2. Industry-Scale Training (V8-Pro Triple Head)

Train the flagship GNN model leveraging DirectML/AMD acceleration:

```bash
# Automated loading and logging included
python src/train_v8.py --data_dir data/v8_shards --model_name v8_pro_triple_head --batch_size 1024
```

## Encoding Evolution

- `v1` to `v5`: Historical geometric/relative encodings for flat MLPs.
- `V8 (Topological)`: Represents the board as a directed graph where:
    - **Nodes**: 64 squares with 128D learned embeddings.
    - **Edges**: 16 relational types (Attacks, Defenses, X-Rays, Pins, Checks) generated on-the-fly by the Rust movegen.
    - **Pooling**: Global Attention mechanism for dynamic feature aggregation.

## Documentation

### Start here

- [Project status](docs/README.md)
- [Live project snapshot](PROJECT_STATUS.md)
- [Scripts guide](scripts/README.md)

### Results and analysis

- [3-piece summary](docs/results/RESUMEN_3_PIEZAS.md)
- [Canonical forms results](docs/results/CANONICAL_FORMS_RESULTS.md)
- [Encoding analysis](docs/results/encoding_analysis.md)
- [Search correction analysis](docs/analysis/CORRECCION_ERRORES_POR_BUSQUEDA.md)

### Research and writing

- [Paper draft](docs/paper/PAPER_DRAFT.md)
- [TalkChess post draft](docs/paper/TALKCHESS_POST.md)

## Requirements

```text
python >= 3.8
torch >= 2.0
numpy >= 1.20
python-chess >= 1.9
```

## License

MIT License. See [LICENSE](LICENSE).

---

Last updated: March 28, 2026
Current focus: V8-Pro Triple Head training (100M+ Lichess positions), GNN-Search integration, ONNX export
