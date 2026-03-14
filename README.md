# Neural Chess Tablebase Compression

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Compression](https://img.shields.io/badge/compression-79.7x-green.svg)](https://github.com/mcarbonell/neural-tablebases)
[![Accuracy](https://img.shields.io/badge/accuracy-99.93%25-brightgreen.svg)](https://github.com/mcarbonell/neural-tablebases)

> Compressing chess endgame tablebases by 80x using neural networks with geometric encoding

## 🎯 Quick Results

- **99.93% average accuracy** on 3-piece endgames
- **97.4% accuracy** on 4-piece endgames (KPvKP) with **V4 Encoding**
- **V4 Encoding**: Perspective normalization + Pawn promotion progress features
- **79.7x compression ratio** (956 MB → 12 MB with exception maps)
- **Parallel Generation**: 6-7x faster dataset creation with canonical forms
- **Search-based correction**: 100% accuracy in 3-piece endgames with 2-ply search

## 📊 Current Status

### Completed Experiments

| Endgame | Positions | Accuracy | Encoding | Status |
|---------|-----------|----------|----------|--------|
| KQvK | 368,452 | 99.92% | v1 | ✅ |
| KRvK | 399,112 | 99.99% | v1 | ✅ |
| KPvK | 331,352 | 99.89% | v1 | ✅ |
| KPvKP | ~6.2M | 97.4% | **v4** | ✅ |
| KRPvKP | ~6.2M | 94.1% | **v4** | 🔄 Training |

### Breakthroughs

- **V4 Encoding**: Solved "race positions" by providing explicit pawn progress features.
- **Minimax Search**: Shallow 2-ply search acts as a perfect patch for neural evaluation errors.

## 🚀 Quick Start

### Generate Dataset

```bash
# Exhaustive, parallel (recommended)
python src/generate_datasets_parallel.py --config KQvK --relative --enumeration permutation

# V4 encoding (KPvKP). `--turns auto` defaults to white_only because V4 normalizes STM
python src/generate_datasets_parallel.py --config KPvKP --relative --version 4 --enumeration permutation

# Canonical forms (pawn-safe auto mode)
python src/generate_datasets_parallel.py --config KPvKP --relative --version 4 --canonical --canonical-mode auto

# Fast smoke-test (non-exhaustive)
python src/generate_datasets_parallel.py --data data/smoke --config KQvK --relative --enumeration combination --limit-items 5000

# Fast smoke-test for pawn endgames: avoid biased prefixes with deterministic shuffling
python src/generate_datasets_parallel.py --data data/smoke --config KPvKP --relative --version 4 --enumeration permutation --limit-items 5000 --shuffle-seed 42 --workers 1
```

The generator always writes a `*_metadata.json` next to the `.npz`. When `--limit-items` and/or `--item-offset` are used, the output file name includes a `_partial_...` suffix to avoid overwriting full datasets.

### Train Model

```bash
python src/train.py --data_path data/KPvKP_canonical.npz --model mlp --epochs 30 --model_name mlp_kpvkp_v4_canonical
```

### Analyze Results

```bash
python scripts/analysis/analyze_models.py --model data/mlp_best.pth
```

## 📁 Project Structure

```
neural-tablebases/
├── README.md                    # This file
├── src/                         # Core source code
│   ├── generate_datasets.py     # Dataset generation
│   ├── generate_datasets_parallel.py  # Parallel version (experimental)
│   ├── models.py                # Neural network architectures
│   └── train.py                 # Training script
├── data/                        # Generated datasets and models
│   ├── *.npz                    # Datasets
│   └── *.pth                    # Trained models
├── logs/                        # Training logs
├── syzygy/                      # Syzygy tablebases (not included)
├── docs/                        # Documentation
│   ├── paper/                   # Paper drafts and publications
│   │   ├── PAPER_DRAFT.md       # ICGA Journal draft
│   │   ├── GITHUB_README.md     # GitHub repository README
│   │   └── TALKCHESS_POST.md    # TalkChess forum post
│   ├── results/                 # Experimental results
│   │   ├── RESUMEN_3_PIEZAS.md  # 3-piece summary
│   │   ├── FINAL_RESULTS.md     # Complete results
│   │   └── *.md                 # Individual experiment results
│   ├── analysis/                # Technical analysis
│   │   ├── ANALISIS_KRVKP.md    # KRvKP analysis
│   │   ├── MEJORA_DISTANCIA_MOVIMIENTO.md  # Move distance improvement
│   │   └── *.md                 # Other analyses
│   └── planning/                # Project planning
│       └── *.md                 # Plans and design docs
└── scripts/                     # Utility scripts
    ├── analysis/                # Analysis scripts
    ├── testing/                 # Test scripts
    └── training/                # Training scripts
```

## 🧠 How It Works

### Geometric Encoding (v1)

For 3 pieces: **43 dimensions**

```
Per piece (10 dims × 3 = 30):
  - Normalized coordinates (x, y): 2 dims
  - Piece type [K,Q,R,B,N,P]: 6 dims
  - Color [White, Black]: 2 dims

Per pair (4 dims × 3 = 12):
  - Manhattan distance: 1 dim
  - Chebyshev distance: 1 dim
  - Direction (dx, dy): 2 dims

Global (1 dim):
  - Side to move: 1 dim
```

### Geometric Encoding v2.1 (with move distance + pair features)

For 3 pieces: **64 dimensions** (adds move-distance + relationship features)

```
Per pair (11 dims × 3 = 33):
  - Manhattan distance: 1 dim
  - Chebyshev distance: 1 dim
  - Direction (dx, dy): 2 dims
  - Weighted move distance (piece-specific): 1 dim  ← NEW
  - Relationship features (ally/enemy/king-king/etc): 6 dims  ← NEW
```

**Move distance** captures how many moves a piece needs to reach another square:
- Rook: 1-2 moves
- Bishop: 1 move (diagonal), ∞ (different color)
- Knight: Unique distances
- Pawn: Forward only

## 📈 Key Results

### Comparison: One-Hot vs Geometric

| Metric | One-Hot | Geometric | Improvement |
|--------|---------|-----------|-------------|
| Input dims | 192 | 43 | -78% |
| Epoch 1 accuracy | 46% | 98% | +52% |
| Best accuracy | 68% | 99.92% | +32% |
| Epochs to 99% | Never | 2 | ∞ |
| Hard examples | 7,000+ | 41 | -99% |

### Model Size

| Format | Size | Compression vs Syzygy |
|--------|------|----------------------|
| FP32 | 1.73 MB | 6x |
| FP16 | 884 KB | 12x |
| INT8 | 442 KB | 24x |

## 📚 Documentation

### For Researchers

- **[Paper Draft](docs/paper/PAPER_DRAFT.md)** - ICGA Journal submission
- **[Complete Results](docs/results/FINAL_RESULTS.md)** - All experimental results
- **[3-Piece Summary](docs/results/RESUMEN_3_PIEZAS.md)** - Detailed 3-piece analysis

### For Developers

- **[GitHub README](docs/paper/GITHUB_README.md)** - Repository documentation
- **[Move Distance Analysis](docs/analysis/MEJORA_DISTANCIA_MOVIMIENTO.md)** - Encoding v2 details
- **[Optimization Guide](docs/analysis/OPTIMIZACION_GENERADOR.md)** - Performance optimization

### For Community

- **[TalkChess Post](docs/paper/TALKCHESS_POST.md)** - Forum discussion draft
- **[DTZ Analysis](docs/analysis/RESPUESTA_DTZ.md)** - Distance-to-zero explanation

## 🔬 Experiments

### Completed (3-piece)

- ✅ KQvK: 99.92% accuracy
- ✅ KRvK: 99.99% accuracy
- ✅ KPvK: 99.89% accuracy

### In Progress (4-piece)

- 🔄 KRRvK: Dataset generating (54%)
- ⏭️ KRvKP: Planned (asymmetric, tactical)
- ⏭️ KQvKQ: Planned (material equal, complex)

## 🛠️ Requirements

```
python >= 3.8
torch >= 2.0
numpy >= 1.20
python-chess >= 1.9
```

## 📝 Citation

```bibtex
@article{neural_tablebase_2026,
  title={Neural Tablebase Compression using Geometric Encoding},
  author={Carbonell, Mario R.},
  journal={ICGA Journal},
  year={2026},
  note={In preparation}
}
```

## 👤 Author

**Mario Raúl Carbonell Martínez**
- Email: marioraulcarbonell@gmail.com
- GitHub: [github.com/mcarbonell/neural-tablebases](https://github.com/mcarbonell/neural-tablebases)

## 📧 Contact

- **Issues:** [GitHub Issues](https://github.com/mcarbonell/neural-tablebases/issues)
- **Discussion:** TalkChess Forum
- **Email:** marioraulcarbonell@gmail.com

## 📜 License

MIT License - See LICENSE file for details

---

**Last Updated:** March 13, 2026  
**Status:** Active Development  
**Current Focus:** 4-piece endgames validation  
**Author:** Mario Carbonell
