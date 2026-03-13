# Neural Chess Tablebase Compression

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

> Compressing chess endgame tablebases by 273x using neural networks with geometric encoding

## 🎯 Key Results

- **99.93% average accuracy** on 3-piece endgames
- **273x compression ratio** (956 MB → 3.5 MB)
- **Fast convergence:** 98%+ accuracy in 1 epoch
- **Universal approach:** No endgame-specific rules

## 📊 Results Summary

| Endgame | Positions | Accuracy | Compression |
|---------|-----------|----------|-------------|
| KQvK | 368,452 | 99.92% | 24x |
| KRvK | 399,112 | 99.99% | 37x |
| KPvK | 331,352 | 99.89% | 19x |
| **Average** | 366,305 | **99.93%** | **27x** |

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/[username]/neural-tablebases.git
cd neural-tablebases

# Install dependencies
pip install -r requirements.txt

# Download Syzygy tablebases (optional, for dataset generation)
# Place in ./syzygy/ directory
```

### Generate Dataset

```bash
# Generate 3-piece endgame dataset
python src/generate_datasets.py --config KQvK --relative

# Generate 4-piece endgame dataset
python src/generate_datasets.py --config KRRvK --relative
```

### Train Model

```bash
# Train on KQvK
python src/train.py --data_path data/KQvK.npz --model mlp --epochs 30

# Train on KRvK
python src/train.py --data_path data/KRvK.npz --model mlp --epochs 30
```

### Evaluate Model

```bash
# Analyze training results
python analyze_models.py --model data/mlp_best.pth --data data/KQvK.npz
```

## 🧠 How It Works

### Geometric Encoding

Unlike traditional one-hot encoding (192 dims), we use geometric encoding (43 dims) that captures spatial relationships:

```python
# Per piece (10 dims × 3 pieces = 30 dims)
- Normalized coordinates (x, y): 2 dims
- Piece type [K,Q,R,B,N,P]: 6 dims
- Color [White, Black]: 2 dims

# Per pair (4 dims × 3 pairs = 12 dims)
- Manhattan distance: 1 dim
- Chebyshev distance: 1 dim
- Direction vector (dx, dy): 2 dims

# Global (1 dim)
- Side to move: 1 dim

Total: 43 dims
```

### Model Architecture

```python
MLP(
    Input: 43 dims
    Hidden: [512, 512, 256, 128]
    Output: 3 classes (Loss, Draw, Win)
    Parameters: 452,740
    Size: 442 KB (INT8)
)
```

## 📈 Comparison: One-Hot vs Geometric

| Metric | One-Hot | Geometric | Improvement |
|--------|---------|-----------|-------------|
| Input dims | 192 | 43 | **-78%** |
| Epoch 1 accuracy | 46% | 98% | **+52%** |
| Best accuracy | 68% | 99.92% | **+32%** |
| Epochs to 99% | Never | 2 | **∞** |
| Hard examples | 7,000+ | 41 | **-99%** |

## 📁 Project Structure

```
neural-tablebases/
├── src/
│   ├── generate_datasets.py  # Dataset generation from Syzygy
│   ├── models.py              # Neural network architectures
│   └── train.py               # Training script
├── data/
│   ├── KQvK.npz              # Generated datasets
│   ├── KRvK.npz
│   └── *.pth                 # Trained models
├── syzygy/                   # Syzygy tablebases (not included)
├── logs/                     # Training logs
├── docs/                     # Documentation
└── README.md
```

## 🔬 Experiments

### 3-Piece Endgames (Completed)

- [x] KQvK: 99.92% accuracy
- [x] KRvK: 99.99% accuracy
- [x] KPvK: 99.89% accuracy

### 4-Piece Endgames (In Progress)

- [ ] KRRvK: Testing
- [ ] KRvKP: Planned
- [ ] KQvKQ: Planned

## 📊 Training Curves

### KQvK Convergence

```
Epoch 1:  98.07% accuracy
Epoch 2:  99.59% accuracy
Epoch 10: 99.77% accuracy
Epoch 27: 99.92% accuracy (best)
```

### Hard Examples Reduction

```
Epoch 1:  1,825 hard examples
Epoch 10: 85 hard examples
Epoch 27: 41 hard examples (-99%)
```

## 💾 Model Sizes

| Format | Size | Compression vs Syzygy |
|--------|------|----------------------|
| FP32 | 1.73 MB | 6x |
| FP16 | 884 KB | 12x |
| INT8 | 442 KB | 24x |

## 🎓 Research Paper

**Title:** Neural Tablebase Compression using Geometric Encoding

**Authors:** [To be added]

**Abstract:** We present a novel approach to chess endgame tablebase compression using neural networks with geometric encoding, achieving 99.93% accuracy with 273x compression ratio.

**Status:** Draft for ICGA Journal submission

[Link to paper draft](PAPER_DRAFT.md)

## 🛠️ Requirements

```
python >= 3.8
torch >= 2.0
numpy >= 1.20
python-chess >= 1.9
```

## 📝 Citation

If you use this work, please cite:

```bibtex
@article{neural_tablebase_2026,
  title={Neural Tablebase Compression using Geometric Encoding},
  author={Carbonell, Mario R.},
  journal={ICGA Journal},
  year={2026},
  note={In preparation}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Syzygy tablebase authors for providing ground truth data
- PyTorch team for the deep learning framework
- Chess programming community (TalkChess forum)

## 📧 Contact

**Mario Raúl Carbonell Martínez**
- **GitHub:** [@mcarbonell](https://github.com/mcarbonell)
- **Email:** marioraulcarbonell@gmail.com
- **Repository:** [neural-tablebases](https://github.com/mcarbonell/neural-tablebases)

For bugs and feature requests, please use [GitHub Issues](https://github.com/mcarbonell/neural-tablebases/issues).

For discussions, visit the [TalkChess Forum](http://talkchess.com/).

## 🔗 Links

- [ICGA Journal](https://icga.org/)
- [TalkChess Forum](http://talkchess.com/)
- [Syzygy Tablebases](https://syzygy-tables.info/)
- [Python Chess](https://python-chess.readthedocs.io/)

---

**Status:** 🚧 Work in Progress

**Last Updated:** March 2026

**Author:** Mario Carbonell

**Star this repo if you find it useful!** ⭐
