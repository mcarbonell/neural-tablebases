# Product: Neural Chess Tablebase Compression

## Purpose
Compress chess endgame tablebases (Syzygy format) using neural networks with geometric board encoding. The goal is to replace large binary tablebase files with small neural models that predict WDL (Win/Draw/Loss) and DTZ (Distance-to-Zero) outcomes with high accuracy.

## Key Results
- 99.93% average accuracy on 3-piece endgames (KQvK, KRvK, KPvK)
- 97.4% accuracy on 4-piece KPvKP with V4 encoding
- 79.7x compression ratio: 956 MB → ~12 MB (model + exception map)
- 100% accuracy achievable via 2-ply minimax search correction

## Core Capabilities

### Dataset Generation
- Exhaustive enumeration of all legal chess positions for a given endgame config
- Parallel generation using multiprocessing (6-7x speedup)
- Canonical form reduction via board symmetries (dihedral group / file mirror)
- Deterministic shuffled sampling for unbiased pawn endgame datasets
- Outputs `.npz` datasets + `_metadata.json` sidecar for reproducibility

### Neural Encoding (Geometric)
- **v1**: 43 dims (3-piece) — normalized coords + piece type/color + pairwise distances
- **v2.1**: 64 dims (3-piece) — adds move-distance and relationship features per pair
- **v4**: 45 dims (3-piece) / 68 dims (4-piece) — perspective normalization (STM→White) + pawn promotion progress; solves "race positions"

### Model Training
- Dual-head MLP or SIREN: WDL classification + DTZ regression
- Hard example mining loop for difficult positions
- Class-weighted CrossEntropy for imbalanced WDL distributions
- Automatic encoding detection from dataset input dimensions

### Search-Based Correction
- NeuralSearcher wraps the model with alpha-beta minimax
- Depth-0: pure neural; Depth-2: near-perfect patch for neural errors
- Falls back to Syzygy for material changes (captures/promotions)

## Target Users
- Chess engine researchers exploring neural compression of tablebases
- ML researchers studying geometric encoding for combinatorial game states
- Anyone interested in the intersection of deep learning and perfect-information games

## Endgame Coverage
| Endgame | Positions | Accuracy | Encoding |
|---------|-----------|----------|----------|
| KQvK    | 368,452   | 99.92%   | v1       |
| KRvK    | 399,112   | 99.99%   | v1       |
| KPvK    | 331,352   | 99.89%   | v1       |
| KPvKP   | ~6.2M     | 97.4%    | v4       |
| KRPvKP  | ~6.2M     | 94.1%    | v4       |
