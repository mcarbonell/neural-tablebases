# Neural Chess Tablebase Compression

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> Compressing chess endgame tablebases with neural networks, geometric encodings, canonical forms, and search-based correction.

Neural Tablebases explores whether Syzygy-style endgame information can be represented much more compactly with neural models while preserving practical accuracy and eventually reaching perfect play through correction mechanisms.

## Current Snapshot

- 3-piece endgames are mature and well documented.
- Canonical generation is integrated into the main dataset pipeline.
- The active 4-piece line is KPvKP with canonical data and relative encoding `v5`.
- The current training run on March 20, 2026 reached `val_acc = 0.9964` on KPvKP canonical data.
- KRRvK datasets already exist in both full and canonical form; it is no longer just a planned generation target.

## Current Results

| Endgame | Dataset | Encoding | Status | Notes |
|---------|---------|----------|--------|-------|
| KQvK | complete | geometric v1 / canonical variants | stable | 3-piece baseline |
| KRvK | complete | geometric v1 / canonical variants | stable | 3-piece baseline |
| KPvK | complete | v1 to v5 experiments | stable | search correction validated |
| KPvKP | complete canonical dataset | relative `v5` | active training | 7,436,088 canonical positions |
| KRPvKP | complete canonical dataset | relative v4 | exploratory | older 5-piece run kept for reference |
| KRRvK | full + canonical datasets present | relative variants | available | ready for training/comparison |

The repo still contains historical V1-V4 artifacts and reports. The current direction is V5-style relative encoding plus canonical datasets and search-based correction.

## Quick Start

### Generate a dataset

```bash
# Exhaustive parallel generation
python src/generate_datasets_parallel.py --config KQvK --relative --enumeration permutation

# KPvKP with current canonical pipeline
python src/generate_datasets_parallel.py --config KPvKP --relative --version 5 --canonical --canonical-mode auto

# Smoke test
python src/generate_datasets_parallel.py --data data/smoke --config KPvKP --relative --version 5 --canonical --limit-items 5000 --shuffle-seed 42 --workers 1
```

Each generated dataset writes a sibling `*_metadata.json` file for reproducibility.

### Train a model

```bash
python src/train.py --data_path data/v5/KPvKP_canonical.npz --model mlp --epochs 1000 --model_name mlp_kpvkp_v5
```

### Export ONNX for evaluation or search

```bash
python src/export_onnx.py --model_path data/mlp_best.pth --output_path data/kpvkp_v5_eval.onnx
```

## GPU Acceleration

Windows + AMD GPUs are supported through DirectML. A separate Python 3.12 environment is already used in this repo for GPU work:

```powershell
py -3.12 -m venv venv_gpu
.\venv_gpu\Scripts\activate
pip install torch torchvision numpy torch-directml
```

Then train with either:

```powershell
python train_canonical.py
```

or:

```powershell
.\venv_gpu\Scripts\python.exe src/train.py --data_path data/v5/KPvKP_canonical.npz
```

## Project Structure

```text
neural-tablebases/
|-- README.md
|-- PROJECT_STATUS.md
|-- src/
|   |-- generate_datasets.py
|   |-- generate_datasets_parallel.py
|   |-- models.py
|   |-- train.py
|   `-- export_onnx.py
|-- data/
|   |-- *.npz
|   |-- *.pth
|   `-- v5/
|-- logs/
|-- docs/
|   |-- README.md
|   |-- results/
|   |-- analysis/
|   |-- planning/
|   |-- vision/
|   `-- paper/
`-- scripts/
```

## Encoding Notes

- `v1`: early geometric baseline for 3-piece work.
- `v2` / `v2 fixed`: move-distance and relationship feature experiments.
- `v4`: pawn-race oriented encoding used heavily in early KPvKP and KRPvKP work.
- `v5`: current relative encoding branch used by the active canonical KPvKP dataset.

Some helper scripts still group dimensions `45/68/95` under shared V4/V5 handling because they reuse the same input sizes. The metadata file next to each dataset is the authoritative source for the dataset version.

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

Last updated: March 20, 2026
Current focus: KPvKP canonical V5 training, evaluation export, and documentation cleanup
