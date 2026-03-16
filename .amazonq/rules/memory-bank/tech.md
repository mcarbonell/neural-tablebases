# Tech: Technology Stack

## Languages & Versions
- Python >= 3.8 (type hints, `math.perm` used with fallback for 3.8)
- No compiled extensions; pure Python + C-backed libraries

## Core Dependencies
| Package | Version | Role |
|---------|---------|------|
| torch | >= 2.0 | Neural network training and inference |
| numpy | >= 1.20 | Array operations, `.npz` I/O |
| python-chess | >= 1.9 | Board representation, Syzygy probing |

## Key PyTorch Patterns
- `nn.Module` subclasses: `MLP`, `SIREN`, `SineLayer`
- `Dataset` / `DataLoader` with `batch_size=4096` default
- `torch.utils.data.random_split` for 90/10 train/val
- `optim.Adam` with `weight_decay=1e-5`
- `lr_scheduler.ReduceLROnPlateau(mode='max', patience=20, factor=0.7)`
- `torch.nn.utils.clip_grad_norm_(max_norm=1.0)` for stability
- `torch.load(map_location=device)` for CPU/GPU portability
- Models saved as `state_dict` (`.pth`), loaded with `model.load_state_dict()`

## Data Formats
- **`.npz`** (compressed): arrays `x` (float32), `wdl` (int8), `dtz` (int16)
- **`_metadata.json`**: UTF-8 JSON with all generation/training parameters
- **`_metadata.pkl`**: legacy pickle format (still read by `train.py` as fallback)
- **`.rtbw` / `.rtbz`**: Syzygy tablebase binary files (read via `chess.syzygy`)

## Parallelism
- `concurrent.futures.ProcessPoolExecutor` for dataset generation
- Workers capped at `min(cpu_count(), 8)` to avoid overhead
- Falls back to single-process on `PermissionError` (Windows restriction)
- Each worker opens its own `chess.syzygy.open_tablebase()` connection

## Development Commands

### Dataset Generation
```bash
# Full exhaustive dataset (recommended)
python src/generate_datasets_parallel.py --config KQvK --relative --enumeration permutation

# V4 encoding for pawn endgames
python src/generate_datasets_parallel.py --config KPvKP --relative --version 4 --enumeration permutation

# With canonical forms (pawn-safe auto mode)
python src/generate_datasets_parallel.py --config KPvKP --relative --version 4 --canonical --canonical-mode auto

# Smoke test (fast, non-exhaustive)
python src/generate_datasets_parallel.py --data data/smoke --config KQvK --relative --enumeration combination --limit-items 5000

# Pawn smoke test (avoid biased prefixes)
python src/generate_datasets_parallel.py --data data/smoke --config KPvKP --relative --version 4 --enumeration permutation --limit-items 5000 --shuffle-seed 42 --workers 1
```

### Training
```bash
python src/train.py --data_path data/KPvKP_canonical.npz --model mlp --epochs 30 --model_name mlp_kpvkp_v4_canonical
```

### Key Training Flags
| Flag | Default | Notes |
|------|---------|-------|
| `--model` | `mlp` | `mlp` or `siren` |
| `--epochs` | 1000 | with early stopping |
| `--batch_size` | 4096 | |
| `--lr` | 1e-3 | |
| `--patience` | 50 | early stopping epochs |
| `--hard_mining` | True | hard example re-training loop |
| `--hard_mining_freq` | 50 | epochs between hard mining passes |
| `--hard_weight` | 2.0 | weight multiplier for wrong predictions |
| `--wdl_classes` | 3 | 3 (standard) or 5 (with cursed/blessed) |
| `--model_name` | None | custom output filename prefix |

### Search / Evaluation
```bash
python src/search_poc.py --model data/mlp_best.pth --syzygy syzygy --config KQvK --samples 100 --depths 0,1,2
python scripts/analysis/analyze_models.py --model data/mlp_best.pth
```

## WDL Class Mapping
- **3-class**: -2→0 (Loss), 0→1 (Draw), 2→2 (Win)
- **5-class**: -2→0, -1→1 (Blessed loss), 0→2, 1→3 (Cursed win), 2→4

## Encoding Versions
| Version | Dims (3-piece) | Key Feature |
|---------|---------------|-------------|
| v1 | 43 | Normalized coords + piece type/color + pairwise distances |
| v2 (old) | 46 | + move distance (buggy) |
| v2.1 | 64 | + move distance (fixed) + relationship features |
| v4 | 45 | Perspective normalization (STM→White) + pawn progress |
