# Scripts Directory

Utility scripts for dataset generation, training, verification, and analysis.

## Current Guidance

- Prefer `src/generate_datasets_parallel.py` for real dataset generation.
- Prefer dataset/checkpoint metadata and active logs over older markdown summaries.
- The most active workflow in the repo is currently KPvKP canonical under `data/v5/`.

## Analysis Scripts

Located in `analysis/`:

- `analyze_3piece_endgames.py` - summarize 3-piece experiments
- `analyze_4piece_endgames.py` - summarize 4-piece experiments
- `analyze_kpvk.py` - KPvK-specific analysis
- `analyze_log.py` - parse training logs
- `analyze_models.py` - compare or inspect saved models
- `analyze_problem.py` - diagnose problem cases
- `analyze_training_details.py` - inspect detailed metrics
- `check_data.py` - basic dataset integrity checks
- `geometric_analysis.py` - geometric encoding analysis
- `plot_training.py` - plot training curves from logs

Examples:

```bash
python scripts/analysis/analyze_log.py logs/train_mlp_20260320_004934.log
python scripts/analysis/plot_training.py --log logs/train_mlp_*.log
python scripts/analysis/analyze_models.py --model data/mlp_best.pth
```

## Testing Scripts

Located in `testing/`:

- `test_encoding_v2.py` - legacy encoding-v2 checks
- `test_relative_encoding.py` - relative encoding checks
- `test_train.py` - training pipeline smoke test
- `debug_kpvk.py` - debug KPvK dataset/model behavior
- `debug_kpvk_detailed.py` - deeper KPvK debugging
- `verify_dataset.py` - verify dataset correctness
- `verify_kpvk.py` - KPvK-specific verification
- `visualize_problem.py` - inspect problematic positions visually

Examples:

```bash
python scripts/testing/verify_dataset.py --data data/v5/KPvKP_canonical.npz
python scripts/testing/debug_kpvk.py
```

## Training Scripts

Located in `training/`:

- `train_improved.bat` - Windows training helper
- `train_improved.sh` - Linux/macOS training helper
- `test_multiple_endgames.bat` - batch-style experiment helper
- `generate_parallel.bat` - wrapper for parallel dataset generation
- `train_sampled.py` - sampled training helper

Examples:

```powershell
scripts\training\generate_parallel.bat KPvKP
scripts\training\train_improved.bat
```

## Common Tasks

### Generate datasets

```bash
# Current canonical KPvKP path
python src/generate_datasets_parallel.py --config KPvKP --relative --version 5 --canonical --canonical-mode auto

# KRRvK canonical
python src/generate_datasets_parallel.py --config KRRvK --relative --canonical --canonical-mode auto

# Smoke test with deterministic shuffle
python src/generate_datasets_parallel.py --data data/smoke --config KPvKP --relative --version 5 --canonical --limit-items 5000 --shuffle-seed 42 --workers 1
```

### Train models

```bash
# Active KPvKP line
python src/train.py --data_path data/v5/KPvKP_canonical.npz --model mlp --epochs 1000 --model_name mlp_kpvkp_v5

# Older canonical datasets remain usable
python src/train.py --data_path data/KRRvK_canonical.npz --model mlp --epochs 200 --model_name mlp_krrvk_canonical
```

### Export and verify inference

```bash
python src/export_onnx.py --model_path data/mlp_best.pth --output_path data/kpvkp_v5_eval.onnx
python src/verify_search_correction.py
```

## Dependencies

Most scripts require:

```text
python >= 3.8
numpy
torch
python-chess
matplotlib
```

Install the basics with:

```bash
pip install numpy torch python-chess matplotlib
```

---

See also:
- [../README.md](../README.md)
- [../PROJECT_STATUS.md](../PROJECT_STATUS.md)
- [../docs/README.md](../docs/README.md)
