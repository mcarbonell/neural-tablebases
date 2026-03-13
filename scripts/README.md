# Scripts Directory

Utility scripts organized by purpose.

## 📊 Analysis Scripts

Located in `analysis/`:

- **analyze_3piece_endgames.py** - Analyze all 3-piece endgames
- **analyze_4piece_endgames.py** - Analyze all 4-piece endgames
- **analyze_kpvk.py** - Specific KPvK analysis
- **analyze_log.py** - Parse training logs
- **analyze_models.py** - Model comparison and analysis
- **analyze_problem.py** - Problem diagnosis
- **analyze_training_details.py** - Detailed training metrics
- **check_data.py** - Dataset integrity check
- **geometric_analysis.py** - Geometric encoding analysis
- **plot_training.py** - Generate training plots

### Usage Examples

```bash
# Analyze 3-piece endgames
python scripts/analysis/analyze_3piece_endgames.py

# Analyze 4-piece endgames
python scripts/analysis/analyze_4piece_endgames.py

# Plot training curves
python scripts/analysis/plot_training.py --log logs/train_mlp_*.log
```

## 🧪 Testing Scripts

Located in `testing/`:

- **test_encoding_v2.py** - Test encoding v2 with move distance
- **test_relative_encoding.py** - Test relative encoding
- **test_train.py** - Test training pipeline
- **debug_kpvk.py** - Debug KPvK dataset
- **debug_kpvk_detailed.py** - Detailed KPvK debugging
- **verify_dataset.py** - Verify dataset correctness
- **verify_kpvk.py** - Verify KPvK dataset
- **visualize_problem.py** - Visualize problematic positions

### Usage Examples

```bash
# Test encoding v2
python scripts/testing/test_encoding_v2.py

# Verify dataset
python scripts/testing/verify_dataset.py --data data/KQvK.npz

# Debug KPvK
python scripts/testing/debug_kpvk.py
```

## 🎓 Training Scripts

Located in `training/`:

- **train_improved.bat** - Windows training script
- **train_improved.sh** - Linux/Mac training script
- **test_multiple_endgames.bat** - Batch test multiple endgames

### Usage Examples

```bash
# Windows
scripts\training\train_improved.bat

# Linux/Mac
bash scripts/training/train_improved.sh

# Test multiple endgames
scripts\training\test_multiple_endgames.bat
```

## 📝 Script Categories

### Data Generation & Verification

- `check_data.py` - Check dataset integrity
- `verify_dataset.py` - Verify dataset correctness
- `verify_kpvk.py` - Verify KPvK specifically

### Analysis & Visualization

- `analyze_*.py` - Various analysis scripts
- `plot_training.py` - Generate plots
- `geometric_analysis.py` - Encoding analysis

### Testing & Debugging

- `test_*.py` - Test various components
- `debug_*.py` - Debug specific issues
- `visualize_problem.py` - Visualize problems

### Training & Execution

- `train_improved.*` - Training scripts
- `test_multiple_endgames.bat` - Batch testing

## 🔧 Common Tasks

### Analyze Training Results

```bash
# Parse training log
python scripts/analysis/analyze_log.py logs/train_mlp_20260312_234058.log

# Plot training curves
python scripts/analysis/plot_training.py --log logs/train_mlp_*.log

# Analyze model performance
python scripts/analysis/analyze_models.py --model data/mlp_best.pth
```

### Verify Datasets

```bash
# Check dataset integrity
python scripts/analysis/check_data.py

# Verify specific dataset
python scripts/testing/verify_dataset.py --data data/KQvK.npz

# Debug KPvK issues
python scripts/testing/debug_kpvk.py
```

### Test Encoding

```bash
# Test encoding v2
python scripts/testing/test_encoding_v2.py

# Test relative encoding
python scripts/testing/test_relative_encoding.py
```

## 📋 Script Dependencies

Most scripts require:
```
python >= 3.8
numpy
torch
python-chess
matplotlib (for plotting)
```

Install with:
```bash
pip install numpy torch python-chess matplotlib
```

---

**Navigation:**
- [← Back to Project Root](../README.md)
- [→ View Documentation](../docs/README.md)
