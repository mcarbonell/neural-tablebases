# Neural Tablebase Implementation Plan (3-piece Prototype)

This plan outlines the steps to create a prototype of the neural tablebase compressor, starting with simple 3-piece endgames. The goal is to evaluate different architectures (MLP, SIREN, KAN) for 100% accurate compression.

## Proposed Changes

### Data Generation
#### [NEW] [generate_datasets.py](file:///e:/neural-tablebases/src/generate_datasets.py)
A general-purpose script to generate datasets for any given endgame configuration using Syzygy tablebases. It will support:
- Canonicalization (symmetry reduction).
- WDL and DTZ/DTZ-bucket labels.
- Binary efficient storage (NumPy or HDF5).

### Model Implementation
#### [NEW] [models.py](file:///e:/neural-tablebases/src/models.py)
Implementation of various neural architectures:
- **Baseline MLP**: Simple dense layers with ReLU.
- **SIREN**: Sinusoidal Representation Networks for high-fidelity mapping.
- **KAN**: Kolmogorov-Arnold Networks for potential parameter reduction.

#### [NEW] [train.py](file:///e:/neural-tablebases/src/train.py)
Training script with:
- "The Overfitting Loop" (Hard Example Mining).
- Progress tracking (WDL accuracy, loss).
- Model checkpointing.

#### [NEW] [evaluate.py](file:///e:/neural-tablebases/src/evaluate.py)
Evaluation script to:
- Measure total accuracy.
- Identify exceptions for the "Residual Map".
- Calculate compression ratios.

## Verification Plan

### Automated Tests
- **Dataset Parity**: Compare labels in the generated dataset with direct Syzygy probes for a sample of positions.
- **Training Convergence**: Verify that for 3-piece endgames (very small datasets), the models can reach >99% accuracy quickly.
- **Inference Speed**: Benchmark inference time on CPU/GPU.

### Manual Verification
- Review the generated "Exception Map" size for a 3-piece endgame like KRvK.
- Verify that the combined "Model + Exception Map" provides 100% accuracy.
