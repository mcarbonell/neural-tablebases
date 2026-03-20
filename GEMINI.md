# Gemini Context: Neural Chess Tablebases

This document provides a comprehensive overview of the `neural-tablebases` project for the Gemini assistant.

## 1. Project Overview

This is a Python-based research project focused on compressing chess endgame tablebases (like Syzygy) using neural networks. The primary goal is to achieve a massive compression ratio while maintaining near-100% accuracy for predicting endgame outcomes (Win/Draw/Loss) and Distance-to-Zero (DTZ).

### Key Concepts & Technologies

*   **Core Idea:** Instead of storing every possible position, the project trains a neural network to act as a function that can evaluate any given position.
*   **Geometric Encoding:** A novel, low-dimensional feature engineering approach is used to represent board states. This is a key innovation, far more efficient than traditional one-hot encodings. Several versions exist (v1, v2.1, v4), with v4 being a breakthrough for complex pawn endgames.
*   **Canonical Forms:** The system reduces the problem space by identifying the single "canonical" representation for all symmetrical equivalents of a position (rotations, reflections). This dramatically speeds up dataset generation.
*   **Neural Architectures:** The project primarily uses two types of models defined in `src/models.py`:
    *   `MLP`: A standard Multi-Layer Perceptron with BatchNorm and Dropout.
    *   `SIREN`: A Sinusoidal Representation Network, which is particularly good at fitting complex signals.
    Both models feature a dual-output head to predict WDL and DTZ simultaneously.
*   **Error Correction via Search:** A key finding is that a very shallow (1-2 ply) Minimax search dramatically improves accuracy. This technique, validated in `src/search_poc.py`, uses the neural network as an evaluation function but corrects its inconsistencies at runtime. It allows a smaller, less-than-perfect model to achieve 100% tablebase accuracy by trading a tiny amount of compute for a massive reduction in model size and error rate.
*   **Tech Stack:**
    *   Python (>= 3.8)
    *   PyTorch (>= 2.0)
    *   NumPy (>= 1.20)
    *   python-chess (>= 1.9)
    *   `torch-directml` for AMD GPU acceleration on Windows.

## 2. How to Build, Run, and Test

The project does not have a single build system. The workflow is centered around generating data, training models, and testing the components.

### 2.1. Environment Setup

It is recommended to use a Python virtual environment. A dedicated environment `venv_gpu` is intended for GPU-accelerated training.

```powershell
# To set up the GPU environment (for Windows/AMD DirectML)
py -3.12 -m venv venv_gpu
.\venv_gpu\Scripts\activate
pip install torch torchvision numpy torch-directml python-chess
```

For a CPU-only environment, omit `torch-directml`.

### 2.2. Dataset Generation

Datasets are generated from Syzygy tablebases (which must be available in the `syzygy/` directory) using a parallel script.

```bash
# Example: Generate a dataset for the KPvKP endgame using v4 encoding and canonical forms
python src/generate_datasets_parallel.py --config KPvKP --relative --version 4 --canonical --canonical-mode auto
```

This process creates `.npz` files in the `data/` directory, along with a corresponding `_metadata.json` file.

### 2.3. Model Training

The training script `src/train.py` is used to train a model on a generated dataset.

```bash
# Example: Train a model on the canonical KPvKP dataset
python src/train.py --data_path data/KPvKP_canonical.npz --model mlp --epochs 30 --model_name mlp_kpvkp_v4_canonical
```

To use GPU acceleration, run the script with the python executable from the `venv_gpu` environment.

```bash
# Example using the GPU venv
.\venv_gpu\Scripts\python.exe src/train.py --data_path data/KPvKP_canonical.npz
```

### 2.4. Testing

The project uses a custom test suite script, `run_tests.py`, to verify the correctness of core components. It does not use a standard framework like pytest.

```bash
# Run the test suite
python run_tests.py
```
This script checks encoding dimensions, board transformations, canonical form logic, model forward passes, and data loading.

## 3. Codebase Conventions & Structure

*   **`src/`**: Core Python source code.
    *   `generate_datasets_parallel.py`: The main script for creating datasets.
    *   `train.py`: The main script for training models.
    *   `models.py`: Defines the `MLP` and `SIREN` neural network architectures.
    *   `canonical_forms.py`: Logic for board symmetries and canonical representation.
    *   `encoding_*.py`: Files related to different encoding schemes.
*   **`data/`**: Default location for generated datasets (`.npz`) and trained models (`.pth`). This directory should be considered ephemeral and is ignored by Git. The `data/smoke` subdirectory contains small datasets suitable for testing.
*   **`docs/`**: Extensive project documentation, including the original design, analysis of results, and paper drafts. This is a key resource for understanding the project's evolution and rationale.
*   **`scripts/`**: Utility and analysis scripts.
*   **`run_tests.py`**: The primary testing script.
*   **`syzygy/`**: Intended location for the Syzygy tablebase files used as the source for dataset generation. This directory is ignored by Git.
*   **`venv_gpu/`**: A pre-configured virtual environment for GPU tasks.
