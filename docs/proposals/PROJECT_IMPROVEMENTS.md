# Project Improvement Proposals: Neural Tablebase Compression

Following a comprehensive analysis of the current project state, this document outlines strategic technical proposals to enhance performance, scalability, and efficiency.

## 1. High-Performance Dataset Generation

### 1.1 Robust Multiprocessing Implementation
**Context:** Current generation for 4-piece endgames takes ~15 hours on a single thread.
**Proposal:** 
- Convert the experimental `generate_datasets_parallel.py` into the primary generation engine.
- Use a `ProcessPoolExecutor` with a shared workload queue.
- **Chunk-based writing:** Instead of gathering all results in memory, each worker process should write its results to a temporary binary file or a shared `Queue` that a dedicated writer process commits to disk.

### 1.2 Generator Sampling for 5+ Piece Endgames
**Context:** Exhaustive enumeration for 5-piece endgames (~10^8 positions) is computationally expensive.
**Proposal:**
- Implement a `--sample-rate` flag in `generate_datasets.py`.
- Use a deterministic seed for reproducibility.
- Focus sampling on "interesting" regions of the state space (e.g., positions near the edges or with pieces in close proximity).

## 2. Memory and Storage Optimization

### 2.1 Memory-Efficient Data Loading (`np.memmap`)
**Context:** Datasets like KRRvK consume 6-8 GB of RAM.
**Proposal:**
- Transition from loading entire `.npz` files into memory to using `np.memmap`.
- This allows training on datasets that exceed available RAM by paging data from disk on-the-fly.
- Change `src/train.py` to handle `memmap` arrays for the `Dataset` class.

### 2.2 Compressed Data Format
**Context:** Raw `.npz` files are large.
**Proposal:**
- Explore a custom binary format where each position is encoded as a bitboard/bitstream before feature extraction.
- Apply `Zstandard` or `LZ4` compression on the dataset chunks to reduce disk I/O during training.

## 3. Advanced Neural Architectures and Training

### 3.1 SIREN vs. MLP Ablation Study
**Context:** Both architectures are implemented, but 3-piece results primarily rely on MLP.
**Proposal:**
- Conduct a formal comparison using SIREN for 4-piece endgames. SIREN's periodic activation functions are often superior at representing sharp boundaries (like "distance-to-mate" transitions).
- Implement a learning rate scheduler specifically for SIREN (e.g., cyclical learning rates).

### 3.2 Automated Quantization-Aware Training (QAT)
**Context:** The goal is a <250 KB INT8 model.
**Proposal:**
- Integrate `torch.quantization` directly into the training loop.
- Use QAT instead of post-training quantization to recover accuracy loss typically associated with 8-bit weights.
- Implement weight pruning during the final 10 epochs to reach the <250 KB target.

## 4. Feature Engineering (Encoding v3)

### 4.1 Piece Symmetry and Canonical Forms
**Context:** Currently, the model sees symmetrical positions as distinct.
**Proposal:**
- Implement a "canonical board transform" in the generator to normalize positions (e.g., always ensuring the White King is in a specific quadrant).
- This effectively reduces the state space by 4x-8x, leading to even faster convergence and higher accuracy.

---
**Status:** PROPOSAL  
**Author:** Antigravity  
**Date:** March 13, 2026
