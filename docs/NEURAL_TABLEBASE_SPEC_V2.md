# Neural Tablebase Spec v2: Lossless Hybrid Compression

**Target Endgame**: K+R vs K+P (KRvKP)
**Objective**: 100% WDL Accuracy + High-Precision DTM/DTZ Approximation
**Architecture**: Hybrid Neural-Symbolic (KAN/SIREN + Residual Exception Map)

---

## 1. Data Analysis (KRvKP)

| Metric | Value |
|--------|-------|
| Total Positions | ~500,000 - 1,500,000 (depending on symmetry) |
| Syzygy Size (WDL+DTZ) | ~10.4 MB |
| **Neural Target Size** | **< 250 KB** (40x reduction from Syzygy) |
| Precision Requirement | 100% WDL, < 0.5 ply MAE for DTZ |

---

## 2. The Hybrid Architecture

The system consists of two primary components: the **Neural Approximator** and the **Lossless Corrector**.

### 2.1 Layer A: Neural Approximator (SIREN/KAN)

We will use a **KAN (Kolmogorov-Arnold Network)** or a **SIREN** for the core backbone.

*   **Input**: Piece-Square Encoding ($4 \text{ pieces} \times 6 \text{ coordinates}$). 
    *   *Optimization*: Use polar coordinates or relative distances to the King/Promotion square to help the network "understand" geometric tension.
*   **Backbone**: 
    *   **Option 1 (KAN)**: 24 inputs $\rightarrow$ [16, 16] hidden (with B-spline activations) $\rightarrow$ 3 outputs (WDL) + 1 output (DTZ).
    *   **Option 2 (SIREN)**: 24 inputs $\rightarrow$ [64, 64, 64] hidden (with $\sin(w_i x + b_i)$ activations).
*   **Goal**: Achieve $>99.8\%$ raw accuracy via massive overfitting.

### 2.2 Layer B: Lossless Corrector (The "Safety Net")

For the $0.2\%$ of positions the network fails to memorize, we use a **Residual Map**.

1.  **Bloom Filter (L1)**: A bit-array (~20KB) that stores whether a position is a "known error" for the network.
2.  **Compressed Hash Map (L2)**: If the Bloom Filter triggers (and is not a false positive), we look up the true WDL/DTZ value in a dictionary containing only the ~1,000-2,000 failing positions.

---

## 3. Training Protocol: "The Overfitting Loop"

Unlike standard ML, we want the network to **not** generalize to other endgames, but to perfectly map this specific domain.

1.  **Phase 1: Global Fit**: Train on the full KRvKP dataset until reaches 99% accuracy.
2.  **Phase 2: Hard Example Mining**: Identify top 5% positions with highest loss.
3.  **Phase 3: Weighted Overfitting**: Re-train with a $10x$ weight on the Hard Examples.
4.  **Phase 4: Freeze & Extract**:
    *   Freeze weights.
    *   Run a full pass on the 1.5M positions.
    *   Export failing FENs to the `exceptions.bin` file.

---

## 4. Technical Comparison

| Feature | Syzygy (Standard) | Neural Tablebase v2 |
|---------|-------------------|---------------------|
| Storage | 10.4 MB | ~200 KB |
| Access | Disk/RAM I/O | GPU/NPU Inference |
| Precision | 100% | 100% (via Hybrid Layer) |
| Latency | Very Low | Low (tensor ops) |

---

## 5. Next Steps (Implementation Plan)

- [ ] **Step 1**: Generate the KRvKP dataset in a binary format (FEN, WDL, DTZ).
- [ ] **Step 2**: Implement the `SIREN-Chess` class in PyTorch.
- [ ] **Step 3**: Run training until validation loss hits a plateau.
- [ ] **Step 4**: Compute the "Exception List" and measure its size when LZMA compressed.
- [ ] **Step 5**: Export the integrated `.neural` file for use in a chess engine.

---
*Document Version: 2.0*
*Project: Neural-Zero-Loss Tablebases*
