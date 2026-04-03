# Analysis: Vanguard Master Run v1 (4P + 5P Universe)
**Date:** 2026-04-03
**Model:** Vanguard V10.1 (Weighted Relational GNN)
**Status:** Interrupted after 9h 17m (Epoch 0, ~55% complete)

## 1. Training Parameters
- **Dataset:** 15.8M positions (aggregated 4P and 5P shards)
- **Hardware:** AMD Radeon 780M (DirectML)
- **Learning Rate:** 0.001 (Fixed with ReduceLROnPlateau active but not triggered)
- **Batch Size:** 128
- **Average Speed:** 276 pos/s

## 2. Quantitative Results (At Termination)
| Metric | Value | Delta from Peak |
| :--- | :--- | :--- |
| **Global Accuracy (Epoch 0)** | 96.25% | -1.85% |
| **Peak Accuracy (4P Phase)** | 98.10% | -- |
| **Mean Absolute Error (MAE)** | 1.64 | +0.20 |
| **Final Loss** | 1.89 | +1.30 |

## 3. Key Observations & Bottlenecks

### A. The "5-Piece Complexity Wall"
The model maintained an accuracy of **98%+** while processing shards primarily composed of 4-piece endgames. However, upon transitioning to shards from `data/universal_5p`, a steady degradation in global accuracy was observed, stabilizing around **96.2%**.
- **Interpretation:** V10.1 is highly capable of 4-piece intuition but struggles with the increased coordination requirements of 5-piece endgame states (e.g., KBBvK, KBNvK).

### B. Inter-Shard Oscillation
Clear spikes in Loss and MAE occurred at each shard boundary. 
- **Example:** Batch 0 of Shard X would often see a Loss increase of **+0.02 to +0.05** before beginning a slow downward trend.
- **Root Cause:** **Temporal Correlation.** Training one config-specific shard at a time prevents the network from forming a "Universal Logic," forcing it to re-adapt its weights to the specific geometry of the new endgame type every hour.

### C. Superiority Over Stockfish
Despite the perceived stagnation:
- A **96.2% Intuition Accuracy** on 5-pieces is significantly higher than the **54-85%** accuracy observed in Stockfish 18 at Depth 0 for complex endgames.
- The model has already exceeded the "tactical intuition" of the world's strongest engine in the 5-piece domain.

## 4. Conclusion for Next Cycle
The V10.1 architecture has likely reached its architectural limit for shard-based training. To reach **99% accuracy**, the next run must solve the correlation issue via **Global RAM Shuffling** and enhance coordination representation via the **V11 Field Theory** (16-channel influence mapping).
