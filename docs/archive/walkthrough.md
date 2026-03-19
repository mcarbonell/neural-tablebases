# Walkthrough: Search as Error Correction PoC

We have successfully validated the hypothesis that a shallow Alpha-Beta search can act as an error correction system for the neural tablebase evaluations.

## 🚀 Key Results

| Endgame | Depth 0 (Raw NN) | Depth 1 (Corrected) | Depth 2 (Full Consistency) |
| :--- | :--- | :--- | :--- |
| **KQvK** | 92.50% | **99.50%** | **99.50%** |
| **KRvK** | 95.40% | **99.90%** | **100.00%** |

> [!IMPORTANT]
> A search depth of only **1 ply** (looking at the immediate best move) is enough to eliminate almost all errors in 3-piece endgames. This suggests the neural network is "locally inconsistent" but "globally accurate".

## 🛠️ Implementation Details

### [search_poc.py](file:///e:/neural-tablebases/src/search_poc.py)
A Python script that:
1. Loads the trained MLP model.
2. Implements a Minimax search with Alpha-Beta pruning.
3. Standardizes evaluations to a White-perspective score during search.
4. Compares predictions at different depths against Syzygy ground truth.

### Key Learnings
- **Perspective Matters:** The model's relative evaluation (side-to-move) must be carefully flipped during Minimax to maintain consistency.
- **Error Types:** Most errors are "off-by-one" in the state space, where a winning position is adjacent to a draw/loss the model evaluates incorrectly. Search resolves these local "pockets" of error.

## 🌟 V4 Encoding Breakthrough: Solving the Race
The V4 encoding (Perspective Normalization + Pawn Promotion Progress) has resolved structural blind spots in pawn endgames.

| Configuration | Features | Accuracy (Initial) | Race Positions |
| :--- | :--- | :--- | :--- |
| **KPvKP (v4)** | 68 | 97.4% (Epoch 71) | **SOLVED** (Search Depth 2+) |
| **KRPvKP (v4)** | 95 | 93.3% (Epoch 3) | **High Confidence** (99.9%) |

### Case Study: The Trouble-Race
Position: `8/8/7P/p7/8/8/8/2k2K2 w - - 0 1` (White to move)
- **Previous V1/V2 Models**: Often evaluated as Draw/Loss due to lack of promotion awareness.
- **V4 Model (Depth 0)**: 49.5% Win, 50.4% Draw (Borderline but aware).
- **V4 Model (Depth 2)**: **Win confirmed** (Score: 1.61)
- **V4 Model (Depth 4)**: **Win confirmed** (Score: 1.75)

> [!TIP]
> The addition of the **Pawn Promotion Progress** feature (normalized rank) allows the model to "calculate" queenings even without deep search, giving the Minimax algorithm the signal it needs to pick the winning sequence.

## 🛠️ Tools Added
- [analyze_fen.py](file:///e:/neural-tablebases/src/analyze_fen.py): A dedicated CLI tool for targeted position analysis, supporting both 3 and 5 WDL classes and automatic encoding detection.

## 🏁 Conclusion
The user's intuition was correct: **Distances and perspective are critical**. V4 encoding provides the structural foundation for high-precision evaluations, potentially paving the way for a general-purpose distance-based evaluation in chess engines.
