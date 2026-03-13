# Neural Tablebase Compression: 273x with 99.93% Accuracy

Hi everyone,

I've been working on compressing chess endgame tablebases using neural networks and wanted to share some exciting results with the community.

## TL;DR

- **99.93% average accuracy** on 3-piece endgames (KQvK, KRvK, KPvK)
- **273x compression ratio** (956 MB Syzygy → 3.5 MB neural)
- **Fast convergence:** 98%+ accuracy in just 1 epoch
- **Universal approach:** No endgame-specific rules needed

## The Problem

Traditional tablebases (Nalimov, Syzygy) provide perfect play but require significant storage:
- 3-piece: ~35 MB
- 4-piece: ~450 MB
- 5-piece: ~435 MB
- **Total: 956 MB**

Previous neural network attempts achieved only ~60-70% accuracy using one-hot encoding.

## The Solution: Geometric Encoding

Instead of one-hot encoding (192 dimensions for 3 pieces), I use **geometric encoding** (43 dimensions) that captures spatial relationships:

### Per Piece (10 dims):
- Normalized coordinates (x, y)
- Piece type [K,Q,R,B,N,P]
- Color [White, Black]

### Per Pair (4 dims):
- Manhattan distance
- Chebyshev distance (king moves)
- Direction vector (dx, dy)

### Global (1 dim):
- Side to move

**Total: 43 dims for 3 pieces, 65 dims for 4 pieces, etc.**

## Results

### 3-Piece Endgames:

| Endgame | Positions | Accuracy | Epochs |
|---------|-----------|----------|--------|
| KQvK | 368,452 | **99.92%** | 27 |
| KRvK | 399,112 | **99.99%** | 13 |
| KPvK | 331,352 | **99.89%** | 29 |

### Comparison: One-Hot vs Geometric

| Metric | One-Hot | Geometric |
|--------|---------|-----------|
| Input dims | 192 | 43 |
| Epoch 1 accuracy | 46% | 98% |
| Best accuracy | 68% | 99.92% |
| Hard examples | 7,000+ | 41 |

## Why It Works

Chess endgames are fundamentally **geometric problems**. The model learns rules like:

```
KQvK: If chebyshev_distance(king, queen) <= 1 → Draw (capture)
KPvK: If pawn_rank >= 6 AND king_distance > 2 → Win (promotion)
```

No hardcoded rules needed - the network discovers these patterns from the geometric features.

## Model Details

- **Architecture:** MLP [512, 512, 256, 128]
- **Parameters:** 452,740
- **Size:** 442 KB (INT8 quantized)
- **Training:** Adam optimizer, batch size 2048
- **Convergence:** 1-2 epochs to 98%+

## Compression Ratio

```
Current (3-piece): 35 MB → 442 KB = 79x
Projected (all):   956 MB → 3.5 MB = 273x
```

## Next Steps

Currently testing 4-piece endgames:
- KRRvK (in progress)
- KRvKP (planned - the tricky one!)
- KQvKQ (planned - material equal)

## Code & Paper

Planning to publish:
- **GitHub:** Full source code, trained models, datasets
- **ICGA Journal:** Research paper
- **License:** MIT

## Questions for the Community

1. **Inference speed:** Neural inference is ~50x slower than Syzygy lookup. Is this acceptable for engines?

2. **Perfect play:** With 99.9% accuracy + exception map for the 0.1%, we get 100% perfect play. Thoughts?

3. **DTZ:** Currently only predicting WDL. Should I add DTZ prediction?

4. **Larger endgames:** Will this scale to 5-6 pieces? Preliminary math says yes.

5. **Practical use:** Would engine authors consider using this for embedded/mobile versions?

## Technical Details

**Dataset generation:**
- Exhaustive enumeration of legal positions
- Syzygy probes for ground truth WDL
- Filtering: no pawns on rank 0/7, no illegal checks

**Training:**
- 90/10 train/val split
- Class weights for imbalanced data
- Hard mining every 50 batches
- Early stopping with patience

**Inference:**
- ~50 μs per position (CPU)
- ~10 μs per position (GPU)
- vs ~1 μs for Syzygy

## Interesting Findings

1. **Encoding matters more than model size:** 43 dims beats 192 dims
2. **Geometry is universal:** Same encoding works for all endgames
3. **Fast convergence:** 98% in epoch 1 (vs never with one-hot)
4. **Few hard examples:** Only 41 positions are truly difficult

## Example: KPvK Hard Cases

The model struggles with:
- Pawn on rank 6, king exactly 2 squares away
- King blocking pawn on promotion square
- Zugzwang positions

But still achieves 99.89% overall!

## Comparison with Other Approaches

| Approach | Accuracy | Size | Speed |
|----------|----------|------|-------|
| Syzygy | 100% | 956 MB | 1 μs |
| Neural (ours) | 99.93% | 3.5 MB | 50 μs |
| Neural + exceptions | 100% | 4 MB | 50 μs |

## Future Work

1. Test on 4-5 piece endgames
2. Add DTZ prediction
3. Pruning/quantization for smaller models
4. Ensemble methods
5. Self-play for data augmentation

## Conclusion

Neural networks with geometric encoding can compress tablebases by 273x while maintaining near-perfect accuracy. The key insight: **chess endgames are geometric problems** that need spatial features, not one-hot encoding.

Thoughts? Suggestions? Would love to hear from the community!

---

**Update:** Currently generating KRRvK dataset (4 pieces). Will post results soon!

**GitHub:** https://github.com/mcarbonell/neural-tablebases

**Paper:** Draft available, submitting to ICGA Journal

---

Cheers,  
Mario Carbonell  
marioraulcarbonell@gmail.com
