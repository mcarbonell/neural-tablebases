# Project Status

Last updated: March 23, 2026

## Current Objective

Consolidate the **Universal V6 Network** incorporating tactical pre-computed logic (`is_pinned`, `is_hanging`, `norm_att/def`). The objective is to eliminate "Geometric Catastrophic Forgetting" and reach >99% accuracy across all 3 and 4-piece endgames by combining classical movegen insights with neural evaluation.

## Active State

### Recently Completed
1. **V6 Tactical Encoding (3-4-5 Piece Universe)**
   - Implemented `is_pinned`, `is_hanging`, `norm_att`, and `norm_def` in `src/generate_datasets.py`.
   - Generated the full 3 and 4-piece V6 dataset (~40M positions) across 36 endgames.
   - Built 10 universal shards with 91/115 variables padding in `data/shards_v6`.

2. **Universal V6 Training Pilot**
   - Launched large-scale training on AMD GPU (DirectML).
   - **Performance Milestone**: Reached **90% Validation Accuracy** in less than 20% of the first epoch.
   - Confirmed tactical flags provide an exponential gradient boost compared to V5.

3. **ONNX-based DTZ-aware Search Workflow**
   - Implemented a robust *WDL-Directed DTZ Negamax* in `src/check_dtz_progress.py`.
   - Solved the local Syzygy lone-king dictionary lookup bug in `python-chess` by adding dictionary aliases during initialization.
   - The search prioritizes WDL classification buckets (Win/Draw/Loss) before actively minimizing/maximizing Side-To-Move (STM) DTZ distance as tiebreakers.

### In progress

1. KPvKP canonical training on `data/v5/KPvKP_canonical.npz`
   - Secondary refinements on KPvKP (V5).

### Available now

1. Canonical and non-canonical datasets for the main 3-piece baselines:
   - KQvK
   - KRvK
   - KPvK

2. Larger datasets already present locally:
   - KPvKP canonical
   - KRPvKP canonical
   - KRRvK full
   - KRRvK canonical

3. Hardware path for AMD GPU training on Windows through DirectML.

## What Is Settled

1. Geometric/relative encodings outperform one-hot style baselines by a large margin in this project.
2. Canonical forms are integrated into the primary generator and are no longer an experimental side path.
3. The parallel generator is the standard dataset creation path for serious runs.
4. Metadata sidecars for datasets and checkpoints are now an important part of reproducibility.

## What Changed Since The Older Docs

1. KPvKP is no longer accurately described as just a V4 experiment around 97.4%.
   - The active line is canonical KPvKP under the `v5` dataset branch.
   - Current training has moved materially beyond the older V4 numbers.

2. KRRvK is no longer just "pending generation".
   - Both `data/KRRvK.npz` and `data/KRRvK_canonical.npz` exist in the repo workspace.

3. The top-level documentation had drifted behind the repo.
   - `README.md`
   - `PROJECT_STATUS.md`
   - `docs/README.md`
   have now been realigned with the current code and artifacts.

## Near-Term Priorities

1. Test the dual WDL-DTZ Negamax logic against other complex 4-piece datasets natively via ONNX.
2. Finish the current KPvKP V5 training run and record a stable reference result.
3. Keep the README/status/docs index synchronized with the latest checkpoints and datasets.
4. Decide whether the next comparison target is KRRvK training refresh or broader 4-piece/5-piece evaluation tooling.

## Known Documentation/Code Friction

1. Some historical reports still discuss older V4 milestones as if they were current.
2. Some scripts use input dimensionality as a proxy and label `45/68/95` as V4-compatible, while dataset metadata may say V5.
3. The documentation tree contains both active documents and archival snapshots; they should not be read as equally current.

## Recommended Source Of Truth

When docs disagree, prefer this order:

1. Dataset metadata in `*_metadata.json`
2. Checkpoint metadata in `*_metadata.json`
3. Active training logs in `logs/`
4. `PROJECT_STATUS.md`
5. Older result summaries and archived notes

## Quick Links

- [Main README](README.md)
- [Documentation index](docs/README.md)
- [Scripts guide](scripts/README.md)
- [Canonical forms results](docs/results/CANONICAL_FORMS_RESULTS.md)
- [Encoding analysis](docs/results/encoding_analysis.md)

---

Current focus: Universal V6 training with tactical encoding and multi-piece shard consolidation.
