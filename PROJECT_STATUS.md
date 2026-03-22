# Project Status

Last updated: March 21, 2026

## Current Objective

Expand the V5 Universal Network to complex 4-piece endgames using curriculum learning. Specifically, the KBNvK model has been successfully trained, achieving near-perfect accuracy and validating the WDL-Directed DTZ logic via Negamax. Keep documentation aligned with the codebase for future macro-network integrations.

## Active State

### Recently Completed
1. **KBNvK (Rey+Alfil+Caballo vs Rey) V5 Training**
   - Trained using Transfer Learning from `mlp_kpvkp_v5_post_queen`.
   - Reached early stopping after 242 epochs.
   - **Validation Accuracy**: `0.9996` (99.96%)
   - **Validation Loss**: `0.0024`
   - **DTZ MAE**: `2.05` half-moves (plies).
   - Checkpoint saved to `data/mlp_kbnvk_v5_final.pth` and ONNX exported.

2. **ONNX-based DTZ-aware Search Workflow**
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

Current focus: KPvKP canonical V5 training and harmonized project documentation
