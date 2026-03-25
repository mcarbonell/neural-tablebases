# Project Status

Last updated: March 25, 2026

## Current Objective

Achieve >99.5% accuracy using the **Universal V7 Network** which incorporates "Dynamic Mobility" (safe squares) and "King Oxygen" features. Following the success of V6 (which reached 99.21% on V1-simple encoding), we aim to leverage:
- [x] **KRPvK Restoration**: Regeneration complete (2026-03-25). CRC bug resolved.
- [x] **V7 Sharding**: 10 Universal Shards built (4M samples each, 40M total). Prefix: `v7_universe`.
- [>] **V7 Training**: Universal MLP training started (DirectML). Targeting 99.5%+ accuracy.
- [ ] **Model Search Testing**: Evaluate V7 models with Minimax-2.

## Active State

### Recently Completed
1. **Universal V6 Training (Refined Milestone)**
   - Reached **99.21% Validation Accuracy** using the large-scale MLP architecture.
   - **Discovery**: Due to an old parameter-passing bug in `encode_board`, the V6 shards were actually encoded with **V1 Simple Features** (Coords + Types + STM). Reaching >99% on such basic data proves the potential of the core universal architecture.
   - Verified that the model recognizes complex wins (e.g. `KRPvK` scored at 1.995).

2. **Full V7 Dataset Generation (Mobility & Oxygen)**
   - Implemented `safe_mobility` and `is_king_oxygen` piece features.
   - Generated the full 3 and 4-piece V7 dataset (~40M positions).
   - Dimensions: 3-piece (63), 4-piece (92), 5-piece (127).
   - Currently regenerating `KRPVK_canonical.npz` (9M positions) to fix CRC corruption.

3. **ONNX-based DTZ-aware Search Workflow**
   - Implemented a robust *WDL-Directed DTZ Negamax* in `src/check_dtz_progress.py`.
   - Solved the local Syzygy lone-king dictionary lookup bug in `python-chess` by adding dictionary aliases during initialization.
   - The search prioritizes WDL classification buckets (Win/Draw/Loss) before actively minimizing/maximizing Side-To-Move (STM) DTZ distance as tiebreakers.

### In progress

1. **V7 Shard Packing**
   - Preparing Universal Shards (10 shards, 4M samples each) using V7 encoding.
   - Using dynamic padding to 92 inputs for 3 and 4-piece universal coverage.

2. **Universal V7 Large Scale Training**
   - Goal: Surpass 99.2% using tactical mobility and oxygen features.
   - Environment: `venv_gpu` with DirectML.

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

Current focus: Shading and Training Universal V7 (Dynamic Mobility).
