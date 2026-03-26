# Project Status

Last updated: March 25, 2026

## Current Objective: The V8 GNN Revolution 🧬♟️

Achieve **Global Parity** with Syzygy using the **V8 Graph Neural Network (GNN)** architecture. Following the massive success of the V8 prototype (100% accuracy on KRvK), we are transitioning from flat MLP features to a **Topological Message Passing** architecture.
- [x] **Rust Engine X88**: Ported and validated (100% tactical parity WAC).
- [x] **GNN Feature Extraction**: 16 tactical channels + node flags implemented.
- [x] **V8 Prototype**: Reached **100.00% Accuracy** on KRvK in 5 epochs.
- [ ] **V8 Universal Sharding**: Generate 40M topological positions (Est. 1.8 GB).
- [ ] **V8 Full Training**: Scale ChessGnnV8 to the universal dataset.
- [ ] **GNN-Search Integration**: Low-latency GNN inference for tactical search.

## Active State

### Recently Completed
1. **Rust MoveGen X88 & Tactical Enrichment**
   - High-performance Rust bridge with 100% WAC tactical precision.
   - Extracts 16 edge types (Captures, Checks, Promotions, etc.) and node flags (Safe, Hanging, Protected).

2. **V8 GNN Prototype (Breakthrough Milestone)**
   - Implemented `ChessGnnV8` (Relational GNN in pure PyTorch).
   - Achieved **100% Accuracy** on KRvK (9,000 positions).
   - **Discovery**: Topology-aware networks solve the tablebase problem materially better than flat MLPs.
   - **Extreme Compression**: Topological shards are ~30x smaller than V7 (40M positions ~ 1.8 GB).

3. **Universal V7 Training (Legacy Branch)**
   - Reached **98.92% (Epoch 22)**. Solid, but surpassed by GNN potential.

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

Current focus: Scaling V8 GNN Universal training.
