# Neural Chess Tablebase Compression

> Working draft of a repository-facing overview. This file is retained for paper/community material, but the authoritative live README is `../README.md`.

## Project Summary

Neural Tablebases studies neural compression of Syzygy-style chess endgame information through compact encodings, canonical position generation, and search-based correction. The long-term aim is practical compression with reliable play and eventually perfect behavior through hybrid inference.

## Current Repo Snapshot

- 3-piece work is mature and broadly stable.
- Canonical dataset generation is integrated into the main parallel generator.
- The active training line is KPvKP canonical under `data/v5/KPvKP_canonical.npz`.
- Current checkpoints materially outperform the older V4-only KPvKP milestone often cited in older notes.
- KRRvK and KRPvKP datasets already exist in the workspace and should not be described as purely hypothetical.

## Practical Highlights

- Strong 3-piece performance with geometric encodings.
- Large dataset reductions from canonical forms.
- ONNX export path already used for evaluation/search tooling.
- DirectML path for AMD GPU training on Windows.

## Minimal Usage

### Generate dataset

```bash
python src/generate_datasets_parallel.py --config KPvKP --relative --version 5 --canonical --canonical-mode auto
```

### Train

```bash
python src/train.py --data_path data/v5/KPvKP_canonical.npz --model mlp --epochs 1000 --model_name mlp_kpvkp_v5
```

### Export ONNX

```bash
python src/export_onnx.py --model_path data/mlp_best.pth --output_path data/kpvkp_v5_eval.onnx
```

## Encoding Evolution

- `v1`: original geometric baseline
- `v2` / `v2 fixed`: move-distance and relationship feature experiments
- `v4`: pawn-race focused relative encoding used in earlier KPvKP/KRPvKP runs
- `v5`: current active canonical KPvKP branch

## Recommended Reading Order

1. `../README.md`
2. `../PROJECT_STATUS.md`
3. `../docs/results/CANONICAL_FORMS_RESULTS.md`
4. `../docs/results/encoding_analysis.md`
5. `../docs/analysis/CORRECCION_ERRORES_POR_BUSQUEDA.md`

## Notes

- This document is intentionally shorter and less “marketing-like” than older drafts.
- If this file disagrees with the root README or project status, prefer those newer documents.

---

Last updated: March 20, 2026
