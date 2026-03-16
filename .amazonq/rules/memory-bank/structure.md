# Structure: Project Organization

## Directory Layout

```
neural-tablebases/
├── src/                          # Core source — all production code lives here
│   ├── generate_datasets.py      # Base dataset generator (single-process)
│   ├── generate_datasets_parallel.py  # Parallel generator (recommended)
│   ├── generate_datasets_canonical.py # Canonical-aware generator variant
│   ├── generate_datasets_parallel_canonical.py
│   ├── models.py                 # MLP and SIREN architectures + factory
│   ├── train.py                  # Training loop with hard mining + metadata
│   ├── search_poc.py             # NeuralSearcher: minimax over neural eval
│   ├── canonical_forms.py        # Board symmetry / canonical form logic
│   ├── canonical_forms_fixed.py  # Bugfix variant of canonical forms
│   ├── canonical_basic.py        # Simplified canonical utilities
│   ├── encoding_invariant.py     # Invariant encoding experiments
│   ├── find_errors.py            # Error analysis on trained models
│   └── analyze_fen.py            # FEN-level position analysis
│
├── scripts/
│   ├── analysis/                 # Post-training analysis scripts
│   │   ├── analyze_models.py     # Model accuracy evaluation
│   │   ├── analyze_training_details.py
│   │   ├── plot_training.py      # Training curve visualization
│   │   └── ...
│   ├── testing/                  # Experiment / debug scripts
│   │   ├── visualize_problem.py  # Board visualization for hard positions
│   │   ├── test_siren_hyperparams.py
│   │   └── ...
│   └── training/                 # Batch training helpers
│       ├── generate_parallel.bat
│       ├── train_improved.bat / .sh
│       └── train_sampled.py
│
├── data/                         # Generated artifacts (not in git)
│   ├── *.npz                     # Datasets (x, wdl, dtz arrays)
│   ├── *_metadata.json           # Dataset provenance sidecar files
│   ├── *_metadata.pkl            # Legacy pickle metadata
│   ├── *.pth                     # Trained model checkpoints
│   └── smoke/                    # Small smoke-test datasets
│
├── syzygy/                       # Syzygy tablebase files (.rtbw / .rtbz)
│   └── *.rtbw, *.rtbz            # Ground-truth oracle (not in git)
│
├── logs/                         # Training logs (timestamped)
│   └── train_mlp_YYYYMMDD_HHMMSS.log
│
├── models/                       # Per-endgame model training logs
│   ├── kqvk_canonical/
│   ├── krvk_canonical/
│   └── kpvk_canonical/
│
├── results/                      # JSON result files from analysis scripts
│   └── canonical_*.json
│
├── docs/                         # All documentation
│   ├── analysis/                 # Technical deep-dives (Markdown)
│   ├── paper/                    # ICGA paper draft + forum posts
│   ├── planning/                 # Session notes and proposals
│   └── results/                  # Experiment result summaries
│
└── .amazonq/rules/memory-bank/   # AI assistant memory bank
```

## Core Components and Relationships

### Data Pipeline
```
Syzygy tablebases (.rtbw/.rtbz)
        ↓  probe_wdl / probe_dtz
generate_datasets_parallel.py
        ↓  encode_board() from generate_datasets.py
*.npz dataset  +  *_metadata.json
        ↓
train.py → TablebaseDataset → DataLoader
        ↓
*.pth model  +  *_metadata.json
        ↓
search_poc.py → NeuralSearcher (optional minimax)
```

### Encoding Detection (auto from input_size)
| input_size | pieces | encoding |
|------------|--------|----------|
| 43         | 3      | v1       |
| 45         | 3      | v4       |
| 64         | 3      | v2.1     |
| 65         | 4      | v1       |
| 68         | 4      | v4       |
| 107        | 4      | v2.1     |

### Model Factory
`get_model_for_endgame(model_type, num_pieces, ...)` in `src/models.py` selects hidden layer sizes based on piece count:
- 3-piece: `[512, 512, 256, 128]`
- 4-piece: `[1024, 512, 256, 128]`
- 5-piece: `[2048, 1024, 512, 256]`

## Architectural Patterns

- **Dual-head output**: every model has a `wdl_head` (classification) and `dtz_head` (regression) sharing a backbone
- **Metadata sidecars**: every `.npz` and `.pth` file has a companion `_metadata.json` recording all generation/training parameters
- **Canonical forms**: board symmetries (dihedral or file-mirror) reduce dataset size; `auto` mode selects `file_mirror` for pawn endgames
- **Chunk-based parallel generation**: `ProcessPoolExecutor` over index ranges; each chunk writes a temp `.npz`, then all are concatenated
- **Partial file naming**: when `--limit-items` is used, output is named `*_partial_N[_seedS][_offsetO].npz` to avoid overwriting full datasets
