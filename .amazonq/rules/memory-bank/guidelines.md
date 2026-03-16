# Guidelines: Development Patterns

## Code Quality Standards

### Module-Level Docstrings
All non-trivial scripts open with a triple-quoted docstring describing purpose:
```python
"""
Parallel version of dataset generation for faster processing.
Uses multiprocessing to utilize all CPU cores efficiently.
"""
```
Analysis scripts use a single-line form: `"""Analyze training details and model size"""`

### Type Hints
Used consistently in `src/` for function signatures:
```python
def generate_dataset_parallel(syzygy_path: str, output_dir: str, config: str,
                              compact: bool = True, relative: bool = False,
                              limit_items: Optional[int] = None,
                              shuffle_seed: Optional[int] = None):
```
`Literal` types used for constrained string params: `EnumerationMode = Literal["permutation", "combination"]`

### Naming Conventions
- Functions: `snake_case` (`encode_board`, `process_chunk`, `find_canonical_form`)
- Classes: `PascalCase` (`MLP`, `SIREN`, `NeuralSearcher`, `TablebaseDataset`)
- Constants / section headers in print output: `UPPER_SNAKE_CASE`
- CLI flags: `--kebab-case` (e.g. `--shuffle-seed`, `--canonical-mode`)
- Argparse internal names: `snake_case` (e.g. `args.shuffle_seed`)

### Print-Based Logging
Scripts use `print()` for progress; `train.py` wraps it in a `log()` closure that also writes to a timestamped file:
```python
def log(message):
    print(message)
    with open(log_file, "a") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
```
Section separators use `"=" * 60` or `"=" * 70`.

### Error Handling
- Broad `except Exception` used in worker processes and tablebase probing to avoid crashing a full run
- Warnings printed with `WARNING:` prefix; execution continues
- Fallback chains: JSON metadata → pickle metadata → None

---

## Structural Conventions

### `src/` vs `scripts/`
- `src/` contains importable modules and the main pipeline scripts
- `scripts/` contains standalone analysis/testing/training helpers that `import` from `src/` via `sys.path.insert(0, 'src')`

### `sys.path` Injection Pattern
All scripts outside `src/` add the source directory at the top:
```python
import sys
sys.path.insert(0, 'src')
from models import get_model_for_endgame
```

### argparse Structure
Every runnable script uses `argparse` with:
- Descriptive `help=` strings on every argument
- Sensible defaults so scripts run with minimal flags
- `choices=` for constrained string arguments
- `action="store_true"` for boolean flags

### Metadata Sidecar Pattern
Every generated artifact (`.npz`, `.pth`) gets a companion `_metadata.json`:
```python
metadata_path = output_path.replace(".npz", "_metadata.json")
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, sort_keys=True)
```
Metadata always includes: `generated_at` / `saved_at` (ISO timestamp), all CLI args, encoding details, dataset stats.

### Partial File Naming
When a dataset is not complete (smoke test / resume), the filename encodes the parameters:
```python
parts = [base_name, "partial", str(total_items)]
if shuffle_seed is not None:
    parts.append(f"seed{int(shuffle_seed)}")
if item_offset:
    parts.append(f"offset{int(item_offset)}")
base_name = "_".join(parts)
```

---

## Semantic Patterns

### Encoding Version Auto-Detection
Both `train.py` and `search_poc.py` infer encoding version from `input_size` using a lookup table:
```python
config_map = {
    43: (3, True, 1),   # 3-piece v1
    45: (3, True, 4),   # 3-piece v4
    65: (4, True, 1),   # 4-piece v1
    68: (4, True, 4),   # 4-piece v4
    ...
}
```
Never hardcode encoding version — always derive from data shape.

### Dual-Head Model Output
All models return `(wdl_logits, dtz_val)`. Always unpack both even if only WDL is needed:
```python
wdl_logits, dtz_pred = model(x)
```
Combined loss: `(loss_wdl * weights).mean() + 0.5 * loss_dtz`

### Class Weight Balancing
Computed from dataset distribution and passed to `CrossEntropyLoss`:
```python
unique, counts = np.unique(wdl_mapped, return_counts=True)
total = len(wdl_mapped)
self.class_weights = torch.FloatTensor([total / (len(unique) * c) for c in counts])
criterion_wdl = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
```
`reduction='none'` is required to apply per-sample hard-mining weights.

### Hard Example Mining Loop
Periodically re-trains on misclassified examples:
```python
if args.hard_mining and (epoch + 1) % args.hard_mining_freq == 0 and len(hard_examples) > 0:
    # stack hard_examples, train for hard_mining_epochs, then clear list
```
Weight ramp: `weights = is_wrong * (hard_example_weight * (1 + epoch_factor)) + 1.0`

### V4 Perspective Normalization
V4 encoding always encodes from White's perspective. Black-to-move boards are flipped before encoding:
```python
if self.encoding_version == 4 and out.turn == chess.BLACK:
    out = flip_board(out)
```
Dataset generation uses `turns_list = (chess.WHITE,)` for v4 (auto mode).

### Canonical Form Application
Applied consistently at both generation and inference time:
```python
# Generation: filter (permutation) or map+dedup (combination)
if canonical_filter:
    if not is_canonical(board, mode=canonical_mode):
        continue

# Inference (search_poc.py):
if self.canonical:
    from canonical_forms import find_canonical_form
    out, _ = find_canonical_form(out, lambda b: (), mode=self.canonical_mode)
```
`canonical_mode="auto"` selects `file_mirror` for pawn endgames, `dihedral` otherwise.

### Chunk-Based Parallel Processing
```python
chunks = [(chunk_id, ..., item_start, count, ...) for i in range(num_chunks)]
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    futures = {executor.submit(process_chunk, chunk): chunk[0] for chunk in chunks}
    for future in as_completed(futures):
        handle_chunk_result(future.result())
```
Each chunk writes a temp `.npz`; all are concatenated after completion, then temp files are deleted.

### Shuffled Index Enumeration
For unbiased sampling of pawn endgames, a full-cycle permutation is used:
```python
rng = random.Random(int(shuffle_seed))
shuffle_start = rng.randrange(space_size)
shuffle_stride = _choose_coprime_stride(space_size, rng)  # gcd(stride, modulus) == 1
# index_i = (shuffle_start + i * shuffle_stride) % space_size
```

### Model Checkpoint Saving
Best model saved on `val_loss` improvement (not accuracy):
```python
if val_loss_avg < best_val_loss:
    torch.save(model.state_dict(), save_path)
    write_checkpoint_metadata(save_path, {"checkpoint": "best", "epoch": epoch+1, ...})
```
Periodic checkpoint every 100 epochs. Final model always saved at end.

### Analysis Script Pattern
Analysis scripts are self-contained: load data, compute metrics, print formatted results:
```python
print("=" * 70)
print("SECTION HEADER")
print("=" * 70)
# ... computations ...
print(f"   Metric: {value:,}")
```
Indented with 3 spaces inside sections. Numbers formatted with `:,` for thousands separators.

### Log Parsing Pattern
`plot_training.py` shows the standard approach for parsing training logs:
```python
epoch_match = re.search(r'Epoch (\d+)/\d+', line)
val_acc_match = re.search(r'Val Acc: ([\d.]+)', line)
```
Log format is fixed: `YYYY-MM-DD HH:MM:SS - Epoch N/M - Time: Xs - Train Loss: X - Val Loss: X - Train Acc: X - Val Acc: X - ... - LR: X`

---

## Common Idioms

- `os.makedirs("logs", exist_ok=True)` — always use `exist_ok=True`
- `torch.device("cuda" if torch.cuda.is_available() else "cpu")` — standard device selection
- `sum(p.numel() for p in model.parameters())` — parameter count
- `data.close()` after `np.load()` before `os.remove()` — required on Windows
- `nonlocal` variables in nested `handle_chunk_result` closure for shared counters
- `_json_safe(obj)` helper to convert numpy scalars before `json.dump`
- `encoding="utf-8"` always specified on `open()` for JSON files
