# Encoding Analysis

This document replaces an older diagnosis written when the project still relied heavily on compact one-hot style inputs. It summarizes the evolution of the encodings actually present in the repo today.

## Bottom Line

The project no longer revolves around the original compact encoding as the recommended path. The effective line is:

1. relative/geometric encodings
2. canonical dataset reduction
3. search-based correction on top of neural evaluation where needed

## Encoding Stages In This Repo

### Compact encoding

- Historical baseline
- Piece-position one-hot style representation
- Larger input size
- Kept mainly for older experiments and comparisons

This encoding is no longer the preferred route for current work.

### Geometric / relative v1

- Strong breakthrough for 3-piece endgames
- Encodes piece coordinates, piece identities, pairwise geometry, and side to move
- Much smaller than compact one-hot baselines

This is the main reason the project moved away from purely positional one-hot inputs.

### v2 and v2 fixed

- Added move-distance and relationship features
- Useful as an intermediate stage in the repo history
- Still relevant when reading older experiment summaries

### v4

- Focused on pawn endgames and race-sensitive structure
- Used in early KPvKP and KRPvKP work
- Important historical milestone, but no longer the whole story

Older docs that present `v4` as the current state of the project are stale.

### v5

- Current active branch for the canonical KPvKP dataset under `data/v5/`
- Used by the present training run documented in `logs/`
- Coexists with helper code that still handles the same dimensionalities as V4/V5 together

The metadata file next to the dataset is the authoritative source for the encoding version.

## Current Practical Interpretation

If you are deciding what to use now:

1. Use `src/generate_datasets_parallel.py`
2. Prefer `--relative`
3. Use `--version 5` for the active KPvKP line
4. Use `--canonical --canonical-mode auto` unless you have a specific reason not to
5. Check the generated `*_metadata.json`

## Important Correction To Older Notes

Older analysis documents described the compact encoding as the "actual" encoding and flagged the absence of side-to-move information as a central flaw. That was accurate for an earlier stage of the project, but it no longer describes the recommended pipeline in this repository.

For current relative encodings, side to move and geometric structure are part of the feature design.

## Why The Relative Path Won

- Smaller input dimensionality
- Better inductive bias for chess geometry
- Stronger empirical performance in the repo's experiments
- Better fit for canonicalized datasets
- Cleaner handoff into search-based correction workflows

## Remaining Friction

There is still some naming drift in the codebase:

- some scripts classify encodings by input dimensionality
- some metadata labels the dataset as `v5`
- some helper comments still mention V4/V5 together

That is a documentation and code-clarity issue, not evidence that the project is still primarily on the old compact path.

## Recommended Source Of Truth

When checking an encoding:

1. dataset metadata
2. checkpoint metadata
3. active logs
4. `README.md`
5. older result notes

---

Last updated: March 20, 2026
