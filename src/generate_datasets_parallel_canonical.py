"""
Backward-compatible wrapper for canonical dataset generation.

Historically this file implemented a separate generator with global deduplication.
The canonical pipeline is now implemented in `src/generate_datasets_parallel.py`
with pawn-safe symmetry modes and correct permutation enumeration.

Use this wrapper if you have existing scripts referencing it.
"""

import argparse

from generate_datasets_parallel import generate_dataset_parallel


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel dataset generation with canonical forms (wrapper)")
    parser.add_argument("--syzygy", type=str, default="syzygy",
                        help="Path to Syzygy tablebase directory")
    parser.add_argument("--data", type=str, default="data",
                        help="Output directory for datasets")
    parser.add_argument("--config", type=str, default="KQvK",
                        help="Endgame configuration (e.g., KQvK, KPvKP, KRRvK)")
    parser.add_argument("--compact", action="store_true", default=True,
                        help="Use compact encoding (default: True)")
    parser.add_argument("--full", action="store_true",
                        help="Use full encoding (disables compact)")
    parser.add_argument("--relative", action="store_true",
                        help="Use relative/geometric encoding")
    parser.add_argument("--move-distance", action="store_true",
                        help="Include piece-specific move distance (v2.1 / fixed)")
    parser.add_argument("--version", type=int, default=1,
                        help="Relative encoding version (use 4 for V4); only used with --relative")
    parser.add_argument("--canonical", action="store_true",
                        help="Use canonical forms (reduce dataset via board symmetries)")
    parser.add_argument("--canonical-mode", type=str, default="auto",
                        choices=["auto", "dihedral", "file_mirror", "none"],
                        help="Canonical symmetry group (auto is pawn-safe)")
    parser.add_argument("--enumeration", type=str, default="permutation",
                        choices=["permutation", "combination"],
                        help="Enumeration mode: permutation (exhaustive) or combination (legacy, non-exhaustive)")
    parser.add_argument("--turns", type=str, default="auto",
                        choices=["auto", "both", "white_only"],
                        help="Which side(s) to generate per placement (auto: v4->white_only else both)")
    parser.add_argument("--limit-items", type=int, default=None,
                        help="Debugging: only process the first N enumeration items (not random)")
    parser.add_argument("--item-offset", type=int, default=0,
                        help="Skip the first N items in the chosen index order (resume / avoid biased prefixes)")
    parser.add_argument("--shuffle-seed", type=int, default=None,
                        help="Deterministically shuffle the index order (helps pawn endgames with small --limit-items)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count, max 8)")
    parser.add_argument("--chunk-size", type=int, default=10000,
                        help="Number of base placements per chunk (default: 10000)")
    args = parser.parse_args()

    compact = not args.full

    print("=" * 60)
    print("PARALLEL DATASET GENERATOR (CANONICAL WRAPPER)")
    print("=" * 60)

    generate_dataset_parallel(
        syzygy_path=args.syzygy,
        output_dir=args.data,
        config=args.config,
        compact=compact,
        relative=args.relative,
        use_move_distance=args.move_distance,
        canonical=args.canonical,
        num_workers=args.workers,
        chunk_size=args.chunk_size,
        enumeration=args.enumeration,
        canonical_mode=args.canonical_mode,
        version=args.version,
        turns=args.turns,
        limit_items=args.limit_items,
        item_offset=args.item_offset,
        shuffle_seed=args.shuffle_seed,
    )


if __name__ == "__main__":
    main()
