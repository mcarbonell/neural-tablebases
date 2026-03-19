"""
Parallel version of dataset generation for faster processing.
Uses multiprocessing to utilize all CPU cores efficiently.

Key improvements:
- Chunk-based processing with direct unranking (no itertools explosion)
- Incremental disk writing to avoid memory issues
- Progress tracking with ETA
- Robust error handling
"""
import chess
import chess.syzygy
import numpy as np
import os
import argparse
from typing import List, Tuple, Dict, Iterable, Optional, Literal
from collections import defaultdict
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import time
import json
import math
import random
from datetime import timedelta, datetime

# Import encoding functions from original script
sys.path.append(os.path.dirname(__file__))
from generate_datasets import encode_board, piece_move_distance, encode_board_relative

EnumerationMode = Literal["permutation", "combination"]


def _perm(n: int, k: int) -> int:
    """nPk (ordered selections without replacement)."""
    if k < 0 or k > n:
        return 0
    perm_fn = getattr(math, "perm", None)
    if perm_fn is not None:
        return perm_fn(n, k)

    out = 1
    for x in range(n, n - k, -1):
        out *= x
    return out


def _choose_coprime_stride(modulus: int, rng: random.Random) -> int:
    """
    Choose a stride in [1, modulus) such that gcd(stride, modulus) == 1.

    This allows building a full-cycle permutation via (start + i*stride) % modulus.
    """
    if modulus <= 1:
        return 1

    # Probability gcd==1 is ~60% on average; expected few iterations.
    while True:
        stride = rng.randrange(1, modulus)
        if math.gcd(stride, modulus) == 1:
            return stride


def unrank_square_permutation(num_pieces: int, idx0: int) -> List[int]:
    """
    Unrank the idx0-th k-permutation (ordered selection) of squares [0..63].

    This is used for exhaustive permutation enumeration and for shuffled sampling.
    """
    n = 64
    total = _perm(n, num_pieces)
    if idx0 < 0 or idx0 >= total:
        raise ValueError(f"Permutation index out of range: {idx0} (total={total})")

    idx = idx0
    available = list(range(n))
    squares: List[int] = []

    for i in range(num_pieces):
        remaining_n = n - i
        remaining_k = num_pieces - i - 1
        block = _perm(remaining_n - 1, remaining_k) if remaining_k > 0 else 1
        pick = idx // block
        idx = idx % block
        squares.append(available.pop(pick))

    return squares


def generate_square_permutations(num_pieces: int, start_idx: int, count: int) -> List[List[int]]:
    """
    Generate k-permutations of squares [0..63] by unranking indices in [0, 64Pk).

    This is the correct exhaustive enumeration for distinct pieces.
    """
    n = 64
    total = _perm(n, num_pieces)
    end = min(start_idx + count, total)
    return [unrank_square_permutation(num_pieces, idx0) for idx0 in range(start_idx, end)]


def generate_square_permutations_from_indices(num_pieces: int, indices: Iterable[int]) -> List[List[int]]:
    """Like `generate_square_permutations`, but for an arbitrary index list (e.g. shuffled)."""
    return [unrank_square_permutation(num_pieces, idx0) for idx0 in indices]


def unrank_square_combination(num_pieces: int, idx: int, n: int = 64) -> List[int]:
    """
    Unrank the idx-th k-combination of squares [0..n-1] using a combinatorial number system.

    WARNING: Combination enumeration is legacy and not exhaustive for distinct piece assignments.
    """
    if idx < 0:
        raise ValueError(f"Combination index out of range: {idx}")

    combination: List[int] = []
    remaining = idx
    upper = n

    for k in range(num_pieces, 0, -1):
        c = k - 1
        while c < upper:
            comb = math.comb(c, k)
            if comb > remaining:
                break
            c += 1
        c -= 1

        combination.append(c)
        remaining -= math.comb(c, k)
        upper = c

    return combination


def generate_square_combinations(num_pieces: int, start_idx: int, count: int) -> List[List[int]]:
    """
    Generate square combinations efficiently without iterating all permutations.
    Uses combinatorial number system to directly compute the nth combination.

    WARNING: This does NOT enumerate all distinct piece placements. It only
    enumerates unique square sets; the assignment of pieces to squares is fixed
    by the current `all_pieces` order. Prefer `--enumeration permutation` for
    an exhaustive dataset.
    """
    results: List[List[int]] = []
    for offset in range(count):
        idx = start_idx + offset
        results.append(unrank_square_combination(num_pieces, idx, n=64))
    return results


def generate_square_combinations_from_indices(num_pieces: int, indices: Iterable[int]) -> List[List[int]]:
    """Like `generate_square_combinations`, but for an arbitrary index list (e.g. shuffled)."""
    return [unrank_square_combination(num_pieces, idx, n=64) for idx in indices]

def process_chunk(args):
    """
    Process a chunk of square placements.

    Returns (chunk_id, positions, wdl_labels, dtz_labels, processed, kept).
    """
    (chunk_id, syzygy_path, all_pieces, item_start, count, compact, relative, use_move_distance,
     canonical, enumeration, canonical_mode, turns, indexing_mode, shuffle_start, shuffle_stride,
     space_size) = args
    
    try:
        # Open tablebase in this process
        tablebase = chess.syzygy.open_tablebase(syzygy_path)
        
        positions = []
        labels_wdl = []
        labels_dtz = []
        
        num_pieces = len(all_pieces)
        
        if indexing_mode == "sequential":
            if enumeration == "permutation":
                square_lists = generate_square_permutations(num_pieces, item_start, count)
            elif enumeration == "combination":
                square_lists = generate_square_combinations(num_pieces, item_start, count)
            else:
                raise ValueError(f"Unknown enumeration mode: {enumeration!r}")
        elif indexing_mode == "shuffled":
            indices = ((shuffle_start + i * shuffle_stride) % space_size
                       for i in range(item_start, item_start + count))
            if enumeration == "permutation":
                square_lists = generate_square_permutations_from_indices(num_pieces, indices)
            elif enumeration == "combination":
                square_lists = generate_square_combinations_from_indices(num_pieces, indices)
            else:
                raise ValueError(f"Unknown enumeration mode: {enumeration!r}")
        else:
            raise ValueError(f"Unknown indexing mode: {indexing_mode!r}")

        # Deduplicate identical pieces by enforcing a strict square-ordering
        # constraint per (piece_type, color) group.
        groups: Dict[Tuple[int, bool], List[int]] = defaultdict(list)
        for i, piece in enumerate(all_pieces):
            groups[(piece.piece_type, piece.color)].append(i)
        duplicate_groups = [idxs for idxs in groups.values() if len(idxs) > 1]

        # Canonical handling:
        # - permutation: filter to canonical representatives (exact, no global set needed)
        # - combination: map->canonical + per-chunk dedup (approximate, but preserves samples)
        canonical_filter = canonical and enumeration == "permutation"
        canonical_map = canonical and enumeration != "permutation"
        canonical_seen = set() if canonical_map else None

        if canonical:
            try:
                from canonical_forms import is_canonical, find_canonical_form, board_to_canonical_key
            except ImportError:
                print("Warning: canonical_forms module not found. Canonical forms disabled.")
                canonical_filter = False
                canonical_map = False
                is_canonical = None  # type: ignore[assignment]
                find_canonical_form = None  # type: ignore[assignment]
                board_to_canonical_key = None  # type: ignore[assignment]

        processed = 0
        kept = 0

        for squares in square_lists:
            processed += 1

            # Skip permutations that only differ by swapping identical pieces.
            # Only relevant for exhaustive permutation enumeration.
            if enumeration == "permutation" and duplicate_groups:
                ok = True
                for idxs in duplicate_groups:
                    group_squares = [squares[i] for i in idxs]
                    if group_squares != sorted(group_squares):
                        ok = False
                        break
                if not ok:
                    continue

            board = chess.Board(None)
            
            # Place pieces on board
            invalid_pawn = False
            for i, piece in enumerate(all_pieces):
                sq = squares[i]
                if piece.piece_type == chess.PAWN:
                    rank = chess.square_rank(sq)
                    if rank == 0 or rank == 7:
                        invalid_pawn = True
                        break
                board.set_piece_at(sq, piece)
            if invalid_pawn:
                continue
            
            # Test both sides to move
            for turn in turns:
                board.turn = turn
                
                if board.is_valid():
                    try:
                        target_board = board

                        if canonical_map:
                            # Map every sample to its canonical representative and dedup within chunk.
                            canonical_board, _ = find_canonical_form(board, lambda b: (), mode=canonical_mode)  # type: ignore[misc]
                            key = board_to_canonical_key(canonical_board)  # type: ignore[misc]
                            if key in canonical_seen:  # type: ignore[operator]
                                continue
                            canonical_seen.add(key)  # type: ignore[union-attr]
                            target_board = canonical_board

                        if canonical_filter:
                            if not is_canonical(board, mode=canonical_mode):  # type: ignore[misc]
                                continue

                        wdl = tablebase.probe_wdl(board)
                        dtz = tablebase.probe_dtz(board)

                        encoding = encode_board(target_board, compact=compact, relative=relative,
                                                use_move_distance=use_move_distance)
                        
                        positions.append(encoding)
                        labels_wdl.append(wdl)
                        labels_dtz.append(dtz)
                        kept += 1
                    except Exception:
                        pass
        
        tablebase.close()
        
        return (chunk_id, 
                np.array(positions, dtype=np.float32),
                np.array(labels_wdl, dtype=np.int8),
                np.array(labels_dtz, dtype=np.int16),
                processed,
                kept)
    
    except Exception as e:
        print(f"Error in chunk {chunk_id}: {e}")
        return (chunk_id, np.array([]), np.array([]), np.array([]), 0, 0)

def generate_dataset_parallel(syzygy_path: str, output_dir: str, config: str, 
                              compact: bool = True, relative: bool = False, 
                              use_move_distance: bool = False, canonical: bool = False,
                              num_workers: int = None, chunk_size: int = 10000,
                              enumeration: EnumerationMode = "permutation",
                              canonical_mode: str = "auto",
                              version: int = 1,
                              turns: str = "auto",
                              limit_items: Optional[int] = None,
                              item_offset: int = 0,
                              shuffle_seed: Optional[int] = None):
    """
    Generate dataset using parallel processing with incremental disk writing.
    
    Args:
        syzygy_path: Path to Syzygy tablebases
        output_dir: Output directory
        config: Endgame config (e.g., 'KRRvK')
        compact: Use compact encoding
        relative: Use relative encoding
        use_move_distance: Use move distance (v2)
        canonical: Use canonical forms (reduce dataset via board symmetries)
        num_workers: Number of parallel workers (default: CPU count)
        chunk_size: Number of base placements per chunk (default: 10000)
        enumeration: "permutation" (exhaustive) or "combination" (legacy, non-exhaustive)
        canonical_mode: "auto", "dihedral", "file_mirror", "none"
        version: Encoding version (currently only affects relative v4)
        turns: "auto" (v4->white_only else both), "both", or "white_only"
        limit_items: Process only the first N items (debugging/smoke tests)
        item_offset: Skip the first N items in the chosen index order (resume / avoid prefixes)
        shuffle_seed: Deterministically shuffle the index order (helps pawn endgames with small limits)
    """
    if not os.path.exists(syzygy_path):
        raise ValueError(f"Syzygy path {syzygy_path} not found.")
    
    # Determine pieces
    if 'v' in config:
        white_side, black_side = config.split('v')
    else:
        white_side = config[:-1]
        black_side = config[-1]
    
    print(f"Generating dataset for {white_side} vs {black_side}...")
    print(f"  White pieces: {white_side}")
    print(f"  Black pieces: {black_side}")
    
    # Convert to pieces
    def symbols_to_pieces(symbols):
        return [chess.Piece(chess.PIECE_SYMBOLS.index(s.lower()), chess.WHITE) for s in symbols]
    
    w_pieces = symbols_to_pieces(white_side)
    b_pieces = symbols_to_pieces(black_side)
    for p in b_pieces:
        p.color = chess.BLACK
    
    all_pieces = w_pieces + b_pieces
    num_pieces = len(all_pieces)
    
    # Handle encoding version (encode_board expects relative=True or relative="v4")
    rel_arg = relative
    if relative:
        if version == 5:
            rel_arg = "v5"
        elif version == 4:
            rel_arg = "v4"

    if turns not in {"auto", "both", "white_only"}:
        raise ValueError(f"Invalid turns mode: {turns!r}")

    if turns == "auto":
        turns_list = (chess.WHITE,) if rel_arg == "v4" else (chess.WHITE, chess.BLACK)
    elif turns == "white_only":
        if rel_arg != "v4":
            print("  WARNING: --turns white_only without v4 encoding drops black-to-move positions.")
        turns_list = (chess.WHITE,)
    else:
        turns_list = (chess.WHITE, chess.BLACK)

    # Calculate full index space size (before any limits/offset)
    if enumeration == "permutation":
        space_size = _perm(64, num_pieces)
    elif enumeration == "combination":
        space_size = math.comb(64, num_pieces)
    else:
        raise ValueError(f"Unknown enumeration mode: {enumeration!r}")

    if item_offset < 0:
        raise ValueError("--item-offset must be >= 0")
    if item_offset >= space_size:
        raise ValueError(f"--item-offset must be < total items ({space_size:,})")

    remaining_items = space_size - item_offset

    if limit_items is not None:
        if limit_items <= 0:
            raise ValueError("--limit-items must be > 0")
        total_items = min(remaining_items, int(limit_items))
    else:
        total_items = remaining_items

    full_total_items = space_size

    indexing_mode = "sequential"
    shuffle_start = 0
    shuffle_stride = 1
    if shuffle_seed is not None:
        indexing_mode = "shuffled"
        rng = random.Random(int(shuffle_seed))
        shuffle_start = rng.randrange(space_size)
        shuffle_stride = _choose_coprime_stride(space_size, rng)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(cpu_count(), 8)  # Cap at 8 to avoid overhead
    
    print(f"\nParallel processing:")
    print(f"  CPU cores: {cpu_count()}")
    print(f"  Workers: {num_workers}")
    print(f"  Enumeration: {enumeration}")
    print(f"  Index order: {indexing_mode} (offset={item_offset:,}" +
          (f", seed={shuffle_seed}" if shuffle_seed is not None else "") + ")")
    print(f"  Total items: {total_items:,}")
    print(f"  Chunk size: {chunk_size:,}")
    print(f"  Turns: {turns} ({len(turns_list)} per placement)")
    print(f"  Canonical: {'Yes' if canonical else 'No'} (mode={canonical_mode})")
    if enumeration != "permutation" and canonical:
        print("  NOTE: canonical+combination uses per-chunk dedup (approximate). Prefer permutation for exact dedup.")
    
    # Create chunks
    num_chunks = (total_items + chunk_size - 1) // chunk_size
    print(f"  Total chunks: {num_chunks:,}")
    
    chunks = []
    for i in range(num_chunks):
        item_start = item_offset + i * chunk_size
        count = min(chunk_size, (item_offset + total_items) - item_start)
        chunks.append((i, syzygy_path, all_pieces, item_start, count,
                      compact, rel_arg, use_move_distance, canonical,
                      enumeration, canonical_mode, turns_list,
                      indexing_mode, shuffle_start, shuffle_stride, space_size))
    
    print(f"\nProcessing with incremental disk writing...")
    
    # Temporary files for incremental writing
    temp_dir = os.path.join(output_dir, f"temp_{config}")
    os.makedirs(temp_dir, exist_ok=True)
    
    start_time = time.time()
    completed_chunks = 0
    total_positions = 0
    total_processed = 0
    
    def handle_chunk_result(result):
        nonlocal completed_chunks, total_positions, total_processed
        chunk_id, positions, wdl, dtz, processed, kept = result

        # Write chunk to temporary file
        if len(positions) > 0:
            chunk_file = os.path.join(temp_dir, f"chunk_{chunk_id:06d}.npz")
            np.savez_compressed(chunk_file, x=positions, wdl=wdl, dtz=dtz)
            total_positions += kept

        completed_chunks += 1
        total_processed += processed

        # Progress update
        elapsed = time.time() - start_time
        progress = completed_chunks / num_chunks
        eta = elapsed / progress - elapsed if progress > 0 else 0

        print(f"Progress: {completed_chunks}/{num_chunks} chunks "
              f"({progress*100:.1f}%) | "
              f"Items: {total_processed:,} | Kept: {total_positions:,} | "
              f"Elapsed: {timedelta(seconds=int(elapsed))} | "
              f"ETA: {timedelta(seconds=int(eta))}")

    # Process chunks with progress tracking
    if num_workers <= 1:
        for chunk in chunks:
            handle_chunk_result(process_chunk(chunk))
    else:
        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(process_chunk, chunk): chunk[0] for chunk in chunks}
                for future in as_completed(futures):
                    handle_chunk_result(future.result())
        except PermissionError as e:
            print(f"WARNING: multiprocessing unavailable ({e}). Falling back to single-process.")
            for chunk in chunks:
                handle_chunk_result(process_chunk(chunk))
    
    print(f"\nCombining {completed_chunks} chunks...")
    
    # Combine all chunks
    all_positions = []
    all_wdl = []
    all_dtz = []
    
    for i in range(num_chunks):
        chunk_file = os.path.join(temp_dir, f"chunk_{i:06d}.npz")
        if os.path.exists(chunk_file):
            data = np.load(chunk_file)
            all_positions.append(data['x'])
            all_wdl.append(data['wdl'])
            all_dtz.append(data['dtz'])
            data.close()  # Close file before removing
            os.remove(chunk_file)
    
    if not all_positions:
        print(f"No valid positions found for {config}.")
        print("Hint: increase --limit-items, change --enumeration, or avoid biased prefixes for pawn endgames.")
        os.rmdir(temp_dir)
        return

    # Concatenate
    positions = np.concatenate(all_positions)
    labels_wdl = np.concatenate(all_wdl)
    labels_dtz = np.concatenate(all_dtz)
    
    # Remove temp directory
    os.rmdir(temp_dir)
    
    valid_count = len(positions)
    print(f"Found {valid_count:,} valid positions for {config}.")
    
    # Save final dataset
    base_name = f"{config}_canonical" if canonical else config
    complete = bool(total_items == full_total_items)
    if not complete:
        # Avoid clobbering a full dataset when running smoke-tests/resuming.
        parts = [base_name, "partial", str(total_items)]
        if shuffle_seed is not None:
            parts.append(f"seed{int(shuffle_seed)}")
        if item_offset:
            parts.append(f"offset{int(item_offset)}")
        base_name = "_".join(parts)

    output_path = os.path.join(output_dir, f"{base_name}.npz")
    
    np.savez_compressed(output_path,
                        x=positions,
                        wdl=labels_wdl,
                        dtz=labels_dtz)
    
    total_time = time.time() - start_time
    print(f"Saved to {output_path}")
    print(f"Total time: {timedelta(seconds=int(total_time))}")
    print(f"Speed: {valid_count / total_time:.0f} positions/second")

    # Save metadata (JSON) next to the dataset for reproducibility/debugging.
    canonical_action = None
    if canonical:
        canonical_action = "filter" if enumeration == "permutation" else "map"

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "config": config,
        "syzygy_path": syzygy_path,
        "enumeration": enumeration,
        "total_items": total_items,
        "full_total_items": full_total_items,
        "complete": complete,
        "item_offset": int(item_offset),
        "indexing": {
            "mode": indexing_mode,
            "shuffle_seed": int(shuffle_seed) if shuffle_seed is not None else None,
            "shuffle_start": int(shuffle_start) if indexing_mode == "shuffled" else None,
            "shuffle_stride": int(shuffle_stride) if indexing_mode == "shuffled" else None,
        },
        "chunk_size": chunk_size,
        "turns": turns,
        "turns_per_placement": len(turns_list),
        "encoding": {
            "compact": bool(compact),
            "relative": bool(relative),
            "relative_arg": rel_arg if isinstance(rel_arg, str) else ("relative" if rel_arg else "compact"),
            "version": version,
            "use_move_distance": bool(use_move_distance),
        },
        "canonical": bool(canonical),
        "canonical_mode": canonical_mode if canonical else None,
        "canonical_action": canonical_action,
        "positions": int(valid_count),
        "dimensions": int(positions.shape[1]),
        "dtypes": {
            "x": str(positions.dtype),
            "wdl": str(labels_wdl.dtype),
            "dtz": str(labels_dtz.dtype),
        },
    }

    metadata_path = output_path.replace(".npz", "_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    print(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel dataset generation for neural tablebases")
    parser.add_argument("--syzygy", type=str, default="syzygy",
                       help="Path to Syzygy tablebase directory")
    parser.add_argument("--data", type=str, default="data",
                       help="Output directory for datasets")
    parser.add_argument("--config", type=str, default="KQvK",
                       help="Endgame configuration (e.g., KQvK, KRRvK)")
    parser.add_argument("--compact", action="store_true", default=True,
                       help="Use compact encoding (default: True)")
    parser.add_argument("--full", action="store_true",
                       help="Use full encoding (disables compact)")
    parser.add_argument("--relative", action="store_true",
                       help="Use relative/geometric encoding")
    parser.add_argument("--move-distance", action="store_true",
                       help="Include piece-specific move distance (encoding v2)")
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
    print("PARALLEL NEURAL TABLEBASE DATASET GENERATOR")
    print("=" * 60)
    
    generate_dataset_parallel(args.syzygy, args.data, args.config, 
                             compact=compact, relative=args.relative,
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
                             shuffle_seed=args.shuffle_seed)
