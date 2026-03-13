"""
Parallel version of dataset generation for faster processing.
Uses multiprocessing to utilize all CPU cores efficiently.

Key improvements:
- Chunk-based processing without iterating all permutations
- Incremental disk writing to avoid memory issues
- Progress tracking with ETA
- Robust error handling
"""
import chess
import chess.syzygy
import numpy as np
import os
import argparse
from typing import List, Tuple
import itertools
from multiprocessing import Pool, cpu_count, Manager
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import time
from datetime import timedelta

# Import encoding functions from original script
sys.path.append(os.path.dirname(__file__))
from generate_datasets import encode_board, piece_move_distance, encode_board_relative

def generate_square_combinations(num_pieces: int, start_idx: int, count: int):
    """
    Generate square combinations efficiently without iterating all permutations.
    Uses combinatorial number system to directly compute the nth combination.
    """
    import math
    
    squares = list(range(64))
    results = []
    
    for offset in range(count):
        idx = start_idx + offset
        
        # Convert index to combination using combinatorial number system
        combination = []
        remaining = idx
        n = 64
        
        for k in range(num_pieces, 0, -1):
            # Find largest c such that C(c, k) <= remaining
            c = k - 1
            while c < n:
                comb = math.comb(c, k)
                if comb > remaining:
                    break
                c += 1
            c -= 1
            
            combination.append(squares[c])
            remaining -= math.comb(c, k)
            n = c
        
        results.append(combination)
    
    return results

def process_chunk(args):
    """
    Process a chunk of square combinations.
    Returns (chunk_id, positions, wdl_labels, dtz_labels).
    """
    chunk_id, syzygy_path, all_pieces, start_idx, count, compact, relative, use_move_distance, canonical = args
    
    try:
        # Open tablebase in this process
        tablebase = chess.syzygy.open_tablebase(syzygy_path)
        
        positions = []
        labels_wdl = []
        labels_dtz = []
        
        num_pieces = len(all_pieces)
        
        # Generate square combinations for this chunk
        combinations = generate_square_combinations(num_pieces, start_idx, count)
        
        # Cache for canonical forms (within this chunk)
        canonical_cache = {}
        
        for squares in combinations:
            board = chess.Board(None)
            
            # Place pieces on board
            for i, piece in enumerate(all_pieces):
                board.set_piece_at(squares[i], piece)
            
            # Check if any pawn is on 1st or 8th rank
            invalid_pawn = False
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN:
                    rank = chess.square_rank(square)
                    if rank == 0 or rank == 7:
                        invalid_pawn = True
                        break
            if invalid_pawn:
                continue
            
            # Test both sides to move
            for turn in [chess.WHITE, chess.BLACK]:
                board.turn = turn
                
                if board.is_valid():
                    try:
                        wdl = tablebase.probe_wdl(board)
                        dtz = tablebase.probe_dtz(board)
                        
                        # Encode board
                        encoding = encode_board(board, compact=compact, relative=relative, 
                                              use_move_distance=use_move_distance)
                        
                        # Apply canonical forms if requested
                        if canonical:
                            try:
                                # Get canonical key for this board
                                from canonical_forms import find_canonical_form, board_to_encoding_key
                                
                                def encoding_func(board):
                                    return encode_board(board, compact=compact, relative=relative,
                                                       use_move_distance=use_move_distance)
                                
                                canonical_board, _ = find_canonical_form(board, encoding_func)
                                canonical_key = board_to_encoding_key(canonical_board, encoding_func)
                                
                                # Check if we've already seen this canonical form in this chunk
                                if canonical_key in canonical_cache:
                                    continue  # Skip duplicate canonical form
                                
                                # Store canonical key and use canonical board encoding
                                canonical_cache[canonical_key] = True
                                encoding = encode_board(canonical_board, compact=compact, relative=relative,
                                                       use_move_distance=use_move_distance)
                            except ImportError:
                                print(f"Warning: canonical_forms module not found. Skipping canonical forms.")
                                canonical = False
                        
                        positions.append(encoding)
                        labels_wdl.append(wdl)
                        labels_dtz.append(dtz)
                    except Exception:
                        pass
        
        tablebase.close()
        
        return (chunk_id, 
                np.array(positions, dtype=np.float32),
                np.array(labels_wdl, dtype=np.int8),
                np.array(labels_dtz, dtype=np.int16))
    
    except Exception as e:
        print(f"Error in chunk {chunk_id}: {e}")
        return (chunk_id, np.array([]), np.array([]), np.array([]))

def generate_dataset_parallel(syzygy_path: str, output_dir: str, config: str, 
                              compact: bool = True, relative: bool = False, 
                              use_move_distance: bool = False, canonical: bool = False,
                              num_workers: int = None, chunk_size: int = 10000):
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
        chunk_size: Number of combinations per chunk (default: 10000)
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
    
    # Calculate total combinations (C(64, num_pieces))
    import math
    total_combinations = math.comb(64, num_pieces)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(cpu_count(), 8)  # Cap at 8 to avoid overhead
    
    print(f"\nParallel processing:")
    print(f"  CPU cores: {cpu_count()}")
    print(f"  Workers: {num_workers}")
    print(f"  Total combinations: {total_combinations:,}")
    print(f"  Chunk size: {chunk_size:,}")
    print(f"  Canonical forms: {'Yes (8 symmetries)' if canonical else 'No'}")
    
    # Create chunks
    num_chunks = (total_combinations + chunk_size - 1) // chunk_size
    print(f"  Total chunks: {num_chunks:,}")
    
    chunks = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        count = min(chunk_size, total_combinations - start_idx)
        chunks.append((i, syzygy_path, all_pieces, start_idx, count,
                      compact, relative, use_move_distance, canonical))
    
    print(f"\nProcessing with incremental disk writing...")
    
    # Temporary files for incremental writing
    temp_dir = os.path.join(output_dir, f"temp_{config}")
    os.makedirs(temp_dir, exist_ok=True)
    
    start_time = time.time()
    completed_chunks = 0
    total_positions = 0
    
    # Process chunks with progress tracking
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_chunk, chunk): chunk[0] for chunk in chunks}
        
        for future in as_completed(futures):
            chunk_id, positions, wdl, dtz = future.result()
            
            # Write chunk to temporary file
            if len(positions) > 0:
                chunk_file = os.path.join(temp_dir, f"chunk_{chunk_id:06d}.npz")
                np.savez_compressed(chunk_file, x=positions, wdl=wdl, dtz=dtz)
                total_positions += len(positions)
            
            completed_chunks += 1
            
            # Progress update
            elapsed = time.time() - start_time
            progress = completed_chunks / num_chunks
            eta = elapsed / progress - elapsed if progress > 0 else 0
            
            print(f"Progress: {completed_chunks}/{num_chunks} chunks "
                  f"({progress*100:.1f}%) | "
                  f"Positions: {total_positions:,} | "
                  f"Elapsed: {timedelta(seconds=int(elapsed))} | "
                  f"ETA: {timedelta(seconds=int(eta))}")
    
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
    
    # Concatenate
    positions = np.concatenate(all_positions)
    labels_wdl = np.concatenate(all_wdl)
    labels_dtz = np.concatenate(all_dtz)
    
    # Remove temp directory
    os.rmdir(temp_dir)
    
    valid_count = len(positions)
    print(f"Found {valid_count:,} valid positions for {config}.")
    
    # Save final dataset
    if canonical:
        output_path = os.path.join(output_dir, f"{config}_canonical.npz")
    else:
        output_path = os.path.join(output_dir, f"{config}.npz")
    
    np.savez_compressed(output_path,
                        x=positions,
                        wdl=labels_wdl,
                        dtz=labels_dtz)
    
    total_time = time.time() - start_time
    print(f"Saved to {output_path}")
    print(f"Total time: {timedelta(seconds=int(total_time))}")
    print(f"Speed: {valid_count / total_time:.0f} positions/second")
    
    # Print canonical reduction if applicable
    if canonical:
        # Estimate original positions (without canonical)
        # For 3 pieces: each canonical form represents ~8 symmetric positions
        estimated_original = valid_count * 8
        reduction = 1 - valid_count / estimated_original if estimated_original > 0 else 0
        print(f"Canonical reduction: ~{reduction:.1%} (estimated {valid_count:,} vs {estimated_original:,})")

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
    parser.add_argument("--canonical", action="store_true",
                       help="Use canonical forms (reduce dataset via board symmetries)")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers (default: CPU count, max 8)")
    parser.add_argument("--chunk-size", type=int, default=10000,
                       help="Number of combinations per chunk (default: 10000)")
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
                             chunk_size=args.chunk_size)
