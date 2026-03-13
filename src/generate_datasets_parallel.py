"""
Parallel version of dataset generation for faster processing.
Uses multiprocessing to utilize all CPU cores.
"""
import chess
import chess.syzygy
import numpy as np
import os
import argparse
from typing import List, Tuple
import itertools
from multiprocessing import Pool, cpu_count
import sys

# Import encoding functions from original script
sys.path.append(os.path.dirname(__file__))
from generate_datasets import encode_board, piece_move_distance, encode_board_relative

def process_chunk(args):
    """
    Process a chunk of square combinations.
    Returns list of (position, wdl, dtz) tuples.
    """
    syzygy_path, all_pieces, start_idx, end_idx, compact, relative, use_move_distance = args
    
    # Open tablebase in this process
    tablebase = chess.syzygy.open_tablebase(syzygy_path)
    
    results = []
    squares_list = list(chess.SQUARES)
    num_pieces = len(all_pieces)
    
    # Generate combinations for this chunk
    all_perms = itertools.permutations(squares_list, num_pieces)
    
    # Skip to start_idx
    for _ in range(start_idx):
        next(all_perms, None)
    
    # Process chunk
    for idx, squares in enumerate(all_perms):
        if idx >= (end_idx - start_idx):
            break
            
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
                    
                    results.append((encoding, wdl, dtz))
                except Exception:
                    pass
    
    tablebase.close()
    return results

def generate_dataset_parallel(syzygy_path: str, output_dir: str, config: str, 
                              compact: bool = True, relative: bool = False, 
                              use_move_distance: bool = False, num_workers: int = None):
    """
    Generate dataset using parallel processing.
    
    Args:
        syzygy_path: Path to Syzygy tablebases
        output_dir: Output directory
        config: Endgame config (e.g., 'KRRvK')
        compact: Use compact encoding
        relative: Use relative encoding
        use_move_distance: Use move distance (v2)
        num_workers: Number of parallel workers (default: CPU count)
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
    
    # Calculate total combinations
    import math
    total_combinations = math.perm(64, num_pieces)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()
    
    print(f"\nParallel processing:")
    print(f"  CPU cores: {cpu_count()}")
    print(f"  Workers: {num_workers}")
    print(f"  Total combinations: {total_combinations:,}")
    
    # Split work into chunks
    chunk_size = total_combinations // num_workers
    chunks = []
    
    for i in range(num_workers):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_workers - 1 else total_combinations
        chunks.append((syzygy_path, all_pieces, start_idx, end_idx, 
                      compact, relative, use_move_distance))
    
    print(f"  Chunk size: {chunk_size:,} combinations per worker")
    print(f"\nProcessing...")
    
    # Process in parallel
    with Pool(num_workers) as pool:
        results_list = pool.map(process_chunk, chunks)
    
    # Combine results
    print("\nCombining results...")
    positions = []
    labels_wdl = []
    labels_dtz = []
    
    for results in results_list:
        for encoding, wdl, dtz in results:
            positions.append(encoding)
            labels_wdl.append(wdl)
            labels_dtz.append(dtz)
    
    valid_count = len(positions)
    print(f"Found {valid_count:,} valid positions for {config}.")
    
    # Save
    output_path = os.path.join(output_dir, f"{config}.npz")
    np.savez_compressed(output_path,
                        x=np.array(positions, dtype=np.float32),
                        wdl=np.array(labels_wdl, dtype=np.int8),
                        dtz=np.array(labels_dtz, dtype=np.int16))
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--syzygy", type=str, default="syzygy")
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--config", type=str, default="KQvK")
    parser.add_argument("--compact", action="store_true", default=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--relative", action="store_true")
    parser.add_argument("--move-distance", action="store_true")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers (default: CPU count)")
    args = parser.parse_args()
    
    compact = not args.full
    generate_dataset_parallel(args.syzygy, args.data, args.config, 
                             compact=compact, relative=args.relative,
                             use_move_distance=args.move_distance,
                             num_workers=args.workers)
