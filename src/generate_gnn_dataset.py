import chess
import chess.syzygy
import numpy as np
import os
import argparse
import sys
import time
import json
import random
from datetime import timedelta, datetime
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional

# Add src to path for relative imports
sys.path.append(os.path.dirname(__file__))
from rust_engine import RustGnnEngine

def process_chunk_gnn(args):
    """
    Process a chunk of positions and extract GNN features using Rust.
    """
    (chunk_id, syzygy_path, fen_list, wdl_list, dtz_list) = args
    
    try:
        # Initialize Rust engine for this process
        engine = RustGnnEngine()
        
        chunk_p_ids = []
        chunk_tac = []
        chunk_edges = []
        chunk_edge_counts = []
        chunk_wdl = []
        chunk_dtz = []
        
        for fen, wdl, dtz in zip(fen_list, wdl_list, dtz_list):
            try:
                p_ids, tac, edges, edge_count = engine.get_raw_features(fen)
                
                chunk_p_ids.append(p_ids)
                chunk_tac.append(tac)
                chunk_edges.append(edges)
                chunk_edge_counts.append(edge_count)
                chunk_wdl.append(wdl)
                chunk_dtz.append(dtz)
            except Exception as e:
                # print(f"Error processing FEN {fen}: {e}")
                continue
        
        if not chunk_p_ids:
            return (chunk_id, None)
            
        return (chunk_id, {
            "p_ids": np.array(chunk_p_ids, dtype=np.int8),
            "tac": np.array(chunk_tac, dtype=np.uint8),
            "edges": np.array(chunk_edges, dtype=np.uint16),
            "edge_counts": np.array(chunk_edge_counts, dtype=np.uint16),
            "wdl": np.array(chunk_wdl, dtype=np.int8),
            "dtz": np.array(chunk_dtz, dtype=np.int16)
        })
    except Exception as e:
        print(f"Critical error in chunk {chunk_id}: {e}")
        return (chunk_id, None)

def generate_gnn_dataset(syzygy_path: str, output_path: str, config: str, 
                         limit: Optional[int] = None, workers: Optional[int] = None,
                         chunk_size: int = 5000):
    """
    Generate a GNN-ready dataset using the Rust engine.
    """
    print(f"Generating GNN dataset for {config}...")
    
    # 1. Collect FENs and labels first (Single process is fine for gathering from Syzygy)
    tablebase = chess.syzygy.open_tablebase(syzygy_path)
    
    # Simple FEN collector (mimics exhaustive but with limit)
    # For now, let's use the logic from generate_datasets_parallel but simplified
    # Or just use a simple random sampling if limit is provided.
    
    # For this proof-of-concept, we'll use a simplified version of exhaustive generation
    # based on the config.
    
    # TODO: For high-scale, we should use the unranking logic.
    # For KRvK / KQvK smoke tests, we can just use a simple generator.
    
    def symbols_to_pieces(symbols):
        return [chess.Piece(chess.PIECE_SYMBOLS.index(s.lower()), chess.WHITE) for s in symbols]
    
    white_side, black_side = config.replace('V', 'v').split('v')
    w_pieces = [chess.Piece(chess.PIECE_SYMBOLS.index(s.lower()), chess.WHITE) for s in white_side]
    b_pieces = [chess.Piece(chess.PIECE_SYMBOLS.index(s.lower()), chess.BLACK) for s in black_side]
    all_pieces = w_pieces + b_pieces
    
    print(f"Collecting valid positions (limit={limit})...")
    fens = []
    wdls = []
    dtzs = []
    
    # Simple random placement for the smoke test
    count = 0
    max_attempts = (limit * 20) if limit else 1000000
    attempts = 0
    
    while (limit is None or count < limit) and attempts < max_attempts:
        attempts += 1
        board = chess.Board(None)
        squares = random.sample(range(64), len(all_pieces))
        
        valid_placement = True
        for i, piece in enumerate(all_pieces):
            if piece.piece_type == chess.PAWN:
                r = chess.square_rank(squares[i])
                if r == 0 or r == 7:
                    valid_placement = False
                    break
            board.set_piece_at(squares[i], piece)
        
        if not valid_placement: continue
        
        for turn in [chess.WHITE, chess.BLACK]:
            board.turn = turn
            if board.is_valid():
                wdl = tablebase.probe_wdl(board)
                dtz = tablebase.probe_dtz(board)
                fens.append(board.fen())
                wdls.append(wdl)
                dtzs.append(dtz)
                count += 1
                if limit and count >= limit: break
    
    tablebase.close()
    print(f"Collected {len(fens)} positions. Starting GNN feature extraction...")
    
    # 2. Parallel extraction
    if workers is None:
        workers = min(cpu_count(), 8)
        
    num_chunks = (len(fens) + chunk_size - 1) // chunk_size
    chunks_args = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, len(fens))
        chunks_args.append((i, syzygy_path, fens[start:end], wdls[start:end], dtzs[start:end]))
        
    all_p_ids = []
    all_tac = []
    all_edges = []
    all_edge_counts = []
    all_wdl = []
    all_dtz = []
    
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_chunk_gnn, arg) for arg in chunks_args]
        for future in as_completed(futures):
            chunk_id, result = future.result()
            if result:
                all_p_ids.append(result["p_ids"])
                all_tac.append(result["tac"])
                all_edges.append(result["edges"])
                all_edge_counts.append(result["edge_counts"])
                all_wdl.append(result["wdl"])
                all_dtz.append(result["dtz"])
                
            elapsed = time.time() - start_time
            print(f"Chunk {chunk_id+1}/{num_chunks} done. Speed: {len(all_p_ids)*chunk_size/elapsed:.0f} pos/s")

    # 3. Concatenate and save
    print("Saving compressed dataset...")
    np.savez_compressed(
        output_path,
        p_ids=np.concatenate(all_p_ids),
        node_tac=np.concatenate(all_tac),
        edges=np.concatenate(all_edges),
        edge_counts=np.concatenate(all_edge_counts),
        wdl=np.concatenate(all_wdl),
        dtz=np.concatenate(all_dtz)
    )
    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--syzygy", type=str, default="syzygy")
    parser.add_argument("--config", type=str, default="KRvK")
    parser.add_argument("--output", type=str, default="data/gnn_smoke_test.npz")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    
    generate_gnn_dataset(args.syzygy, args.output, args.config, limit=args.limit, workers=args.workers)
