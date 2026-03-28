import chess
import chess.syzygy
import numpy as np
import os
import argparse
import sys
import time
import json
import math
import random
from datetime import timedelta, datetime
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple, Dict, Iterable
from collections import defaultdict

# Add src to path for relative imports
sys.path.append(os.path.dirname(__file__))
from search.rust_engine import RustGnnEngine
from data.canonical_forms import is_canonical

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

def unrank_square_permutation(num_pieces: int, idx0: int) -> List[int]:
    """Unrank the idx0-th k-permutation (ordered selection) of squares [0..63]."""
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

def process_chunk_gnn(args):
    """
    Process a chunk of positions and extract GNN features using Rust.
    Returns (chunk_id, data_dict, processed_count, kept_count)
    """
    (chunk_id, syzygy_path, all_pieces, start_idx, count, canonical, canonical_mode) = args
    
    try:
        # Initialize Rust engine for this process
        engine = RustGnnEngine()
        tablebase = chess.syzygy.open_tablebase(syzygy_path)
        
        chunk_p_ids = []
        chunk_tac = []
        chunk_edges = []
        chunk_edge_counts = []
        chunk_wdl = []
        chunk_dtz = []
        
        num_pieces = len(all_pieces)
        
        # Deduplicate identical pieces by enforcing a strict square-ordering
        groups: Dict[Tuple[int, bool], List[int]] = defaultdict(list)
        for i, piece in enumerate(all_pieces):
            groups[(piece.piece_type, piece.color)].append(i)
        duplicate_groups = [idxs for idxs in groups.values() if len(idxs) > 1]
        
        processed = 0
        kept = 0
        
        for i in range(start_idx, start_idx + count):
            processed += 1
            squares = unrank_square_permutation(num_pieces, i)
            
            # Swapping identical pieces dedup
            if duplicate_groups:
                ok = True
                for idxs in duplicate_groups:
                    group_squares = [squares[j] for j in idxs]
                    if group_squares != sorted(group_squares):
                        ok = False
                        break
                if not ok: continue

            board = chess.Board(None)
            invalid_pawn = False
            for j, piece in enumerate(all_pieces):
                sq = squares[j]
                if piece.piece_type == chess.PAWN:
                    rank = chess.square_rank(sq)
                    if rank == 0 or rank == 7:
                        invalid_pawn = True
                        break
                board.set_piece_at(sq, piece)
            if invalid_pawn: continue
            
            for turn in [chess.WHITE, chess.BLACK]:
                board.turn = turn
                if board.is_valid():
                    if canonical and not is_canonical(board, mode=canonical_mode):
                        continue
                        
                    fen = board.fen()
                    try:
                        wdl = tablebase.probe_wdl(board)
                        dtz = tablebase.probe_dtz(board)
                        
                        p_ids, tac, edges, edge_count = engine.get_raw_features(fen)
                        
                        chunk_p_ids.append(p_ids)
                        chunk_tac.append(tac)
                        chunk_edges.append(edges)
                        chunk_edge_counts.append(edge_count)
                        chunk_wdl.append(wdl)
                        chunk_dtz.append(dtz)
                        kept += 1
                    except Exception:
                        continue
        
        tablebase.close()
        
        if not chunk_p_ids:
            return (chunk_id, None, processed, kept)
            
        return (chunk_id, {
            "p_ids": np.array(chunk_p_ids, dtype=np.int8),
            "node_tac": np.array(chunk_tac, dtype=np.uint8),
            "edges": np.array(chunk_edges, dtype=np.uint16),
            "edge_counts": np.array(chunk_edge_counts, dtype=np.uint16),
            "wdl": np.array(chunk_wdl, dtype=np.int8),
            "dtz": np.array(chunk_dtz, dtype=np.int16)
        }, processed, kept)
    except Exception as e:
        print(f"Critical error in chunk {chunk_id}: {e}")
        return (chunk_id, None, 0, 0)

class ShardManager:
    def __init__(self, output_dir, shard_size=4_000_000, prefix="v8_shard"):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.prefix = prefix
        self.current_shard_idx = 0
        self.buffer = {
            "p_ids": [], "node_tac": [], "edges": [], 
            "edge_counts": [], "wdl": [], "dtz": []
        }
        self.buffered_count = 0
        os.makedirs(output_dir, exist_ok=True)

    def add_data(self, data_dict):
        if not data_dict: return
        
        count = len(data_dict["wdl"])
        for key in self.buffer:
            self.buffer[key].append(data_dict[key])
        self.buffered_count += count
        
        if self.buffered_count >= self.shard_size:
            self.flush()

    def flush(self):
        if self.buffered_count == 0: return
        
        shard_path = os.path.join(self.output_dir, f"{self.prefix}_{self.current_shard_idx:03d}.npz")
        print(f"Flushing shard {self.current_shard_idx} ({self.buffered_count} positions) to {shard_path}...")
        
        np.savez_compressed(
            shard_path,
            p_ids=np.concatenate(self.buffer["p_ids"]),
            node_tac=np.concatenate(self.buffer["node_tac"]),
            edges=np.concatenate(self.buffer["edges"]),
            edge_counts=np.concatenate(self.buffer["edge_counts"]),
            wdl=np.concatenate(self.buffer["wdl"]),
            dtz=np.concatenate(self.buffer["dtz"])
        )
        
        # Reset buffer
        for key in self.buffer:
            self.buffer[key] = []
        self.buffered_count = 0
        self.current_shard_idx += 1

def generate_gnn_universe(syzygy_path: str, output_dir: str, configs: List[str], 
                          limit_per_config: Optional[int] = None, 
                          workers: Optional[int] = None,
                          chunk_size: int = 10000,
                          shard_size: int = 4_000_000,
                          canonical: bool = True):
    """
    Generate a Universal GNN dataset for multiple configurations.
    """
    print(f"Starting Universal GNN Generation...")
    print(f"Configs: {configs}")
    
    shard_manager = ShardManager(output_dir, shard_size=shard_size)
    
    if workers is None:
        workers = min(cpu_count(), 8)
        
    global_start_time = time.time()
    total_kept = 0
    
    for config in configs:
        print(f"\n[Config: {config}]")
        white_side, black_side = config.replace('V', 'v').split('v')
        w_pieces = [chess.Piece(chess.PIECE_SYMBOLS.index(s.lower()), chess.WHITE) for s in white_side]
        b_pieces = [chess.Piece(chess.PIECE_SYMBOLS.index(s.lower()), chess.BLACK) for s in black_side]
        all_pieces = w_pieces + b_pieces
        num_pieces = len(all_pieces)
        
        total_items = _perm(64, num_pieces)
        if limit_per_config:
            total_items = min(total_items, limit_per_config)
            
        num_chunks = (total_items + chunk_size - 1) // chunk_size
        print(f"Placements to check: {total_items:,} ({num_chunks} chunks)")
        
        chunks_args = []
        for i in range(num_chunks):
            start = i * chunk_size
            count = min(chunk_size, total_items - start)
            chunks_args.append((i, syzygy_path, all_pieces, start, count, canonical, "auto"))
            
        config_start_time = time.time()
        completed = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Using map to avoid creating a massive list of futures
            for idx, result, processed, kept in executor.map(process_chunk_gnn, chunks_args):
                if result:
                    shard_manager.add_data(result)
                    total_kept += kept
                
                completed += 1
                if completed % 10 == 0 or completed == num_chunks:
                    elapsed = time.time() - config_start_time
                    speed = (completed * chunk_size) / elapsed if elapsed > 0 else 0
                    print(f"  Chunk {completed}/{num_chunks} | Progress: {completed/num_chunks*100:.1f}% | Speed: {speed:.0f} pos/s", flush=True)
        
        print(f"\nConfig {config} finished. Total kept so far: {total_kept:,}", flush=True)

    # Final flush
    shard_manager.flush()
    
    duration = time.time() - global_start_time
    print(f"\nAll done! Total positions: {total_kept:,} | Total time: {timedelta(seconds=int(duration))}")
    
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "configs": configs,
        "total_positions": total_kept,
        "shard_size": shard_size,
        "canonical": canonical,
        "version": "v8_gnn"
    }
    with open(os.path.join(output_dir, "dataset_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal GNN Dataset Generator")
    parser.add_argument("--syzygy", type=str, default="syzygy")
    parser.add_argument("--output", type=str, default="data/v8")
    parser.add_argument("--configs", type=str, default="KRvK,KQvK,KPvK", help="Comma separated list of endgames")
    parser.add_argument("--limit", type=int, default=None, help="Limit per endgame (for smoke tests)")
    parser.add_argument("--shard_size", type=int, default=4000000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--canonical", action="store_true", default=True)
    parser.add_argument("--no-canonical", action="store_false", dest="canonical")
    args = parser.parse_args()
    
    config_list = [c.strip() for c in args.configs.split(',')]
    
    generate_gnn_universe(
        args.syzygy, args.output, config_list, 
        limit_per_config=args.limit, 
        workers=args.workers,
        shard_size=args.shard_size,
        canonical=args.canonical
    )
