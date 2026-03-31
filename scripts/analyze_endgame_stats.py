import chess
import chess.syzygy
import argparse
import time
import os
import sys
import random
from typing import List, Tuple, Dict
from math import perm

# Add src to path for canonical logic
sys.path.append(os.path.join(os.getcwd(), 'src'))
from data.canonical_forms import is_canonical

def parse_config(config_name: str) -> Tuple[List[chess.Piece], List[chess.Piece]]:
    """Parse 'KRvKP' type string into white and black piece lists."""
    parts = config_name.replace("V", "v").split("v")
    if len(parts) != 2:
        raise ValueError(f"Invalid config: {config_name}. Expected format like KRvKP.")
    sym_map = {
        'k': chess.KING, 'q': chess.QUEEN, 'r': chess.ROOK,
        'b': chess.BISHOP, 'n': chess.KNIGHT, 'p': chess.PAWN
    }
    w_pieces = [chess.Piece(sym_map[s.lower()], chess.WHITE) for s in parts[0]]
    b_pieces = [chess.Piece(sym_map[s.lower()], chess.BLACK) for s in parts[1]]
    return w_pieces, b_pieces

def analyze_endgame_sampling(config: str, syzygy_path: str, n_samples: int = 100000):
    print(f"\nAnalyzing Endgame (Sampling Mode): {config}")
    print(f"Samples: {n_samples:,}")
    
    try:
        tablebase = chess.syzygy.open_tablebase(syzygy_path)
    except Exception as e:
        print(f"Error opening Syzygy: {e}")
        return

    w_pieces, b_pieces = parse_config(config)
    all_pieces = w_pieces + b_pieces
    num_pieces = len(all_pieces)
    
    # Theoretical max placements (ignoring turn and rank restrictions)
    theor_max = perm(64, num_pieces)
    
    # Statistics
    attempts = 0
    legal_found = 0
    canonical_found = 0
    wdl_counts = {0: 0, 1: 0, 2: 0} # 0: Loss, 1: Draw, 2: Win
    
    start_time = time.time()
    
    # Random Sampling
    while canonical_found < n_samples and attempts < n_samples * 1000:
        attempts += 1
        squares = random.sample(range(64), num_pieces)
        
        # Pawn rank restrictions check (fast check before board creation)
        pawn_ok = True
        for i, piece in enumerate(all_pieces):
            if piece.piece_type == chess.PAWN:
                rank = chess.square_rank(squares[i])
                if rank == 0 or rank == 7:
                    pawn_ok = False
                    break
        if not pawn_ok:
            continue
            
        board = chess.Board(None)
        for i, piece in enumerate(all_pieces):
            board.set_piece_at(squares[i], piece)
            
        # Try both turns
        for turn in [chess.WHITE, chess.BLACK]:
            board.turn = turn
            if board.is_valid():
                legal_found += 1
                if is_canonical(board, mode="auto"):
                    canonical_found += 1
                    try:
                        wdl_raw = tablebase.probe_wdl(board)
                        wdl_class = 1
                        if wdl_raw <= -1: wdl_class = 0
                        elif wdl_raw >= 1: wdl_class = 2
                        wdl_counts[wdl_class] += 1
                    except:
                        pass
        
        if canonical_found % (n_samples // 10) == 0 and canonical_found > 0:
            elapsed = time.time() - start_time
            print(f"  Progress: {canonical_found:,} / {n_samples:,} ({canonical_found/elapsed:.0f} pos/s)", flush=True)

    duration = time.time() - start_time
    tablebase.close()

    # ESTIMATION LOGIC
    # Theoretical placements adjusted for turns (2x) and pawn occupancy (approx 48/64 per pawn)
    # But a cleaner way is just success_rate scaling.
    success_rate = canonical_found / (attempts * 2) # *2 because we try 2 turns per placement attempt
    est_canonical_total = theor_max * 2 * success_rate
    
    print("\n" + "="*50)
    print(f" STATS ESTIMATE (N={canonical_found:,}): {config}")
    print("="*50)
    print(f"Estimated Total Canonical Pos: ~{int(est_canonical_total):,}")
    print(f"Success Rate (Canonical):        {success_rate*100:.4f}%")
    print(f"Analysis Time:                    {duration:.2f}s")
    print("-"*50)
    
    if canonical_found > 0:
        win_pct = wdl_counts[2] / canonical_found * 100
        draw_pct = wdl_counts[1] / canonical_found * 100
        loss_pct = wdl_counts[0] / canonical_found * 100
        
        print(f"WDL Distribution (Estimated):")
        print(f"  Win:  {win_pct:6.2f}%")
        print(f"  Draw: {draw_pct:6.2f}%")
        print(f"  Loss: {loss_pct:6.2f}%")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Endgame Statistics Sampler")
    parser.add_argument("--config", type=str, required=True, help="Endgame config (e.g. KPvK, KPvKP)")
    parser.add_argument("--syzygy", type=str, default="syzygy", help="Path to Syzygy tablebases")
    parser.add_argument("--samples", type=int, default=100000, help="Number of canonical positions to sample")
    args = parser.parse_args()
    
    analyze_endgame_sampling(args.config, args.syzygy, args.samples)
