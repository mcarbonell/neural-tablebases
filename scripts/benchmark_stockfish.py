import chess
import chess.syzygy
import subprocess
import re
import argparse
import time
import os
import random
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

# Mapping Syzygy WDL (-2 to 2) to Internal WDL (0 to 2)
WDL_SYZYGY_TO_CLASS = {-2: 0, -1: 0, 0: 1, 1: 2, 2: 2}
WDL_LABELS = {0: "Loss", 1: "Draw", 2: "Win"}

class StockfishAnalyzer:
    def __init__(self, sf_path: str):
        self.sf_path = sf_path
        if not os.path.exists(sf_path):
            raise FileNotFoundError(f"Stockfish not found at {sf_path}")
        
        self.process = subprocess.Popen(
            [self.sf_path], 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True
        )
        self._send_command("uci")
        self._send_command("isready")
        
    def _send_command(self, cmd: str):
        self.process.stdin.write(cmd + "\n")
        self.process.stdin.flush()
        
    def get_eval(self, fen: str) -> Optional[float]:
        """Runs Stockfish 'eval' and parses the Final evaluation score."""
        try:
            self._send_command(f"position fen {fen}")
            self._send_command("eval")
            
            output = ""
            while True:
                line = self.process.stdout.readline()
                output += line
                if "Final evaluation" in line:
                    break
            
            # Parse 'Final evaluation' line
            match = re.search(r"Final evaluation\s+([+-]?\d+\.?\d*)", output)
            if match:
                return float(match.group(1))
            return None
        except Exception as e:
            print(f"Error running Stockfish: {e}")
            return None
            
    def close(self):
        self._send_command("quit")
        self.process.wait()

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

def sample_positions(
    config: str,
    tablebase: chess.syzygy.Tablebase,
    n_samples: int
) -> List[Tuple[chess.Board, int]]:
    """Sample random valid positions for a given endgame config."""
    w_pieces, b_pieces = parse_config(config)
    all_pieces = w_pieces + b_pieces
    n = len(all_pieces)
    results = []
    attempts = 0
    max_attempts = n_samples * 1000

    while len(results) < n_samples and attempts < max_attempts:
        attempts += 1
        squares = random.sample(list(chess.SQUARES), n)
        board = chess.Board(None)

        invalid_pawn = False
        for i, piece in enumerate(all_pieces):
            sq = squares[i]
            if piece.piece_type == chess.PAWN and chess.square_rank(sq) in (0, 7):
                invalid_pawn = True
                break
            board.set_piece_at(sq, piece)
        if invalid_pawn:
            continue

        for turn in [chess.WHITE, chess.BLACK]:
            board.turn = turn
            if not board.is_valid():
                continue
            try:
                wdl_raw = tablebase.probe_wdl(board)
                wdl_class = WDL_SYZYGY_TO_CLASS.get(wdl_raw, 1)
                results.append((board.copy(), wdl_class))
                if len(results) >= n_samples:
                    break
            except Exception:
                continue

    return results

def calculate_metrics(y_true, y_pred):
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    
    # Confusion Matrix
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
        
    # Fatal Errors
    # Win as Loss (2 as 0) or Loss as Win (0 as 2)
    win_as_loss = np.sum((np.array(y_true) == 2) & (np.array(y_pred) == 0))
    loss_as_win = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 2))
    
    return acc, cm, win_as_loss, loss_as_win

def main():
    parser = argparse.ArgumentParser(description="Benchmark Stockfish NNUE raw precision on endgames.")
    parser.add_argument("--sf", type=str, required=True, help="Path to Stockfish binary")
    parser.add_argument("--syzygy", type=str, default="syzygy", help="Path to Syzygy tablebases")
    parser.add_argument("--configs", type=str, default="KPvKP", help="Comma-separated list of endgames (e.g. KRvK,KPvKP)")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples per endgame")
    parser.add_argument("--threshold", type=float, default=50, help="Centipawn threshold for Win/Loss (proxy for WDL)")
    args = parser.parse_args()

    print(f"SF Path: {args.sf}")
    print(f"Syzygy Path: {args.syzygy}")
    
    analyzer = StockfishAnalyzer(args.sf)
    tablebase = chess.syzygy.open_tablebase(args.syzygy)
    
    configs = [c.strip() for c in args.configs.split(",")]
    
    grand_true = []
    grand_pred = []
    
    print("\n" + "="*60)
    print(f"{'Endgame':<12} | {'Pos':<6} | {'Acc (%)':<8} | {'Fatal L':<8} | {'Fatal W':<8}")
    print("-" * 60)
    
    threshold = args.threshold
    
    for config in configs:
        positions = sample_positions(config, tablebase, args.samples)
        if not positions:
            print(f"{config:<12} | No valid positions found.")
            continue
            
        y_true = []
        y_pred = []
        
        for board, true_wdl in positions:
            score = analyzer.get_eval(board.fen())
            if score is None:
                continue
                
            # Score from 'eval' is typically in Pawns (e.g. +0.63)
            # Scale to Centipawns to match threshold 50.
            score = score * 100
            
            # Internal Stockfish eval is usually side-to-move relative?
            # Actually 'eval' command output for SF18 says 'white side'.
            # We need to adjust based on whose turn it is.
            if board.turn == chess.BLACK:
                score = -score
            
            # Map score to WDL
            if score > threshold:
                pred_wdl = 2 # Win
            elif score < -threshold:
                pred_wdl = 0 # Loss
            else:
                pred_wdl = 1 # Draw
                
            y_true.append(true_wdl)
            y_pred.append(pred_wdl)
            
        acc, cm, fal, faw = calculate_metrics(y_true, y_pred)
        print(f"{config:<12} | {len(y_true):<6} | {acc*100:>7.2f}% | {fal:<8} | {faw:<8}")
        
        grand_true.extend(y_true)
        grand_pred.extend(y_pred)

    if grand_true:
        g_acc, g_cm, g_fal, g_faw = calculate_metrics(grand_true, grand_pred)
        print("="*60)
        print(f"{'OVERALL':<12} | {len(grand_true):<6} | {g_acc*100:>7.2f}% (WDL Exact Match)")
        print("="*60)
        
        print("\nOverall Confusion Matrix (True \ Pred):")
        labels = ["Loss ", "Draw ", "Win  "]
        print("           | Loss  | Draw  | Win   ")
        print("-" * 38)
        for i, row in enumerate(g_cm):
            print(f"{labels[i]} | {row[0]:5} | {row[1]:5} | {row[2]:5}")
            
        print("\nPrecision Failure Breakdown (Mistakes):")
        # W -> D, W -> L, D -> W, D -> L, L -> W, L -> D
        mistakes = [
            ("Win as Draw ", g_cm[2, 1]),
            ("Win as Loss ", g_cm[2, 0]),
            ("Draw as Win ", g_cm[1, 2]),
            ("Draw as Loss", g_cm[1, 0]),
            ("Loss as Win ", g_cm[0, 2]),
            ("Loss as Draw", g_cm[0, 1]),
        ]
        
        for label, count in mistakes:
            if count > 0:
                print(f"  {label}: {count}")
        
        if g_fal == 0 and g_faw == 0:
            print("\n  No 'Fatal Errors' (Win vs Loss) detected in this set.")
        else:
            print(f"\n  Note: {g_fal + g_faw} positions had Fatal Errors (W vs L).")
        
    tablebase.close()

if __name__ == "__main__":
    main()
