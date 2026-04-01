import chess
import chess.syzygy
import numpy as np
import subprocess
import os
import argparse
import random
import json
import time
from tqdm import tqdm
from itertools import combinations_with_replacement, product

# Constants
SF_PATH = r"C:\Users\mrcm_\Local\proj\ajedrez\simple-chess-ai\bin\stockfish\stockfish-windows-x86-64-avx2.exe"
SYZYGY_PATH = "syzygy"
MAP_FILE = "data/sf_intuition_map.json"

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
PIECE_SYMBOLS = {chess.PAWN: 'P', chess.KNIGHT: 'N', chess.BISHOP: 'B', chess.ROOK: 'R', chess.QUEEN: 'Q'}

def get_sf_eval_wdl(fen, sf_process, threshold=150):
    """
    Sends 'eval' to Stockfish and parses the result.
    If 'eval' fails (e.g. in check), it falls back to 'go nodes 1' to get a score.
    Returns 2 (Win), 1 (Draw), 0 (Loss) relative to White.
    """
    # 1. Try static 'eval'
    sf_process.stdin.write(f"position fen {fen}\n")
    sf_process.stdin.write("eval\n")
    sf_process.stdin.flush()
    
    score = None
    in_check = False
    
    for _ in range(100):
        line = sf_process.stdout.readline().strip()
        if not line: continue
        if "Final evaluation: none (in check)" in line:
            in_check = True
            break
        if "Final evaluation" in line:
            parts = line.split()
            for p in parts:
                clean_p = p.split('(')[0]
                try:
                    score = float(clean_p) * 100 # convert to cp
                    break
                except ValueError: continue
            break
            
    # 2. If in check, fallback to 'go nodes 1'
    if in_check:
        sf_process.stdin.write("go nodes 1\n")
        sf_process.stdin.flush()
        is_white_turn = " w " in fen
        
        while True:
            line = sf_process.stdout.readline().strip()
            if not line: continue
            if "score cp" in line or "score mate" in line:
                parts = line.split()
                try:
                    idx = parts.index("score")
                    score_type = parts[idx+1] # 'cp' or 'mate'
                    score_val = int(parts[idx+2])
                    if score_type == "mate":
                        score = 10000 if score_val > 0 else -10000
                    else:
                        score = score_val
                    if not is_white_turn:
                        score = -score
                except (ValueError, IndexError): pass
            if "bestmove" in line:
                break
            
    if score is None: 
        return 1
    
    if score > threshold: return 2
    if score < -threshold: return 0
    return 1

def generate_endgame_configs(total_pieces):
    """
    Generates all unique endgame configurations for a given total number of pieces (including 2 kings).
    Example: total_pieces=3 -> K+1 vs K
    """
    non_king_count = total_pieces - 2
    configs = set()
    
    # Split non_king_count into two sides: (left, right)
    # where left >= right to avoid mirror duplicates like KRvK and KvKR
    for left_count in range(non_king_count, (non_king_count // 2) - 1, -1):
        right_count = non_king_count - left_count
        
        # Combinations for left side
        left_combos = list(combinations_with_replacement(PIECE_TYPES, left_count))
        # Combinations for right side
        right_combos = list(combinations_with_replacement(PIECE_TYPES, right_count))
        
        for w_pieces, b_pieces in product(left_combos, right_combos):
            # Sort piece types for canonical naming
            w_str = "".join(sorted([PIECE_SYMBOLS[p] for p in w_pieces], key=lambda x: "QRBNP".index(x)))
            b_str = "".join(sorted([PIECE_SYMBOLS[p] for p in b_pieces], key=lambda x: "QRBNP".index(x)))
            
            config = f"K{w_str}vK{b_str}"
            configs.add(config)
            
    return sorted(list(configs))

def generate_random_positions(config, n, tablebase):
    white_side, black_side = config.split('v')
    
    def parse_side(s):
        pieces = []
        for char in s[1:]: # Skip 'K'
            if char == 'P': pieces.append(chess.PAWN)
            elif char == 'R': pieces.append(chess.ROOK)
            elif char == 'N': pieces.append(chess.KNIGHT)
            elif char == 'B': pieces.append(chess.BISHOP)
            elif char == 'Q': pieces.append(chess.QUEEN)
        return pieces

    w_types = parse_side(white_side)
    b_types = parse_side(black_side)
    all_types = [(chess.KING, chess.WHITE)] + [(t, chess.WHITE) for t in w_types] + \
                [(chess.KING, chess.BLACK)] + [(t, chess.BLACK) for t in b_types]

    positions = []
    attempts = 0
    while len(positions) < n and attempts < n * 200:
        attempts += 1
        board = chess.Board(None)
        squares = random.sample(range(64), len(all_types))
        
        valid_placement = True
        for (p_type, color), sq in zip(all_types, squares):
            if p_type == chess.PAWN and (chess.square_rank(sq) == 0 or chess.square_rank(sq) == 7):
                valid_placement = False
                break
            board.set_piece_at(sq, chess.Piece(p_type, color))
        
        if not valid_placement: continue
        
        for turn in [chess.WHITE, chess.BLACK]:
            board.turn = turn
            if board.is_valid():
                try:
                    wdl = tablebase.probe_wdl(board)
                    if wdl > 0: tc = 2
                    elif wdl < 0: tc = 0
                    else: tc = 1
                    positions.append((board.fen(), tc))
                    if len(positions) >= n: break
                except:
                    continue
        if len(positions) >= n: break
    return positions

def load_map():
    if os.path.exists(MAP_FILE):
        with open(MAP_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_map(data):
    os.makedirs(os.path.dirname(MAP_FILE), exist_ok=True)
    with open(MAP_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Stockfish Intuition Mapper")
    parser.add_argument("--pieces", type=int, nargs="+", default=[3, 4, 5], help="Piece counts to map (including kings)")
    parser.add_argument("--n", type=int, default=10000, help="Samples per endgame")
    parser.add_argument("--threshold", type=int, default=150, help="SF Eval threshold for WDL")
    args = parser.parse_args()

    results_map = load_map()
    tablebase = chess.syzygy.open_tablebase(SYZYGY_PATH)
    
    sf_process = subprocess.Popen(
        [SF_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    sf_process.stdin.write("uci\nisready\n")
    sf_process.stdin.flush()

    try:
        for p_count in args.pieces:
            configs = generate_endgame_configs(p_count)
            print(f"\nScanning {len(configs)} configurations for {p_count} pieces...")
            
            for config in configs:
                if config in results_map:
                    continue
                
                # Check Syzygy availability for this specific config
                # We do a tiny test generation to see if it probes
                test_pos = generate_random_positions(config, 1, tablebase)
                if not test_pos:
                    print(f"Skipping {config} (No Syzygy files or invalid)")
                    continue

                print(f"Mapping {config}...")
                dataset = generate_random_positions(config, args.n, tablebase)
                
                y_true = []
                y_pred = []
                for fen, true_wdl_stm in tqdm(dataset, desc=config, leave=False):
                    pred_wdl_white = get_sf_eval_wdl(fen, sf_process, threshold=args.threshold)
                    
                    # Normalize Syzygy (side-to-move) to White's perspective
                    is_white_turn = " w " in fen
                    if is_white_turn:
                        true_wdl_white = true_wdl_stm
                    else:
                        # Reverse: STM Win (2) -> White Loss (0), STM Loss (0) -> White Win (2)
                        if true_wdl_stm == 2: true_wdl_white = 0
                        elif true_wdl_stm == 0: true_wdl_white = 2
                        else: true_wdl_white = 1
                        
                    y_true.append(true_wdl_white)
                    y_pred.append(pred_wdl_white)
                
                y_true = np.array(y_true)
                y_pred = np.array(y_pred)
                acc = np.mean(y_true == y_pred)
                
                cm = np.zeros((3, 3), dtype=int)
                for t, p in zip(y_true, y_pred):
                    cm[int(t), int(p)] += 1
                
                results_map[config] = {
                    "accuracy": float(acc),
                    "cm": cm.tolist(),
                    "samples": len(y_true),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                print(f"  Result: {acc:.2%} Accuracy")
                save_map(results_map)

    except KeyboardInterrupt:
        print("\nInterrupt received. Saving progress...")
        save_map(results_map)
    finally:
        sf_process.stdin.write("quit\n")
        sf_process.stdin.flush()
        sf_process.terminate()
        tablebase.close()

if __name__ == "__main__":
    main()
