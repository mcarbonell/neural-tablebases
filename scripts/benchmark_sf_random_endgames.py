import chess
import chess.syzygy
import numpy as np
import subprocess
import os
import argparse
import random
from tqdm import tqdm

# Path to Stockfish 18
SF_PATH = r"C:\Users\mrcm_\Local\proj\ajedrez\simple-chess-ai\bin\stockfish\stockfish-windows-x86-64-avx2.exe"
SYZYGY_PATH = "syzygy"

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
        
        # We need to determine whose turn it is to handle perspective
        # FEN format: ... [turn] ...
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
                        
                    # UCI scores are ALWAYS relative to side-to-move.
                    # Convert to relative to White.
                    if not is_white_turn:
                        score = -score
                except (ValueError, IndexError):
                    pass
            if "bestmove" in line:
                break

    if score is None: 
        return 1 # Final fallback to Draw
    
    if score > threshold: return 2
    if score < -threshold: return 0
    return 1

def generate_random_positions(config, n, tablebase):
    """
    Generates N random legal positions for a given config (e.g. 'KPvKP')
    config format: 'K'P'v'K'P' (White pieces vs Black pieces)
    """
    # Simple parser for config like 'KPvKR'
    white_side, black_side = config.split('v')
    
    white_pieces = []
    for char in white_side:
        if char == 'K': white_pieces.append(chess.KING)
        elif char == 'P': white_pieces.append(chess.PAWN)
        elif char == 'R': white_pieces.append(chess.ROOK)
        elif char == 'N': white_pieces.append(chess.KNIGHT)
        elif char == 'B': white_pieces.append(chess.BISHOP)
        elif char == 'Q': white_pieces.append(chess.QUEEN)
        
    black_pieces = []
    for char in black_side:
        if char == 'K': black_pieces.append(chess.KING)
        elif char == 'P': black_pieces.append(chess.PAWN)
        elif char == 'R': black_pieces.append(chess.ROOK)
        elif char == 'N': black_pieces.append(chess.KNIGHT)
        elif char == 'B': black_pieces.append(chess.BISHOP)
        elif char == 'Q': black_pieces.append(chess.QUEEN)

    positions = []
    attempts = 0
    with tqdm(total=n, desc=f"Generating {config}") as pbar:
        while len(positions) < n and attempts < n * 100:
            attempts += 1
            board = chess.Board(None)
            squares = random.sample(range(64), len(white_pieces) + len(black_pieces))
            
            # Place pieces
            idx = 0
            for p_type in white_pieces:
                # Pawns can't be on rank 1 or 8
                if p_type == chess.PAWN and (squares[idx] < 8 or squares[idx] > 55):
                    break
                board.set_piece_at(squares[idx], chess.Piece(p_type, chess.WHITE))
                idx += 1
            else:
                for p_type in black_pieces:
                    if p_type == chess.PAWN and (squares[idx] < 8 or squares[idx] > 55):
                        break
                    board.set_piece_at(squares[idx], chess.Piece(p_type, chess.BLACK))
                    idx += 1
                else:
                    # Legal check
                    if board.is_valid():
                        # Tablebase check (ensure it's indexed)
                        try:
                            wdl = tablebase.probe_wdl(board)
                            # Normalize WDL to 0, 1, 2
                            if wdl > 0: true_class = 2
                            elif wdl < 0: true_class = 0
                            else: true_class = 1
                            
                            positions.append((board.fen(), true_class))
                            pbar.update(1)
                        except Exception:
                            continue
    return positions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["KPvKP", "KPvKR", "KRvKN", "KBvKP"])
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--threshold", type=int, default=150)
    args = parser.parse_args()
    
    if not os.path.exists(SF_PATH):
        print(f"Error: Stockfish not found at {SF_PATH}")
        return

    tablebase = chess.syzygy.open_tablebase(SYZYGY_PATH)
    
    # Start Stockfish
    sf_process = subprocess.Popen(
        [SF_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    sf_process.stdin.write("uci\nisready\n")
    sf_process.stdin.flush()
    
    results = {}

    for config in args.configs:
        print(f"\nTargeting Endgame: {config}")
        dataset = generate_random_positions(config, args.n, tablebase)
        
        y_true = []
        y_pred = []
        
        print(f"Benchmarking SF on {config}...")
        for fen, true_wdl in tqdm(dataset):
            pred_wdl = get_sf_eval_wdl(fen, sf_process, threshold=args.threshold)
            y_true.append(true_wdl)
            y_pred.append(pred_wdl)
            
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        acc = np.mean(y_true == y_pred)
        
        # Confusion Matrix
        cm = np.zeros((3, 3), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
            
        results[config] = {
            "accuracy": acc,
            "cm": cm
        }
        
        print(f"RESULT {config}: Accuracy {acc*100:.2f}%")

    sf_process.stdin.write("quit\n")
    sf_process.stdin.flush()
    sf_process.terminate()
    tablebase.close()
    
    print("\n" + "="*60)
    print("FINAL SUMMARY: STOCKFISH INTUITION (DEPTH 0 EVAL)")
    print("="*60)
    for config, res in results.items():
        print(f"{config:10} | Acc: {res['accuracy']*100:.2f}%")
        
    print("\nDetailed breakdown:")
    for config, res in results.items():
        print(f"\n--- {config} ---")
        cm = res['cm']
        labels = ["Loss", "Draw", "Win"]
        print("      | Loss | Draw | Win ")
        print("--------------------------")
        for i in range(3):
            print(f"{labels[i]:4} | {cm[i,0]:4} | {cm[i,1]:4} | {cm[i,2]:4}")
        
        fatal = cm[2, 0] + cm[0, 2]
        print(f"Fatal Errors (W vs L): {fatal}")

if __name__ == "__main__":
    main()
