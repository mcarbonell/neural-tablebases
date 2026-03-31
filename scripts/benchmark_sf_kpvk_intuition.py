import chess
import numpy as np
import subprocess
import os
import argparse
import random
from tqdm import tqdm

# Path to Stockfish 16.1
SF_PATH = r"C:\Users\mrcm_\Local\proj\ajedrez\simple-chess-ai\bin\stockfish\stockfish-windows-x86-64-avx2.exe"

def get_sf_eval_wdl(fen, sf_process, threshold=150):
    """
    Sends 'eval' to Stockfish and parses the result.
    Returns 2 (Win), 1 (Draw), 0 (Loss) relative to White.
    """
    sf_process.stdin.write(f"position fen {fen}\n")
    sf_process.stdin.write("eval\n")
    sf_process.stdin.flush()
    
    score = None
    while True:
        line = sf_process.stdout.readline()
        if not line: break
        # Look for "Final evaluation       +0.63 (white side)"
        if "Final evaluation" in line:
            parts = line.split()
            try:
                # The score is usually the 3rd part (+0.63)
                score_str = parts[2]
                score = float(score_str)
                break
            except Exception:
                continue
    
    if score is None: return 1
    
    # Map CP to Class (0-2)
    cp = score * 100
    if cp > threshold: return 2
    if cp < -threshold: return 0
    return 1

def pids_to_fen(p_ids, turn_white=True):
    """Converts project p_ids (64) to FEN."""
    board = chess.Board(None)
    for i in range(64):
        p = p_ids[i]
        if p == 0: continue
        # Project mapping: 1..6 (W P..K), 7..12 (B P..K)
        color = chess.WHITE if p <= 6 else chess.BLACK
        p_type = p if p <= 6 else p - 6
        board.set_piece_at(i, chess.Piece(p_type, color))
    
    board.turn = chess.WHITE if turn_white else chess.BLACK
    return board.fen()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/v9_full_kpvk/v8_shard_000.npz")
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--threshold", type=int, default=150)
    args = parser.parse_args()
    
    if not os.path.exists(SF_PATH):
        print(f"Error: Stockfish not found at {SF_PATH}")
        return

    print(f"Loading data from {args.data}...")
    data = np.load(args.data)
    p_ids_all = data['p_ids']
    wdl_raw_all = data['wdl'] # Tablebase WDL (-2..2)
    
    # Sample N positions
    indices = random.sample(range(len(p_ids_all)), min(args.n, len(p_ids_all)))
    
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
    
    y_true = []
    y_pred = []
    
    print(f"Benchmarking Stockfish Intuition on {len(indices)} positions...")
    for idx in tqdm(indices):
        p_ids = p_ids_all[idx]
        raw_wdl = wdl_raw_all[idx]
        
        # Ground Truth WDL (0..2)
        if raw_wdl > 0: true_wdl = 2
        elif raw_wdl < 0: true_wdl = 0
        else: true_wdl = 1
        
        # Reconstruct FEN
        fen = pids_to_fen(p_ids)
        
        # Get SF Intuition (eval)
        pred_wdl = get_sf_eval_wdl(fen, sf_process, threshold=args.threshold)
        
        y_true.append(true_wdl)
        y_pred.append(pred_wdl)
        
    sf_process.stdin.write("quit\n")
    sf_process.stdin.flush()
    sf_process.terminate()
    
    # Metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = np.mean(y_true == y_pred)
    
    print("\n" + "="*40)
    print(f"STOCKFISH INTUITION (EVAL DEPTH 0) | KPvK")
    print(f"Accuracy: {acc*100:.2f}%")
    print("="*40)
    
    # Confusion Matrix
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
        
    labels = ["Loss", "Draw", "Win"]
    print("\nConfusion Matrix (True / Pred):")
    print("      | Loss | Draw | Win ")
    print("--------------------------")
    for i in range(3):
        print(f"{labels[i]:4} | {cm[i,0]:4} | {cm[i,1]:4} | {cm[i,2]:4}")
        
    print("\nNotable Mistakes:")
    # Win as Draw
    win_as_draw = cm[2, 1]
    # Draw as Win
    draw_as_win = cm[1, 2]
    # Fatal: Win as Loss
    fatal_w_l = cm[2, 0]
    # Fatal: Loss as Win
    fatal_l_w = cm[0, 2]
    
    print(f"  Win as Draw: {win_as_draw}")
    print(f"  Draw as Win: {draw_as_win}")
    print(f"  Fatal (W vs L): {fatal_w_l + fatal_l_w}")

if __name__ == "__main__":
    main()
