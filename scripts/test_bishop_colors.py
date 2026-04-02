import chess
import chess.syzygy
import subprocess
import random
import numpy as np
import os

SF_PATH = r"C:\Users\mrcm_\Local\proj\ajedrez\simple-chess-ai\bin\stockfish\stockfish-windows-x86-64-avx2.exe"
SYZYGY_PATH = "syzygy"

def get_sf_eval(fen, sf_process):
    sf_process.stdin.write(f"position fen {fen}\n")
    sf_process.stdin.write("eval\n")
    sf_process.stdin.flush()
    for _ in range(50):
        line = sf_process.stdout.readline().strip()
        if "Final evaluation" in line:
            return 2 if "+" in line else 1
    return 1

def main():
    tablebase = chess.syzygy.open_tablebase(SYZYGY_PATH)
    sf_process = subprocess.Popen([SF_PATH], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    
    results = {'same': [], 'diff': []}
    
    print("Analizando posiciones de KBBvK...")
    while len(results['same']) < 100 or len(results['diff']) < 100:
        board = chess.Board(None)
        squares = random.sample(range(64), 4)
        
        # Place Kings and two White Bishops
        board.set_piece_at(squares[0], chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(squares[1], chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(squares[2], chess.Piece(chess.BISHOP, chess.WHITE))
        board.set_piece_at(squares[3], chess.Piece(chess.BISHOP, chess.WHITE))
        
        if not board.is_valid(): continue
        
        b1_color = (squares[2] % 8 + squares[2] // 8) % 2
        b2_color = (squares[3] % 8 + squares[3] // 8) % 2
        cat = 'same' if b1_color == b2_color else 'diff'
        
        if len(results[cat]) >= 100: continue
        
        try:
            wdl = tablebase.probe_wdl(board)
            true_wdl = 2 if wdl > 0 else (1 if wdl == 0 else 0)
            pred_wdl = get_sf_eval(board.fen(), sf_process)
            
            results[cat].append(true_wdl == pred_wdl)
        except: continue

    print(f"\nRESULTADOS:")
    print(f"Precisión (Alfiles Distinto Color): {np.mean(results['diff']):.2%}")
    print(f"Precisión (Alfiles Mismo Color):    {np.mean(results['same']):.2%}")
    
    sf_process.stdin.write("quit\n")
    sf_process.terminate()
    tablebase.close()

if __name__ == "__main__":
    main()
