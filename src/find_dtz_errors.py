import chess
import chess.syzygy
import onnxruntime as ort
import numpy as np
import os
import argparse
from generate_datasets import encode_board

def find_dtz_errors(onnx_path, syzygy_path, config="KBNvK", encoding_version=5, max_errors=20, threshold=1.5):
    print(f"Loading ONNX model: {onnx_path}")
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    
    print(f"Opening Syzygy: {syzygy_path}")
    tablebase = chess.syzygy.open_tablebase(syzygy_path)
    
    # Pieces
    sides = config.lower().split('v')
    def symbols_to_pieces(symbols, color):
        return [chess.Piece(chess.PIECE_SYMBOLS.index(s.lower()), color) for s in symbols]
        
    w_pieces = symbols_to_pieces(sides[0], chess.WHITE)
    b_pieces = symbols_to_pieces(sides[1], chess.BLACK)
    all_pieces = w_pieces + b_pieces
    num_pieces = len(all_pieces)
    
    import itertools
    print(f"Searching for DTZ outliers in {config}...")
    errors_found = 0
    checked = 0
    
    rel_arg = f"v{encoding_version}"
    
    # We use a random sample of placements to avoid the O(64^N) issue
    import random
    squares_list = list(chess.SQUARES)
    
    while errors_found < max_errors:
        squares = random.sample(squares_list, num_pieces)
        board = chess.Board(None)
        for i, piece in enumerate(all_pieces):
            board.set_piece_at(squares[i], piece)
            
        # Pawn ranks
        invalid_pawn = False
        for sq in squares:
            p = board.piece_at(sq)
            if p and p.piece_type == chess.PAWN:
                r = chess.square_rank(sq)
                if r == 0 or r == 7:
                    invalid_pawn = True; break
        if invalid_pawn: continue
        
        for turn in [chess.WHITE, chess.BLACK]:
            board.turn = turn
            if not board.is_valid(): continue
            if board.is_checkmate() or board.is_stalemate(): continue
            
            checked += 1
            if checked % 5000 == 0:
                print(f"Checked {checked} positions... Found {errors_found} outliers.")
                if checked > 1000000: # Safety break
                    print("Reached 1M positions. stopping.")
                    tablebase.close()
                    return

            # Ground truth
            dtz_true = tablebase.probe_dtz(board)
            if dtz_true is None: continue # Material mismatch or other issue
            
            # Normalize DTZ (abs)
            dtz_true = abs(dtz_true)
            
            # Prediction
            encoding = encode_board(board, relative=rel_arg)
            inp = encoding.reshape(1, -1).astype(np.float32)
            
            out = sess.run(None, {input_name: inp})
            dtz_pred = out[1][0][0]
            
            if abs(dtz_pred - dtz_true) > threshold:
                errors_found += 1
                print(f"\n[OUTLIER {errors_found}]")
                print(f"FEN: {board.fen()}")
                print(f"True DTZ: {dtz_true} | Pred DTZ: {dtz_pred:.2f} | Diff: {abs(dtz_pred-dtz_true):.2f}")
                
    tablebase.close()
    print(f"Finished. Total checked: {checked}, Total outliers: {errors_found}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="data/mlp_kbnvk_v5.onnx")
    parser.add_argument("--syzygy", type=str, default="syzygy")
    parser.add_argument("--config", type=str, default="KBNvK")
    parser.add_argument("--threshold", type=float, default=1.5)
    args = parser.parse_args()
    
    find_dtz_errors(args.onnx, args.syzygy, args.config, threshold=args.threshold)
