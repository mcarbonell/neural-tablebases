import chess
import chess.syzygy
import onnxruntime as ort
import numpy as np
import os
import argparse
from generate_datasets import encode_board

def find_errors(onnx_path, syzygy_path, config="KPvK", encoding_version=5, max_errors=20):
    print(f"Loading ONNX model: {onnx_path}")
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    
    print(f"Opening Syzygy: {syzygy_path}")
    tablebase = chess.syzygy.open_tablebase(syzygy_path)
    
    # Pieces
    if 'v' in config:
        white_side, black_side = config.split('v')
    else:
        white_side = config[:-1]
        black_side = config[-1]
        
    def symbols_to_pieces(symbols, color):
        return [chess.Piece(chess.PIECE_SYMBOLS.index(s.lower()), color) for s in symbols]
        
    w_pieces = symbols_to_pieces(white_side, chess.WHITE)
    b_pieces = symbols_to_pieces(black_side, chess.BLACK)
    all_pieces = w_pieces + b_pieces
    num_pieces = len(all_pieces)
    
    import itertools
    print(f"Searching for errors in {config}...")
    errors_found = 0
    checked = 0
    
    rel_arg = f"v{encoding_version}"
    
    # Use a subset of squares or a sample to avoid taking forever if config is large
    # For KPvK, 64^3 is small enough.
    for squares in itertools.permutations(chess.SQUARES, num_pieces):
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
            
            checked += 1
            if checked % 10000 == 0:
                print(f"Checked {checked} positions... Found {errors_found} errors.")
            
            # Ground truth
            wdl_true_raw = tablebase.probe_wdl(board)
            # Map to 0, 1, 2
            wdl_true = 0 if wdl_true_raw == -2 else (1 if wdl_true_raw == 0 else 2)
            
            # Prediction
            encoding = encode_board(board, relative=rel_arg)
            # Add batch dimension
            inp = encoding.reshape(1, -1).astype(np.float32)
            
            out = sess.run(None, {input_name: inp})
            wdl_logits = out[0]
            wdl_pred = np.argmax(wdl_logits, axis=1)[0]
            
            if wdl_pred != wdl_true:
                errors_found += 1
                print(f"\n[ERROR {errors_found}]")
                print(f"FEN: {board.fen()}")
                print(f"True WDL: {wdl_true} | Pred WDL: {wdl_pred}")
                
                if errors_found >= max_errors:
                    print(f"\nReached max errors ({max_errors}). stopping.")
                    tablebase.close()
                    return

    tablebase.close()
    print(f"Finished. Total checked: {checked}, Total errors: {errors_found}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="data/mlp_kpvk_v5.onnx")
    parser.add_argument("--syzygy", type=str, default="syzygy")
    parser.add_argument("--config", type=str, default="KPvK")
    parser.add_argument("--version", type=int, default=5)
    args = parser.parse_args()
    
    find_errors(args.onnx, args.syzygy, args.config, args.version)
