import chess
import chess.syzygy
import torch
import numpy as np
import argparse
import random
from search_poc import NeuralSearcher
from generate_datasets import encode_board

def find_errors(model_path, syzygy_path, config, num_samples=1000):
    searcher = NeuralSearcher(model_path, syzygy_path)
    
    # Clean config name
    config_clean = config.replace('_canonical', '')
    if 'v' in config_clean:
        white_side, black_side = config_clean.split('v')
    else:
        white_side = config_clean[:-1]
        black_side = config_clean[-1]

    def symbols_to_pieces(symbols):
        piece_map = {'k': chess.KING, 'q': chess.QUEEN, 'r': chess.ROOK, 'b': chess.BISHOP, 'n': chess.KNIGHT, 'p': chess.PAWN}
        return [chess.Piece(piece_map[s.lower()], chess.WHITE) for s in symbols if s.lower() in piece_map]

    w_pieces = symbols_to_pieces(white_side)
    b_pieces = symbols_to_pieces(black_side)
    for p in b_pieces: p.color = chess.BLACK
    all_pieces = w_pieces + b_pieces
        
    errors = []
    print(f"Searching for errors in {config}...")
    
    attempts = 0
    checked = 0
    while len(errors) < 15 and attempts < num_samples * 20:
        attempts += 1
        board = chess.Board(None)
        try:
            squares = random.sample(chess.SQUARES, len(all_pieces))
            for i, piece in enumerate(all_pieces):
                board.set_piece_at(squares[i], piece)
        except ValueError:
            continue
        
        # Pawn rules
        if any(board.piece_at(sq) and board.piece_at(sq).piece_type == chess.PAWN and 
               chess.square_rank(sq) in [0, 7] for sq in chess.SQUARES):
            continue
            
        for turn in [chess.WHITE, chess.BLACK]:
            board.turn = turn
            if board.is_valid():
                checked += 1
                try:
                    # Get side-to-move true WDL
                    true_wdl = searcher.tablebase.probe_wdl(board)
                    # Syzygy WDL: -2=Loss, 0=Draw, 2=Win (Relative to STM)
                    stm_true = 0 if true_wdl == -2 else (1 if true_wdl == 0 else 2)
                    
                    # Get Raw NN prediction (Relative to STM)
                    encoding_flag = 'v4' if searcher.encoding_version == 4 else True
                    x = encode_board(board, relative=encoding_flag, 
                                     use_move_distance=(searcher.encoding_version == 3))
                    x_tensor = torch.from_numpy(x).float().unsqueeze(0).to(searcher.device)
                    wdl_logits, _ = searcher.model(x_tensor)
                    stm_pred = torch.argmax(wdl_logits, dim=1).item()
                    
                    if stm_pred != stm_true:
                        # Find out if search fixes it
                        d1_white = searcher.get_search_wdl(board, 1)
                        d1_stm = d1_white if board.turn == chess.WHITE else 2 - d1_white
                        
                        d3_white = searcher.get_search_wdl(board, 3)
                        d3_stm = d3_white if board.turn == chess.WHITE else 2 - d3_white
                        
                        errors.append({
                            "fen": board.fen(),
                            "true": stm_true,
                            "pred": stm_pred,
                            "d1": d1_stm,
                            "d3": d3_stm
                        })
                        print(f"FOUND ERROR: {board.fen()} | True: {stm_true} | Pred: {stm_pred} | D1: {d1_stm} | D3: {d3_stm}")
                        if len(errors) >= 15: break
                except Exception as e:
                    continue
        if len(errors) >= 15: break
    
    print(f"\nChecked {checked} positions. Found {len(errors)} errors.")
    
    print("\n--- ANALYSIS OF HARD POSITIONS ---")
    for e in errors:
        fix_str = "FIXED" if e["d3"] == e["true"] else "STILL WRONG"
        print(f"FEN: {e['fen']} | True: {e['true']} | Raw: {e['pred']} | D3: {e['d3']} -> {fix_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="data/mlp_best.pth")
    parser.add_argument("--syzygy", type=str, default="syzygy")
    parser.add_argument("--config", type=str, default="KPvKP")
    parser.add_argument("--version", type=int, default=None, help="Explicitly set encoding version")
    args = parser.parse_args()
    
    find_errors(args.model, args.syzygy, args.config)
