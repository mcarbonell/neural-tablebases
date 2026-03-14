import chess
import chess.syzygy
import argparse
import random
from search_poc import NeuralSearcher

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
                    # Syzygy WDL is relative to STM.
                    if searcher.num_wdl_classes == 5:
                        stm_true = 0 if true_wdl == -2 else (1 if true_wdl == -1 else (2 if true_wdl == 0 else (3 if true_wdl == 1 else 4)))
                    else:
                        stm_true = 0 if true_wdl == -2 else (1 if true_wdl == 0 else 2)
                    
                    # Raw NN prediction (convert back to STM perspective)
                    wdl_white_pred, _ = searcher.evaluate_nn(board)
                    stm_pred = wdl_white_pred if board.turn == chess.WHITE else (searcher.num_wdl_classes - 1 - wdl_white_pred)
                    
                    if stm_pred != stm_true:
                        # Find out if search fixes it
                        d1_stm = searcher.get_search_wdl(board, 1)
                        d3_stm = searcher.get_search_wdl(board, 3)
                        
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
    args = parser.parse_args()
    
    find_errors(args.model, args.syzygy, args.config)
