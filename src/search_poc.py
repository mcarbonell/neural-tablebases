import chess
import chess.syzygy
import torch
import numpy as np
import os
import argparse
import time
from typing import List, Tuple, Dict, Optional
from models import get_model_for_endgame
from generate_datasets import encode_board

class NeuralSearcher:
    def __init__(self, model_path: str, syzygy_path: str, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Loading model from {model_path} on {self.device}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Peek into weights to see input size
        first_key = 'backbone.0.weight'
        if first_key not in checkpoint:
            # Maybe it's a SIREN model or has different naming
            first_key = next(iter(checkpoint.keys()))
            
        weights = checkpoint[first_key]
        input_size = weights.shape[1]
        
        # Mapping from input_size to config
        config_map = {
            43: (3, True, 1),
            45: (3, True, 4),
            65: (4, True, 1),
            68: (4, True, 4),
            91: (5, True, 1),
            95: (5, True, 4),
            46: (3, True, 2),
            64: (3, True, 3),
            107: (4, True, 3),
            161: (5, True, 3)
        }
        
        if input_size in config_map:
            num_pieces, use_relative_encoding, self.encoding_version = config_map[input_size]
        else:
            num_pieces = input_size // 64
            use_relative_encoding = False
            self.encoding_version = 0
            
        self.num_pieces = num_pieces
        self.use_relative_encoding = use_relative_encoding
        self.input_size = input_size
        
        self.model = get_model_for_endgame('mlp', num_pieces, input_size=input_size).to(self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        print(f"Model loaded. Config: {num_pieces} pieces, input_size={input_size}, relative={use_relative_encoding}")
        self.tablebase = chess.syzygy.open_tablebase(syzygy_path)
        
    def evaluate_nn(self, board: chess.Board) -> Tuple[int, float]:
        # Count pieces
        num_on_board = len(board.piece_map())
        
        if num_on_board < self.num_pieces:
            # Fewer pieces than model expects (capture occurred)
            # In 3-piece endgames, this is always a draw
            return 1, 0.0  # Draw, DTZ=0
            
        if num_on_board > self.num_pieces:
            # More pieces? Shouldn't happen in this PoC but handle just in case
            return 1, 0.0
            
        x = encode_board(board, relative=('v4' if self.encoding_version == 4 else True), 
                         use_move_distance=(self.encoding_version == 3))
        x_tensor = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            wdl_logits, dtz_val = self.model(x_tensor)
            wdl_class = torch.argmax(wdl_logits, dim=1).item()
            dtz = dtz_val.item()
            
        # Standardize score to White's perspective
        # 3 classes: 0=Loss, 1=Draw, 2=Win (relative to side to move)
        if self.encoding_version != 4 and board.turn == chess.BLACK:
            wdl_class = 2 - wdl_class
            
        return wdl_class, dtz

    def minimax(self, board: chess.Board, depth: int, alpha: float, beta: float, is_maximizing: bool) -> float:
        if depth == 0 or board.is_game_over():
            wdl_class, _ = self.evaluate_nn(board)
            return float(wdl_class)

        if is_maximizing:
            max_eval = -float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def get_search_wdl(self, board: chess.Board, depth: int) -> int:
        # minimax returns score from White's perspective
        score = self.minimax(board, depth, -float('inf'), float('inf'), board.turn == chess.WHITE)
        wdl_white = int(round(score))
        
        # For comparison with Syzygy (which is relative to side to move),
        # we convert back to side to move perspective
        if board.turn == chess.BLACK:
            return 2 - wdl_white
        return wdl_white

    def verify_accuracy(self, config_name: str, samples: int = 100, depths: List[int] = [0, 1, 2]):
        config = config_name.replace('_canonical', '').replace('_v2_fixed', '').replace('_v2_old', '')
        print(f"Testing on {config}...")
        
        if 'v' in config:
            white_side, black_side = config.split('v')
        else:
            white_side = config[:-1]
            black_side = config[-1]
            
        print(f"Sides: white='{white_side}', black='{black_side}'")

        def symbols_to_pieces(symbols):
            piece_map = {'k': chess.KING, 'q': chess.QUEEN, 'r': chess.ROOK, 'b': chess.BISHOP, 'n': chess.KNIGHT, 'p': chess.PAWN}
            return [chess.Piece(piece_map[s.lower()], chess.WHITE) for s in symbols if s.lower() in piece_map]

        w_pieces = symbols_to_pieces(white_side)
        b_pieces = symbols_to_pieces(black_side)
        for p in b_pieces: p.color = chess.BLACK
        all_pieces = w_pieces + b_pieces
        
        import random
        
        valid_boards = []
        print(f"Searching for {samples} valid sample positions...")
        
        attempts = 0
        while len(valid_boards) < samples and attempts < samples * 1000:
            attempts += 1
            board = chess.Board(None)
            squares = random.sample(chess.SQUARES, len(all_pieces))
            for i, piece in enumerate(all_pieces):
                board.set_piece_at(squares[i], piece)
            
            # Pawn check
            if any(board.piece_at(sq) and board.piece_at(sq).piece_type == chess.PAWN and 
                   chess.square_rank(sq) in [0, 7] for sq in chess.SQUARES):
                continue
                
            for turn in [chess.WHITE, chess.BLACK]:
                board.turn = turn
                if board.is_valid():
                    try:
                        true_wdl = self.tablebase.probe_wdl(board)
                        mapped_wdl = 0 if true_wdl == -2 else (1 if true_wdl == 0 else 2)
                        valid_boards.append((board.copy(), mapped_wdl))
                        if len(valid_boards) >= samples: break
                    except:
                        continue
            if len(valid_boards) >= samples: break
            
        print(f"Found {len(valid_boards)} positions. Starting evaluation...")
        
        results = {d: {"correct": 0, "total": 0} for d in depths}
        
        for i, (board, true_wdl) in enumerate(valid_boards):
            if (i + 1) % (max(1, samples // 10)) == 0:
                print(f"Processed {i+1}/{len(valid_boards)} positions...")
                
            for d in depths:
                pred_wdl = self.get_search_wdl(board, d)
                if pred_wdl == true_wdl:
                    results[d]["correct"] += 1
                results[d]["total"] += 1
                
        print("\n" + "="*40)
        print(f"RESULTS FOR {config} ({len(valid_boards)} positions)")
        print("="*40)
        for d in depths:
            res = results[d]
            acc = res["correct"] / res["total"] if res["total"] > 0 else 0
            print(f"Depth {d}: Accuracy = {acc:.2%} ({res['correct']}/{res['total']})")
        print("="*40)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="data/mlp_final.pth")
    parser.add_argument("--syzygy", type=str, default="syzygy")
    parser.add_argument("--config", type=str, default="KQvK")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--depths", type=str, default="0,1,2")
    args = parser.parse_args()
    
    depths = [int(d) for d in args.depths.split(',')]
    searcher = NeuralSearcher(args.model, args.syzygy)
    searcher.verify_accuracy(args.config, samples=args.samples, depths=depths)

if __name__ == "__main__":
    main()
