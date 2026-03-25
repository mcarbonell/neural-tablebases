import chess
import chess.syzygy
import torch
import numpy as np
import os
import json
import argparse
import time
from typing import List, Tuple, Dict, Optional
from src.models import get_model_for_endgame
from src.generate_datasets import encode_board

class NeuralSearcherV6:
    def __init__(self, model_path: str, syzygy_path: str, device: str = None):
        if device is None:
            try:
                import torch_directml
                self.device = torch_directml.device()
                print(f"Using DirectML: {self.device}")
            except:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Loading V6 model from {model_path}...")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Determine input size from checkpoint
        if 'backbone.0.weight' in checkpoint:
            input_size = checkpoint['backbone.0.weight'].shape[1]
        else:
            input_size = next(v.shape[1] for k, v in checkpoint.items() if len(v.shape) == 2)
            
        print(f"Detected input size: {input_size}")

        # Number of pieces (V6: pieces*15 + pairs*4)
        # 4 pieces: 4*15 + 6*4 = 84
        # 5 pieces: 5*15 + 10*4 = 115
        # Wait, if it's 91, let's see... 
        # Actually, let's just use the input_size from checkpoint
        self.num_pieces = 4 # Default for universal model
        if "num_pieces" in model_path: # Heuristic
             self.num_pieces = 5 if "5" in model_path else 4

        self.model = get_model_for_endgame(num_pieces=self.num_pieces, model_type='mlp', input_size=input_size)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        
        self.tablebase = chess.syzygy.open_tablebase(syzygy_path)
        
    def evaluate_nn(self, board: chess.Board) -> float:
        # Side to move: -2, 0, 2
        # We need to map -2, 0, 2 to 0, 1, 2 for classification, 
        # but here we return the raw expected value [0, 2]
        
        # V6 uses version=6
        x = encode_board(board, relative='v6')
        
        # Padding logic for universal models
        # If the model was trained on larger endgames, it expects more features
        # We must pad with zeros to match input_size
        expected_size = self.model.backbone[0].in_features
        if len(x) < expected_size:
            padding = np.zeros(expected_size - len(x), dtype=np.float32)
            x = np.concatenate([x, padding])
        
        x_tensor = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out_wdl, _ = self.model(x_tensor)
            probs = torch.softmax(out_wdl, dim=1).squeeze(0)
            # Expected value: 0*P(L) + 1*P(D) + 2*P(W)
            score = probs[1] + 2.0 * probs[2]
        return score.item()

    def minimax(self, board: chess.Board, depth: int, alpha: float, beta: float, is_white_moving: bool) -> float:
        if depth == 0 or board.is_game_over():
            score = self.evaluate_nn(board)
            # score is [0, 2] from STM perspective. 
            # Convert to White perspective
            if board.turn == chess.BLACK:
                return 2.0 - score
            return score

        if is_white_moving:
            max_eval = -1.0
            for move in board.legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha: break
            return max_eval
        else:
            min_eval = 3.0
            for move in board.legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha: break
            return min_eval

    def get_wdl_decision(self, board: chess.Board, depth: int) -> int:
        # Returns 0 (Loss), 1 (Draw), 2 (Win) from STM perspective
        score_white = self.minimax(board, depth, -1.0, 3.0, board.turn == chess.WHITE)
        
        # Round 
        wdl_white = int(round(score_white))
        
        if board.turn == chess.BLACK:
            return 2 - wdl_white
        return wdl_white

    def run_benchmark(self, config: str, samples: int = 200, depths: List[int] = [0]):
        print(f"\nBenchmarking {config} | Samples: {samples}...")
        
        # Syzygy pieces
        sides = config.upper().split('V')
        w_syms = list(sides[0])
        b_syms = list(sides[1])
        
        w_pieces = [chess.Piece(chess.PIECE_SYMBOLS.index(s.lower()), chess.WHITE) for s in w_syms]
        b_pieces = [chess.Piece(chess.PIECE_SYMBOLS.index(s.lower()), chess.BLACK) for s in b_syms]
        all_pieces = w_pieces + b_pieces
        
        import random
        valid_boards = []
        while len(valid_boards) < samples:
            board = chess.Board(None)
            squares = random.sample(chess.SQUARES, len(all_pieces))
            for i, piece in enumerate(all_pieces):
                board.set_piece_at(squares[i], piece)
            
            # Pawn check
            if any(board.piece_at(sq) and board.piece_at(sq).piece_type == chess.PAWN and 
                   chess.square_rank(sq) in [0, 7] for sq in chess.SQUARES):
                continue
            
            board.turn = random.choice([chess.WHITE, chess.BLACK])
            if board.is_valid():
                try:
                    true_wdl_raw = self.tablebase.probe_wdl(board)
                    # map -2,0,2 to 0,1,2
                    true_wdl = (true_wdl_raw + 2) // 2
                    valid_boards.append((board.copy(), true_wdl))
                except: continue
        
        for depth in depths:
            correct = 0
            start_time = time.time()
            for b, target in valid_boards:
                pred = self.get_wdl_decision(b, depth)
                if pred == target:
                    correct += 1
            
            acc = correct / len(valid_boards)
            elapsed = time.time() - start_time
            print(f"Depth {depth} | Accuracy: {acc:.2%} ({correct}/{len(valid_boards)}) | Time: {elapsed:.1f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="data/models/mlp_universal_v6_tactical_large_scale_best.pth")
    parser.add_argument("--syzygy", type=str, default="syzygy")
    parser.add_argument("--config", type=str, default="KRPvK")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--depths", type=str, default="0,1,2")
    args = parser.parse_args()
    
    # Set PYTHONPATH
    import sys
    sys.path.append(os.getcwd())
    
    depths = [int(d) for d in args.depths.split(',')]
    searcher = NeuralSearcherV6(args.model, args.syzygy)
    searcher.run_benchmark(args.config, args.samples, depths=depths)
