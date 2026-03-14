import torch
import chess
import chess.syzygy
import numpy as np
import argparse
import os
import time
from models import get_model_for_endgame
from generate_datasets import encode_board

class FenAnalyzer:
    def __init__(self, model_path, syzygy_path, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Loading model: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Peek input size and WDL classes
        first_key = next(iter(checkpoint.keys()))
        input_size = checkpoint[first_key].shape[1]
        
        # Detect WDL classes from wdl_head weights
        wdl_key = 'wdl_head.bias'
        if wdl_key in checkpoint:
            self.num_wdl_classes = checkpoint[wdl_key].shape[0]
        else:
            self.num_wdl_classes = 3 # fallback
            
        # Determine num_pieces and encoding version
        if input_size == 68:
            num_pieces = 4
            self.version = 4
        elif input_size == 65:
            num_pieces = 4
            self.version = 1
        elif input_size == 95:
            num_pieces = 5
            self.version = 4
        elif input_size == 91:
            num_pieces = 5
            self.version = 1
        else:
            num_pieces = input_size // 64
            self.version = 0 # unknown

        print(f"Detected: {num_pieces} pieces, V{self.version} encoding, {self.num_wdl_classes} WDL classes")
        
        self.model = get_model_for_endgame('mlp', num_pieces, input_size=input_size, num_wdl_classes=self.num_wdl_classes).to(self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        self.tablebase = None
        if os.path.exists(syzygy_path):
            self.tablebase = chess.syzygy.open_tablebase(syzygy_path)

    def evaluate(self, board):
        rel_arg = (f"v4" if self.version == 4 else True)
        # Use find_canonical_from if standardizing, but here we just encode
        features = encode_board(board, compact=True, relative=rel_arg)
        x = torch.from_numpy(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            wdl_logits, dtz_out = self.model(x)
            wdl_probs = torch.softmax(wdl_logits, dim=1).cpu().numpy()[0]
            dtz = dtz_out.cpu().numpy()[0][0]
            
        pred_wdl_idx = np.argmax(wdl_probs)
        return pred_wdl_idx, wdl_probs, dtz

    def minimax(self, board, depth, alpha, beta, is_maximizing):
        if depth == 0 or board.is_game_over():
            idx, probs, dtz = self.evaluate(board)
            if self.num_wdl_classes == 5:
                # 0: Loss, 1: Blessed Loss, 2: Draw, 3: Cursed Win, 4: Win
                score = (probs[4] * 2.0 + probs[3] * 1.9 + probs[2] * 1.0 + probs[1] * 0.1)
            else:
                # 0: Loss, 1: Draw, 2: Win
                score = (probs[2] * 2.0 + probs[1] * 1.0)
            return score

        if is_maximizing:
            max_eval = -float('inf')
            for move in board.legal_moves:
                board.push(move)
                ev = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                score = 2.0 - ev if ev is not None else 1.0
                max_eval = max(max_eval, score)
                alpha = max(alpha, score)
                if beta <= alpha: break
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                ev = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                score = 2.0 - ev if ev is not None else 1.0
                min_eval = min(min_eval, score)
                beta = min(beta, score)
                if beta <= alpha: break
            return min_eval

    def analyze(self, fen, max_depth=3):
        board = chess.Board(fen)
        print("\n" + "="*60)
        print(f"FEN: {fen}")
        print(board)
        print("="*60)
        
        if self.tablebase:
            try:
                true_wdl = self.tablebase.probe_wdl(board)
                true_dtz = self.tablebase.probe_dtz(board)
                wdl_str = {
                    -2: "Loss", 
                    -1: "Cursed Loss" if self.num_wdl_classes == 5 else "Loss", 
                    0: "Draw", 
                    1: "Blessed Win" if self.num_wdl_classes == 5 else "Win", 
                    2: "Win"
                }.get(true_wdl, "Unknown")
                print(f"SYZYGY: {wdl_str} (WDL={true_wdl}, DTZ={true_dtz})")
            except:
                print("SYZYGY: Position not found in tablebase")
        
        # Pure Neural
        wdl_idx, probs, dtz = self.evaluate(board)
        if self.num_wdl_classes == 5:
            wdl_str = ["Loss", "Cursed Loss", "Draw", "Blessed Win", "Win"][wdl_idx]
            prob_str = f"L={probs[0]:.4f}, CL={probs[1]:.4f}, D={probs[2]:.4f}, BW={probs[3]:.4f}, W={probs[4]:.4f}"
        else:
            wdl_str = ["Loss", "Draw", "Win"][wdl_idx]
            prob_str = f"L={probs[0]:.4f}, D={probs[1]:.4f}, W={probs[2]:.4f}"
            
        print(f"NEURAL (Depth 0): {wdl_str}")
        print(f"  Probs: {prob_str}")
        print(f"  DTZ: {dtz:.2f}")

        # Search
        for d in range(1, max_depth + 1):
            best_move = None
            best_score = -float('inf')
            
            moves = list(board.legal_moves)
            for move in moves:
                board.push(move)
                # After push, it's opponent's turn. 
                # minimax(opponent) returns opponent's score.
                # Our score is 2 - opponent_score
                score = 2 - self.minimax(board, d - 1, -float('inf'), float('inf'), False)
                board.pop()
                
                if score > best_score:
                    best_score = score
                    best_move = move
            
            wdl_idx = int(round(best_score))
            wdl_str = ["Loss", "Draw", "Win"][np.clip(wdl_idx, 0, 2)]
            print(f"SEARCH (Depth {d}): {wdl_str} (Score={best_score:.4f}), Best Move: {best_move}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fen", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--syzygy", type=str, default="syzygy")
    parser.add_argument("--depth", type=int, default=3)
    args = parser.parse_args()
    
    analyzer = FenAnalyzer(args.model, args.syzygy)
    analyzer.analyze(args.fen, max_depth=args.depth)

if __name__ == "__main__":
    main()
