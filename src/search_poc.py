import chess
import chess.syzygy
import torch
import numpy as np
import os
import json
import argparse
import time
from typing import List, Tuple, Dict, Optional
from models import get_model_for_endgame
from generate_datasets import encode_board, flip_board

class NeuralSearcher:
    def __init__(self, model_path: str, syzygy_path: str, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Loading model from {model_path} on {self.device}...")

        self.model_path = model_path
        self.model_metadata = None
        meta_path = model_path.replace(".pth", "_metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.model_metadata = json.load(f)
            except Exception as e:
                print(f"Warning: failed to load model metadata {meta_path}: {e}")
                self.model_metadata = None

        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Peek into weights to see input size
        first_key = 'backbone.0.weight'
        if first_key not in checkpoint:
            # Maybe it's a SIREN model or has different naming
            first_key = next(iter(checkpoint.keys()))
            
        weights = checkpoint[first_key]
        input_size = weights.shape[1]

        # Infer number of WDL classes from the checkpoint (wdl_head shape)
        num_wdl_classes = 3
        if "wdl_head.weight" in checkpoint and hasattr(checkpoint["wdl_head.weight"], "shape"):
            num_wdl_classes = int(checkpoint["wdl_head.weight"].shape[0])
        elif self.model_metadata and "dataset" in self.model_metadata:
            num_wdl_classes = int(self.model_metadata["dataset"].get("num_wdl_classes", num_wdl_classes))
        
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
        self.num_wdl_classes = num_wdl_classes

        # Canonicalization (optional): if the model was trained on canonicalized inputs,
        # we must apply the same canonicalization before encoding at inference time.
        self.canonical = False
        self.canonical_mode = "auto"
        if self.model_metadata and isinstance(self.model_metadata.get("dataset_metadata"), dict):
            ds_meta = self.model_metadata["dataset_metadata"]
            self.canonical = bool(ds_meta.get("canonical", False))
            self.canonical_mode = str(ds_meta.get("canonical_mode", "auto"))

        # Detect model type (mlp vs siren) from checkpoint keys if metadata isn't available
        model_type = "mlp"
        if self.model_metadata and isinstance(self.model_metadata.get("args"), dict):
            model_type = str(self.model_metadata["args"].get("model", model_type))
        else:
            if any(k.startswith("net.") for k in checkpoint.keys()):
                model_type = "siren"
        
        self.model = get_model_for_endgame(model_type, num_pieces, num_wdl_classes=num_wdl_classes,
                                           input_size=input_size).to(self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        print(f"Model loaded. Config: {num_pieces} pieces, input_size={input_size}, "
              f"relative={use_relative_encoding}, wdl_classes={num_wdl_classes}, "
              f"encoding_v={self.encoding_version}, canonical={self.canonical}")
        self.tablebase = chess.syzygy.open_tablebase(syzygy_path)
        
        # Expected material (to detect promotions/material changes)
        self.expected_pieces = None

    def _invert_perspective(self, wdl_score: float) -> float:
        """Convert WDL score between players (e.g., Loss<->Win)."""
        return float(self.num_wdl_classes - 1) - wdl_score

    def _prepare_board_for_encoding(self, board: chess.Board) -> chess.Board:
        """
        Apply the same normalization/canonicalization the model expects.
        Returns a (possibly new) board object.
        """
        out = board

        # v4 encoding normalizes side-to-move to White via flip_board when BLACK.
        if self.encoding_version == 4 and out.turn == chess.BLACK:
            out = flip_board(out)

        if self.canonical:
            from canonical_forms import find_canonical_form
            out, _ = find_canonical_form(out, lambda b: (), mode=self.canonical_mode)

        return out
        
    def evaluate_nn(self, board: chess.Board) -> Tuple[float, float]:
        # Count pieces
        piece_map = board.piece_map()
        num_on_board = len(piece_map)
        
        # Determine current material (ignoring color for piece types)
        # We use a sorted tuple of piece types as a simplified material signature
        current_pieces = sorted([p.piece_type for p in piece_map.values()])

        # If this is the first evaluation, establish expected material
        if self.expected_pieces is None:
            self.expected_pieces = current_pieces

        # Handle terminal states with perfect information
        if board.is_game_over():
            res = board.result()
            if res == "1-0":
                wdl_white = float(self.num_wdl_classes - 1)
            elif res == "0-1":
                wdl_white = 0.0
            else:
                # Draw (usually class 1 in 3-class, or 2 in 5-class)
                wdl_white = 1.0 if self.num_wdl_classes == 3 else 2.0
            return wdl_white, 0.0

        if num_on_board != self.num_pieces or current_pieces != self.expected_pieces:
            # Material changed (capture or promotion)
            # Try to use Syzygy if available, otherwise assume draw
            try:
                true_wdl = self.tablebase.probe_wdl(board)
                if self.num_wdl_classes == 5:
                    if true_wdl == -2: mapped_wdl = 0
                    elif true_wdl == -1: mapped_wdl = 1
                    elif true_wdl == 0: mapped_wdl = 2
                    elif true_wdl == 1: mapped_wdl = 3
                    else: mapped_wdl = 4
                else:
                    if true_wdl < 0: mapped_wdl = 0  # -2 and -1
                    elif true_wdl == 0: mapped_wdl = 1
                    else: mapped_wdl = 2  # 1 and 2
                
                # Convert to White's perspective
                if board.turn == chess.BLACK:
                    return self._invert_perspective(float(mapped_wdl)), 0.0
                return float(mapped_wdl), 0.0
            except:
                # Fallback to Draw
                return (1.0 if self.num_wdl_classes == 3 else 2.0), 0.0
            
        original_turn = board.turn
        board_enc = self._prepare_board_for_encoding(board)

        relative_arg = False
        if self.use_relative_encoding:
            relative_arg = ('v4' if self.encoding_version == 4 else True)

        x = encode_board(board_enc, relative=relative_arg,
                         use_move_distance=(self.encoding_version == 3))
        x_tensor = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            wdl_logits, dtz_val = self.model(x_tensor)
            # Use softmax to get a more granular score instead of just argmax
            probs = torch.softmax(wdl_logits, dim=1).squeeze(0)
            
            # Calculate expected WDL value
            score_stm = 0.0
            for i in range(self.num_wdl_classes):
                score_stm += probs[i].item() * i
            
            dtz = dtz_val.item()
            
        # Convert from side-to-move perspective to White's perspective for minimax.
        wdl_white = self._invert_perspective(score_stm) if original_turn == chess.BLACK else score_stm
            
        return wdl_white, dtz

    def minimax(self, board: chess.Board, depth: int, alpha: float, beta: float, is_maximizing: bool) -> float:
        if depth == 0 or board.is_game_over():
            wdl_score, _ = self.evaluate_nn(board)
            return wdl_score

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
        
        # Convert back to side-to-move perspective (Syzygy convention)
        if board.turn == chess.BLACK:
            return self._invert_perspective(wdl_white)
        return int(wdl_white)

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
                        if self.num_wdl_classes == 5:
                            if true_wdl == -2: mapped_wdl = 0
                            elif true_wdl == -1: mapped_wdl = 1
                            elif true_wdl == 0: mapped_wdl = 2
                            elif true_wdl == 1: mapped_wdl = 3
                            else: mapped_wdl = 4
                        else:
                            if true_wdl < 0: mapped_wdl = 0
                            elif true_wdl == 0: mapped_wdl = 1
                            else: mapped_wdl = 2
                        valid_boards.append((board.copy(), mapped_wdl))
                        if len(valid_boards) >= samples: break
                    except:
                        continue
            if len(valid_boards) >= samples: break
            
        print(f"Found {len(valid_boards)} positions. Starting evaluation...")
        
        results = {d: {"correct": 0, "total": 0} for d in depths}
        
        for i, (board, true_wdl) in enumerate(valid_boards):
            # Reset expected pieces for each new position in accuracy verification
            self.expected_pieces = None
            
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
