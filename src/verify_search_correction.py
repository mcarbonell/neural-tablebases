import chess
import chess.syzygy
import onnxruntime as ort
import numpy as np
import os
import argparse
import random
import random
from generate_datasets import encode_board, get_material_config

class NeuralMinimax:
    def __init__(self, onnx_path, syzygy_path, version=5, target_config="KPvKP"):
        self.sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.sess.get_inputs()[0].name
        self.tablebase = chess.syzygy.open_tablebase(syzygy_path)
        self.version = version
        self.rel_arg = f"v{version}"
        self.target_config = target_config
        
    def evaluate_nn_white(self, board):
        """Returns WDL as a soft score [0, 2] from White's perspective."""
        if board.is_game_over():
            res = board.result()
            if res == "1-0": return 2.0
            if res == "0-1": return 0.0
            return 1.0 # Draw
            
        # --- CASCADE ORACLE FALLBACK ---
        # 1. Check piece count
        num_current_pieces = len(board.piece_map())
        # 2. Check piece types (New: Promotion Support)
        current_config = get_material_config(board)
        
        # If it's not the exact configuration the NN was trained for (e.g. KPvKP),
        # then we must fallback to Syzygy (Oracle).
        if current_config != self.target_config:
            wdl_raw = self.tablebase.probe_wdl(board)
            if wdl_raw is None: return 1.0 # Safe default
            wdl_val = 0 if wdl_raw == -2 else (1 if wdl_raw == 0 else 2)
            if board.turn == chess.BLACK:
                wdl_val = 2 - wdl_val
            return float(wdl_val)
        # -------------------------------

        # NN expects STM perspective
        encoding = encode_board(board, relative=self.rel_arg)
        # Verify size matches ONNX metadata (68 for 4-piece, 45 for 3-piece)
        if encoding.shape[0] != 68:
            # High-level safety: if mismatch persists, fallback to probe
            wdl_raw = self.tablebase.probe_wdl(board)
            return float(0 if wdl_raw == -2 else (1 if wdl_raw == 0 else 2))
            
        inp = encoding.reshape(1, -1).astype(np.float32)
        logits = self.sess.run(None, {self.input_name: inp})[0][0]
        
        # Softmax for smoother minimax values
        probs = np.exp(logits) / np.sum(np.exp(logits))
        score_stm = probs[1] * 1.0 + probs[2] * 2.0
        
        # Convert to White's perspective
        if board.turn == chess.BLACK:
            return 2.0 - score_stm
        return score_stm

    def minimax(self, board, depth, alpha, beta, is_max):
        if depth == 0 or board.is_game_over():
            return self.evaluate_nn_white(board)
            
        if is_max:
            val = -float('inf')
            for move in board.legal_moves:
                board.push(move)
                val = max(val, self.minimax(board, depth - 1, alpha, beta, False))
                board.pop()
                alpha = max(alpha, val)
                if beta <= alpha: break
            return val
        else:
            val = float('inf')
            for move in board.legal_moves:
                board.push(move)
                val = min(val, self.minimax(board, depth - 1, alpha, beta, True))
                board.pop()
                beta = min(beta, val)
                if beta <= alpha: break
            return val

    def get_wdl(self, board, depth=0):
        # Result from White's perspective
        score = self.minimax(board, depth, -float('inf'), float('inf'), board.turn == chess.WHITE)
        wdl_white = int(round(score))
        # Convert back to STM (Syzygy style)
        if board.turn == chess.BLACK:
            return 2 - wdl_white
        return wdl_white

def verify_endgame(config, onnx_path, syzygy_path, version=5, limit=100000):
    searcher = NeuralMinimax(onnx_path, syzygy_path, version, target_config=config)
    tablebase = chess.syzygy.open_tablebase(syzygy_path)
    
    # Configure pieces based on shorthand (e.g., KPvKP)
    white_part, black_part = config.split('v')
    
    def decode_pieces(part, color):
        p_list = []
        for char in part:
            if char == 'K': p_list.append(chess.Piece(chess.KING, color))
            elif char == 'Q': p_list.append(chess.Piece(chess.QUEEN, color))
            elif char == 'R': p_list.append(chess.Piece(chess.ROOK, color))
            elif char == 'B': p_list.append(chess.Piece(chess.BISHOP, color))
            elif char == 'N': p_list.append(chess.Piece(chess.KNIGHT, color))
            elif char == 'P': p_list.append(chess.Piece(chess.PAWN, color))
        return p_list

    all_pieces = decode_pieces(white_part, chess.WHITE) + decode_pieces(black_part, chess.BLACK)
    num_pieces = len(all_pieces)
    
    print(f"\nAuditing accuracy for {config} via Search (D0, D1, D2)...")
    print(f"ONNX Model: {onnx_path}")
    
    counts = {"total": 0, "d0_errors": 0, "d1_errors": 0, "d2_errors": 0}
    failed_fens = []

    # For 4+ pieces, exhaustive search is too slow. Use randomized sampling.
    while counts["total"] < limit:
        # Generate valid random position
        board = chess.Board(None)
        squares = random.sample(range(64), num_pieces)
        
        # Check pawn ranks
        valid_pawns = True
        for i, piece in enumerate(all_pieces):
            if piece.piece_type == chess.PAWN:
                rank = chess.square_rank(squares[i])
                if rank == 0 or rank == 7:
                    valid_pawns = False; break
        if not valid_pawns: continue
        
        for i, piece in enumerate(all_pieces):
            board.set_piece_at(squares[i], piece)
            
        for turn in [chess.WHITE, chess.BLACK]:
            board.turn = turn
            if not board.is_valid(): continue
            
            # Additional check for Syzygy: No self-check
            if board.was_into_check(): continue
            
            counts["total"] += 1
            if counts["total"] % 5000 == 0:
                print(f"Processed {counts['total']}... D0 Errors: {counts['d0_errors']} | D1 Errors: {counts['d1_errors']}")

            # Syzygy label
            wdl_true_raw = tablebase.probe_wdl(board)
            if wdl_true_raw is None: continue
            
            # Map Syzygy WDL to [0, 1, 2]
            wdl_true = 0 if wdl_true_raw == -2 else (1 if wdl_true_raw == 0 else 2)
            
            # Test D0
            d0_pred = searcher.get_wdl(board, depth=0)
            if d0_pred != wdl_true:
                counts["d0_errors"] += 1
                
                # Check D1
                d1_pred = searcher.get_wdl(board, depth=1)
                if d1_pred != wdl_true:
                    counts["d1_errors"] += 1
                    
                    # Check D2
                    d2_pred = searcher.get_wdl(board, depth=2)
                    if d2_pred != wdl_true:
                        counts["d2_errors"] += 1
                        failed_fens.append(board.fen())
                
            if counts["total"] >= limit: break

    print("\n" + "="*50)
    print(f"SEARCH CORRECTOR PERFORMANCE ({config})")
    print("="*50)
    print(f"Total positions: {counts['total']}")
    print(f"Raw NN (D0) Accuracy: {(1 - counts['d0_errors']/counts['total'])*100:.4f}% ({counts['d0_errors']} errors)")
    print(f"D1 Search Accuracy: {(1 - counts['d1_errors']/counts['total'])*100:.4f}% ({counts['d1_errors']} errors)")
    print(f"D2 Search Accuracy: {(1 - counts['d2_errors']/counts['total'])*100:.4f}% ({counts['d2_errors']} errors)")
    print("="*50)
    
    if failed_fens:
        print("\nTOP 20 FENs that still fail at Depth 2:")
        for fen in failed_fens[:20]:
            print(fen)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="KPvKP")
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--syzygy", type=str, default="syzygy")
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--version", type=int, default=5)
    args = parser.parse_args()
    
    verify_endgame(args.config, args.onnx, args.syzygy, args.version, limit=args.limit)
