import chess
import chess.syzygy
import onnxruntime as ort
import numpy as np
import os
import argparse
from generate_datasets import encode_board

class NeuralMinimax:
    def __init__(self, onnx_path, syzygy_path, version=5):
        self.sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.sess.get_inputs()[0].name
        self.tablebase = chess.syzygy.open_tablebase(syzygy_path)
        self.version = version
        self.rel_arg = f"v{version}"
        
    def evaluate_nn_white(self, board):
        """Returns WDL as a soft score [0, 2] from White's perspective."""
        if board.is_game_over():
            res = board.result()
            if res == "1-0": return 2.0
            if res == "0-1": return 0.0
            return 1.0 # Draw
            
        # NN expects STM perspective (STM is White on flipped board)
        encoding = encode_board(board, relative=self.rel_arg)
        inp = encoding.reshape(1, -1).astype(np.float32)
        logits = self.sess.run(None, {self.input_name: inp})[0][0]
        
        # Softmax for smoother minimax values
        probs = np.exp(logits) / np.sum(np.exp(logits))
        score_stm = probs[1] * 1.0 + probs[2] * 2.0 # 0*p0 + 1*p1 + 2*p2
        
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

def verify_all_kpvk(onnx_path, syzygy_path, version=5, limit=None):
    searcher = NeuralMinimax(onnx_path, syzygy_path, version)
    tablebase = chess.syzygy.open_tablebase(syzygy_path)
    all_pieces = [chess.Piece(chess.KING, chess.WHITE), chess.Piece(chess.PAWN, chess.WHITE), chess.Piece(chess.KING, chess.BLACK)]
    
    import itertools
    print(f"Auditing KPvK accuracy via Search (D0, D1, D2)...")
    
    counts = {
        "total": 0,
        "d0_errors": 0,
        "d1_errors": 0,
        "d2_errors": 0
    }
    
    # Track failed FENs for analysis
    failed_fens = []

    # To be faster, we sample or use a smaller set if limit is set
    # But for 3-piece, 165k is fine if we are parallel or fast?
    # Minimax depth 2 is slow in pure Python.
    
    for squares in itertools.permutations(chess.SQUARES, 3):
        board = chess.Board(None)
        for i, piece in enumerate(all_pieces):
            board.set_piece_at(squares[i], piece)
            
        if any(board.piece_at(sq) and board.piece_at(sq).piece_type == chess.PAWN and chess.square_rank(sq) in [0, 7] for sq in chess.SQUARES):
            continue
            
        for turn in [chess.WHITE, chess.BLACK]:
            board.turn = turn
            if not board.is_valid(): continue
            
            counts["total"] += 1
            if counts["total"] % 5000 == 0:
                print(f"Processed {counts['total']}... D0 Errors: {counts['d0_errors']} | D1 Errors: {counts['d1_errors']}")

            # Syzygy label
            wdl_true_raw = tablebase.probe_wdl(board)
            wdl_true = 0 if wdl_true_raw == -2 else (1 if wdl_true_raw == 0 else 2)
            
            # Test D0
            d0_pred = searcher.get_wdl(board, depth=0)
            if d0_pred != wdl_true:
                counts["d0_errors"] += 1
                
                # Check D1 (only if D0 failed, to be fast)
                d1_pred = searcher.get_wdl(board, depth=1)
                if d1_pred != wdl_true:
                    counts["d1_errors"] += 1
                    
                    # Check D2 (only if D1 failed)
                    d2_pred = searcher.get_wdl(board, depth=2)
                    if d2_pred != wdl_true:
                        counts["d2_errors"] += 1
                        failed_fens.append(board.fen())
                
            if limit and counts["total"] >= limit:
                break
        if limit and counts["total"] >= limit:
            break
            
    print("\n" + "="*40)
    print("SEARCH CORRECTOR PERFORMANCE (KPvK)")
    print("="*40)
    print(f"Total positions: {counts['total']}")
    print(f"Raw NN (D0) Accuracy: {(1 - counts['d0_errors']/counts['total'])*100:.4f}% ({counts['d0_errors']} errors)")
    print(f"D1 Search Accuracy: {(1 - counts['d1_errors']/counts['total'])*100:.4f}% ({counts['d1_errors']} errors)")
    print(f"D2 Search Accuracy: {(1 - counts['d2_errors']/counts['total'])*100:.4f}% ({counts['d2_errors']} errors)")
    print("="*40)
    
    if failed_fens:
        print("\nFENs that still fail at Depth 2:")
        for fen in failed_fens[:10]:
            print(fen)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="data/mlp_kpvk_v5.onnx")
    parser.add_argument("--syzygy", type=str, default="syzygy")
    parser.add_argument("--limit", type=int, default=100000) # Full is ~165k
    args = parser.parse_args()
    
    verify_all_kpvk(args.onnx, args.syzygy, limit=args.limit)
