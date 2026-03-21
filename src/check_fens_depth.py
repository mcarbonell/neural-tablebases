# Diagnostic script for KQvKP failing FENs
import chess
import chess.syzygy
import onnxruntime as ort
import numpy as np
import os
from generate_datasets import encode_board, get_material_config

class NeuralMinimax:
    def __init__(self, onnx_path, syzygy_path, version=5, target_config="KQvKP"):
        self.sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.sess.get_inputs()[0].name
        self.tablebase = chess.syzygy.open_tablebase(syzygy_path)
        self.version = version
        self.rel_arg = f"v{version}"
        self.target_config = target_config
        
    def evaluate_nn_white(self, board):
        if board.is_game_over():
            res = board.result()
            if res == "1-0": return 2.0
            if res == "0-1": return 0.0
            return 1.0
            
        current_config = get_material_config(board)
        if current_config != self.target_config:
            wdl_raw = self.tablebase.probe_wdl(board)
            if wdl_raw is None: return 1.0
            val = 0 if wdl_raw == -2 else (1 if wdl_raw == 0 else 2)
            if board.turn == chess.BLACK:
                val = 2 - val
            return float(val)

        encoding = encode_board(board, relative=self.rel_arg)
        inp = encoding.reshape(1, -1).astype(np.float32)
        logits = self.sess.run(None, {self.input_name: inp})[0][0]
        probs = np.exp(logits) / np.sum(np.exp(logits))
        score_stm = probs[1] * 1.0 + probs[2] * 2.0
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
        score = self.minimax(board, depth, -float('inf'), float('inf'), board.turn == chess.WHITE)
        wdl_white = int(round(score))
        return wdl_white

# Failing FENs from E189 audit (200k positions)
failing_fens = [
    "8/7Q/8/2K5/8/8/5p2/5k2 w - - 0 1",
    "8/2Q5/8/3K4/8/5p2/8/5k2 b - - 0 1",
    "8/8/8/2K1Q3/8/2k5/5p2/8 b - - 0 1",
    "3Q4/3K4/8/8/5p2/8/4k3/8 b - - 0 1",
    "8/8/8/4K3/8/5pk1/3Q4/8 b - - 0 1",
    "8/8/3K2Q1/8/8/5p2/8/5k2 b - - 0 1"
]

if __name__ == "__main__":
    onnx_path = "data/kqvkp_v5_eval_e189.onnx"
    syzygy_path = "syzygy/"
    # We pass the target config so it knows when pieces have been captured/promoted
    searcher = NeuralMinimax(onnx_path, syzygy_path, version=5, target_config="KQvKP")
    
    print(f"{'FEN':<35} | Syzygy | D0 | D2 | D4 | D6")
    print("-" * 65)
    
    for fen in failing_fens:
        board = chess.Board(fen)
        true_raw = searcher.tablebase.probe_wdl(board)
        true_wdl = 0 if true_raw == -2 else (1 if true_raw == 0 else 2)
        
        results = []
        for d in [0, 2, 4, 6]:
            res = searcher.get_wdl(board, d)
            results.append("OK" if res == true_wdl else "FAIL")
        
        print(f"{fen[:35]:<35} | {true_wdl}      | {' | '.join(results)}")
