# Diagnostic script to check failing FENs at various depths
import chess
import chess.syzygy
import onnxruntime as ort
import numpy as np
import os
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
        # Softmax to get probs for [Loss, Draw, Win]
        probs = np.exp(logits) / np.sum(np.exp(logits))
        # Expected value from White's perspective
        score_stm = probs[1] * 1.0 + probs[2] * 2.0
        # If it's Black's turn, the NN output is from Black's perspective
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

# Failing FENs from E328 audit (200k positions)
failing_fens = [
    "3K4/2p5/4P3/8/8/8/3k4/8 b - - 0 1",
    "8/8/4K1P1/8/8/3p4/2k5/8 b - - 0 1",
    "8/8/8/3P4/p7/6K1/8/5k2 w - - 0 1",
    "5K2/8/8/8/5P2/3k2p1/8/8 w - - 0 1",
    "1K6/8/k7/7P/6p1/8/8/8 b - - 0 1",
    "8/7k/5K2/8/3P2p1/8/8/8 w - - 0 1",
    "8/8/8/3p4/8/6P1/K1k5/8 w - - 0 1",
    "1K6/8/1k3P2/8/8/3p4/8/8 b - - 0 1",
    "8/8/8/1P3K2/p6k/8/8/8 w - - 0 1",
    "5K2/8/5k2/6p1/8/8/6P1/8 w - - 0 1",
    "8/2K5/1P6/8/8/1p6/8/1k6 w - - 0 1",
    "8/K2k4/8/5p2/P7/8/8/8 w - - 0 1",
    "8/8/5P1K/8/2p2k2/8/8/8 b - - 0 1",
    "8/8/K1P5/8/5p2/3k4/8/8 w - - 0 1",
    "8/8/8/7P/3p4/7K/8/5k2 b - - 0 1",
    "8/8/3P4/7k/5K2/p7/8/8 w - - 0 1",
    "8/8/5P2/5K2/8/5k1p/8/8 w - - 0 1",
    "8/8/1P6/8/4p3/8/5k2/7K b - - 0 1",
    "5K2/8/3P3k/8/8/7p/8/8 w - - 0 1",
    "8/8/2P5/8/p7/4K3/8/4k3 b - - 0 1"
]

if __name__ == "__main__":
    onnx_path = "data/kpvkp_v5_eval_e328.onnx"
    syzygy_path = "syzygy/"
    searcher = NeuralMinimax(onnx_path, syzygy_path, version=5, target_config="KPvKP")
    
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
