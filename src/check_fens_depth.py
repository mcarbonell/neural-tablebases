import chess
import chess.syzygy
import onnxruntime as ort
import numpy as np
import os
from generate_datasets import encode_board

class NeuralMinimax:
    def __init__(self, onnx_path, syzygy_path, version=5):
        self.sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.sess.get_inputs()[0].name
        self.tablebase = chess.syzygy.open_tablebase(syzygy_path)
        self.version = version
        self.rel_arg = f"v{version}"
        
    def evaluate_nn_white(self, board):
        if board.is_game_over():
            res = board.result()
            if res == "1-0": return 2.0
            if res == "0-1": return 0.0
            return 1.0
            
        # Fallback if pieces captured
        num_pieces = len(board.piece_map())
        if num_pieces < 4:
            wdl_raw = self.tablebase.probe_wdl(board)
            if wdl_raw is None: return 1.0
            val = 0 if wdl_raw == -2 else (1 if wdl_raw == 0 else 2)
            return float(2 - val if board.turn == chess.BLACK else val)

        encoding = encode_board(board, relative=self.rel_arg)
        if encoding.shape[0] != 68:
            wdl_raw = self.tablebase.probe_wdl(board)
            return float(0 if wdl_raw == -2 else (1 if wdl_raw == 0 else 2))
            
        inp = encoding.reshape(1, -1).astype(np.float32)
        logits = self.sess.run(None, {self.input_name: inp})[0][0]
        probs = np.exp(logits) / np.sum(np.exp(logits))
        score_stm = probs[1] * 1.0 + probs[2] * 2.0
        return 2.0 - score_stm if board.turn == chess.BLACK else score_stm

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
        return 2 - wdl_white if board.turn == chess.BLACK else wdl_white

fens = [
    "8/2p5/8/6KP/8/8/8/4k3 b - - 0 1",
    "8/8/1P1K4/8/5k2/2p5/8/8 b - - 0 1",
    "8/8/7K/1P1p4/7k/8/8/8 b - - 0 1",
    "7K/3p4/8/8/4k3/P7/8/8 w - - 0 1",
    "8/1p6/8/6K1/P7/8/8/3k4 w - - 0 1",
    "8/4P3/8/K7/8/1k6/3p4/8 w - - 0 1",
    "8/8/8/8/p1P5/8/k2K4/8 w - - 0 1",
    "8/8/8/2PK1p1k/8/8/8/8 b - - 0 1",
    "6K1/8/7k/8/1P2p3/8/8/8 w - - 0 1",
    "k7/8/2K4p/8/8/8/3P4/8 w - - 0 1"
]

searcher = NeuralMinimax("data/kpvkp_v5_eval.onnx", "syzygy")
tablebase = chess.syzygy.open_tablebase("syzygy")

print(f"{'FEN':<35} | Syzygy | D0 | D2 | D4 | D6")
print("-" * 65)

for fen in fens:
    board = chess.Board(fen)
    true_raw = tablebase.probe_wdl(board)
    true_wdl = 0 if true_raw == -2 else (1 if true_raw == 0 else 2)
    
    d0 = searcher.get_wdl(board, 0)
    d2 = searcher.get_wdl(board, 2)
    d4 = searcher.get_wdl(board, 4)
    d6 = searcher.get_wdl(board, 6)
    
    def fmt(val):
        return "OK" if val == true_wdl else "FAIL"
        
    print(f"{fen[:35]:<35} | {true_wdl}      | {fmt(d0)} | {fmt(d2)} | {fmt(d4)} | {fmt(d6)}")
