import chess
import chess.syzygy
import onnxruntime as ort
import numpy as np
import argparse
from typing import Tuple, Optional

class DTZNeuralMinimax:
    def __init__(self, onnx_path: str, syzygy_path: str, target_config: str = "KBNvK"):
        self.sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.sess.get_inputs()[0].name
        self.tablebase = chess.syzygy.open_tablebase(syzygy_path)
        
        # --- Parche para compatibilidad de python-chess con tablas como KBNvK.rtbw ---
        for k in list(self.tablebase.wdl.keys()):
            if k.endswith('vK') and k[:-1] not in self.tablebase.wdl:
                self.tablebase.wdl[k[:-1]] = self.tablebase.wdl[k]
        for k in list(self.tablebase.dtz.keys()):
            if k.endswith('vK') and k[:-1] not in self.tablebase.dtz:
                self.tablebase.dtz[k[:-1]] = self.tablebase.dtz[k]
                
        self.target_config = target_config
        
    def encode_board(self, board: chess.Board) -> np.ndarray:
        from generate_datasets import encode_board
        # v5 is the current universal architecture standard
        enc = encode_board(board, relative="v5")
        # Pad up to 68 for 3-piece captures evaluated by a 4-piece Universal Model
        if len(enc) < 68:
            padding = np.zeros(68 - len(enc), dtype=np.float32)
            enc = np.concatenate([enc, padding])
        return enc

    def evaluate_state(self, board: chess.Board) -> Tuple[float, float]:
        """Returns (WDL_stm, DTZ_stm) from the side-to-move's perspective."""
        if board.is_game_over():
            res = board.result()
            if res == "1/2-1/2":
                return 0.0, 0.0  # Draw
            if board.is_check():
                return -1.0, -1.0  # Loss (STM is mated)
            return 0.0, 0.0  # Stalemate (Draw)

        encoding = self.encode_board(board)
        inp = encoding.reshape(1, -1).astype(np.float32)
        out = self.sess.run(None, {self.input_name: inp})
        
        # WDL
        logits = out[0][0]
        probs = np.exp(logits) / np.sum(np.exp(logits))
        # Map 3 classes (Loss=0, Draw=1, Win=2) to [-1, 1]
        wdl_stm = probs[2] - probs[0]
        
        # DTZ
        dtz_stm = float(out[1][0][0])
        return wdl_stm, dtz_stm

    def is_better_eval(self, eval_new: Tuple[float, float], eval_best: Optional[Tuple[float, float]]) -> bool:
        """Returns True if eval_new is strictly better than eval_best for the STM."""
        if eval_best is None:
            return True
        w1, d1 = eval_new
        w2, d2 = eval_best
        
        # Buckets: Win (>0.3), Draw (-0.3 to 0.3), Loss (<-0.3)
        b1 = 1 if w1 > 0.3 else (-1 if w1 < -0.3 else 0)
        b2 = 1 if w2 > 0.3 else (-1 if w2 < -0.3 else 0)
        
        if b1 != b2:
            return b1 > b2
            
        # If in the same bucket and not a draw, use DTZ (-dtz is the maximization target)
        if b1 != 0:
            if -d1 > -d2:
                return True
            elif -d1 < -d2:
                return False
                
        # Tiebreaker: Soft WDL
        return w1 > w2

    def negamax(self, board: chess.Board, depth: int) -> Tuple[float, float]:
        if depth == 0 or board.is_game_over():
            return self.evaluate_state(board)
            
        best_eval: Optional[Tuple[float, float]] = None
        
        for move in board.legal_moves:
            board.push(move)
            child_wdl, child_dtz = self.negamax(board, depth - 1)
            board.pop()
            
            # Invert child's perspective to my perspective
            my_wdl = -child_wdl
            my_dtz = -child_dtz + float(np.sign(my_wdl))
            
            my_eval = (my_wdl, my_dtz)
            if self.is_better_eval(my_eval, best_eval):
                best_eval = my_eval
                
        if best_eval is not None:
            return best_eval
        return self.evaluate_state(board)

    def get_best_move(self, board: chess.Board, depth: int=2) -> Tuple[Optional[chess.Move], Optional[Tuple[float, float]]]:
        best_move = None
        best_eval: Optional[Tuple[float, float]] = None
        
        for move in board.legal_moves:
            board.push(move)
            child_wdl, child_dtz = self.negamax(board, depth - 1)
            board.pop()
            
            # Convert
            my_wdl = -child_wdl
            my_dtz = -child_dtz + float(np.sign(my_wdl))
            my_eval = (my_wdl, my_dtz)
            
            if self.is_better_eval(my_eval, best_eval):
                best_eval = my_eval
                best_move = move
                
        return best_move, best_eval

def test_fen_progress(fen: str, onnx_path: str, syzygy_path: str, depth: int=2) -> None:
    board = chess.Board(fen)
    searcher = DTZNeuralMinimax(onnx_path, syzygy_path)
    
    true_wdl = searcher.tablebase.probe_wdl(board)
    true_dtz = searcher.tablebase.probe_dtz(board)
    print(f"\\n--- DIAGNÓSTICO PARA FEN: {fen} ---")
    print(f"STM (Syzygy) WDL_raw: {true_wdl} | DTZ: {true_dtz}")
    
    nn_wdl, nn_dtz = searcher.evaluate_state(board)
    print(f"NN Eval (D0) -> WDL: {nn_wdl:.3f} | DTZ: {nn_dtz:.2f}")
    
    if depth > 0:
        best_move, search_eval = searcher.get_best_move(board, depth)
        
        if best_move is None or search_eval is None:
            print("No legal moves!")
            return
            
        assert search_eval is not None
        print(f"\\nSearch (D{depth}) Best Move: {best_move} -> (WDL: {search_eval[0]:.3f}, DTZ: {search_eval[1]:.2f})")
        
        board.push(best_move)
        after_wdl = searcher.tablebase.probe_wdl(board)
        after_dtz = searcher.tablebase.probe_dtz(board)
        print(f"Syzygy tras {best_move}  -> WDL_raw: {after_wdl} | DTZ: {after_dtz}")
        
        board.pop()
        # Find Syzygy absolute best DTZ move to compare
        best_possible_dtz: Optional[Tuple[float, float]] = None
        for m in board.legal_moves:
            board.push(m)
            # Evaluar usando tablas y aplicar invesión negamax
            cw = searcher.tablebase.probe_wdl(board)
            cd = searcher.tablebase.probe_dtz(board)
            board.pop()
            
            # Syzygy maneja WDL: -2,-1 (Loss), 0 (Draw), 1,2 (Win)
            # Lo mapeamos groseramente a -1, 0, 1 para simplificar
            mw_norm = 0.0
            if cw < 0: mw_norm = 1.0 # Si hijo pierde, yo gano
            elif cw > 0: mw_norm = -1.0 # Si hijo gana, yo pierdo
            
            md = -cd + float(np.sign(mw_norm)) if cw != 0 else 0.0
            
            if best_possible_dtz is None or searcher.is_better_eval((mw_norm, md), best_possible_dtz):
                best_possible_dtz = (mw_norm, md)
                
        # Evaluar la calidad real de nuestro best_move elegido
        board.push(best_move)
        chosen_cw = searcher.tablebase.probe_wdl(board)
        chosen_cd = searcher.tablebase.probe_dtz(board)
        board.pop()
        
        chosen_mw_norm = 0.0
        if chosen_cw < 0: chosen_mw_norm = 1.0
        elif chosen_cw > 0: chosen_mw_norm = -1.0
        chosen_md = -chosen_cd + float(np.sign(chosen_mw_norm)) if chosen_cw != 0 else 0.0
        
        is_best = False
        if best_possible_dtz is not None:
            assert best_possible_dtz is not None
            is_best = (chosen_md == best_possible_dtz[1] and chosen_mw_norm == best_possible_dtz[0])
            print(f"Move Quality: {'OPTIMAL' if is_best else 'SUBOPTIMAL'} (Elegido: WDL {chosen_mw_norm}, DTZ {chosen_md} | Best: WDL {best_possible_dtz[0]}, DTZ {best_possible_dtz[1]})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fen", type=str, required=True)
    parser.add_argument("--onnx", type=str, default="data/mlp_kbnvk_v5.onnx")
    parser.add_argument("--syzygy", type=str, default="syzygy")
    parser.add_argument("--depth", type=int, default=2)
    args = parser.parse_args()
    
    test_fen_progress(args.fen, args.onnx, args.syzygy, args.depth)
