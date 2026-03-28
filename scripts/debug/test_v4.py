import sys
import os
sys.path.append('src')
from generate_datasets import encode_board
import chess
import numpy as np

board = chess.Board() # 32 pieces
# But let's use a 4-piece endgame board
board = chess.Board('8/8/8/8/8/k7/2P5/K7 w - - 0 1') # KPK
print(f"Board pieces: {len(board.piece_map())}")
encoding = encode_board(board, relative='v4')
print(f"Encoding shape (v4): {encoding.shape}")

board = chess.Board('8/8/8/8/8/k7/2P5/K1R5 w - - 0 1') # KRPK
print(f"Board pieces: {len(board.piece_map())}")
encoding = encode_board(board, relative='v4')
print(f"Encoding shape (v4): {encoding.shape}")
