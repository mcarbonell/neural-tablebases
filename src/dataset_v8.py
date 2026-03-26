import torch
from torch.utils.data import Dataset
import numpy as np
import chess
import sys
import os

# Import local tools
sys.path.append(os.path.dirname(__file__))
from gnn_utils import board_array_to_chess_board

class ChessGNNDataset(Dataset):
    """
    V8 GNN Dataset: Loads 64-element piece arrays and builds adjacency matrices.
    """
    def __init__(self, data_path: str, precalculate_adj: bool = False):
        data = np.load(data_path)
        self.x = data['x'] # (N, 64) int8
        self.wdl = data['wdl'] # (N,) int8
        # Standardize WDL labels: -2, 0, 2 -> 0, 1, 2
        self.wdl = (self.wdl // 2) + 1
        self.dtz = data['dtz'] # (N,) int16
        
        self.precalculate_adj = precalculate_adj
        self.adj_cache = {} if precalculate_adj else None

    def __len__(self):
        return len(self.x)

    def _get_adj(self, idx: int) -> torch.Tensor:
        """
        Builds the 64x64 adjacency matrix from the board at idx.
        A[i, j] = 1 if piece at square i attacks square j.
        """
        arr = self.x[idx]
        board = board_array_to_chess_board(arr)
        
        adj = np.zeros((64, 64), dtype=np.float32)
        for square in range(64):
            piece = board.piece_at(square)
            if piece:
                attacks = board.attacks(square)
                for target_square in attacks:
                    adj[square, target_square] = 1.0
        
        return torch.from_numpy(adj)

    def __getitem__(self, idx):
        # 1. Node Features (Piece IDs 0-12)
        x_piece_ids = torch.from_numpy(self.x[idx]).long()
        
        # 2. Labels
        y_wdl = torch.tensor(self.wdl[idx], dtype=torch.long)
        y_dtz = torch.tensor(self.dtz[idx], dtype=torch.float32).unsqueeze(0)
        
        # 3. Adjacency Matrix
        if self.precalculate_adj and idx in self.adj_cache:
            adj = self.adj_cache[idx]
        else:
            adj = self._get_adj(idx)
            if self.precalculate_adj:
                self.adj_cache[idx] = adj
                
        return x_piece_ids, adj, y_wdl, y_dtz
