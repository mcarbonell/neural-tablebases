import torch
import numpy as np
import chess
from typing import List, Tuple

def board_array_to_chess_board(arr: np.ndarray) -> chess.Board:
    """
    Converts a 64-element piece array (0-12) to a chess.Board object.
    """
    board = chess.Board(None)
    board.turn = chess.WHITE # Standard perspective from encode_board_gnn
    
    for i in range(64):
        val = arr[i]
        if val > 0:
            color = chess.WHITE if val <= 6 else chess.BLACK
            piece_type = val if val <= 6 else val - 6
            board.set_piece_at(i, chess.Piece(piece_type, color))
    
    return board

def build_graph_from_array(arr: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts a single board array into node features and edge indices.
    
    Node Features (X): [64, F]
    Features:
    - Piece ID (0-12, one-hot encoded)
    - Square coordinates (rank, file normalized)
    
    Edge Indices (A): [2, E]
    Edges: (Source -> Target) where Source piece attacks/sees Target.
    """
    board = board_array_to_chess_board(arr)
    
    # Node Features: [64, 13 + 2] = 15 features
    X = np.zeros((64, 15), dtype=np.float32)
    for i in range(64):
        val = arr[i]
        X[i, val] = 1.0 # One-hot piece
        X[i, 13] = chess.square_rank(i) / 7.0
        X[i, 14] = chess.square_file(i) / 7.0
        
    # Edge Indices
    sources = []
    targets = []
    
    for square in range(64):
        piece = board.piece_at(square)
        if piece:
            # Get squares attacked by this piece
            attacks = board.attacks(square)
            for target_square in attacks:
                sources.append(square)
                targets.append(target_square)
    
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    return torch.from_numpy(X), edge_index
