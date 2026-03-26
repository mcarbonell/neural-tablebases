import chess
import numpy as np

def encode_board_gnn(board: chess.Board) -> np.ndarray:
    """
    Encodes the board as a 64-element array of piece IDs.
    0: Empty
    1-6: White Pawn, Knight, Bishop, Rook, Queen, King
    7-12: Black Pawn, Knight, Bishop, Rook, Queen, King
    """
    # Normalize perspective: side to move is always White
    if board.turn == chess.BLACK:
        # Flip board vertically and swap colors
        flipped = board.transform(chess.flip_vertical)
        normalized_board = chess.Board(None)
        normalized_board.turn = chess.WHITE
        for square, piece in flipped.piece_map().items():
            normalized_board.set_piece_at(square, chess.Piece(piece.piece_type, not piece.color))
        board = normalized_board
    
    encoding = np.zeros(64, dtype=np.int8)
    for square, piece in board.piece_map().items():
        # Piece types: PAWN=1, ..., KING=6
        val = piece.piece_type
        if piece.color == chess.BLACK:
            val += 6
        encoding[square] = val
    
    return encoding
