"""
Proof-of-concept: Canonical Forms básicas (rotaciones)
"""
import chess
import numpy as np
import sys
sys.path.append('.')

def rotate_board_90(board):
    """Rota el tablero 90 grados en sentido horario."""
    rotated = chess.Board()
    rotated.clear_board()
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Convertir coordenadas: (r, c) -> (c, 7-r)
            r = chess.square_rank(square)
            c = chess.square_file(square)
            new_r = c
            new_c = 7 - r
            new_square = chess.square(new_c, new_r)
            rotated.set_piece_at(new_square, piece)
    
    # Mantener turno
    rotated.turn = board.turn
    return rotated

def rotate_board_180(board):
    """Rota el tablero 180 grados."""
    rotated = chess.Board()
    rotated.clear_board()
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Convertir coordenadas: (r, c) -> (7-r, 7-c)
            r = chess.square_rank(square)
            c = chess.square_file(square)
            new_r = 7 - r
            new_c = 7 - c
            new_square = chess.square(new_c, new_r)
            rotated.set_piece_at(new_square, piece)
    
    rotated.turn = board.turn
    return rotated

def rotate_board_270(board):
    """Rota el tablero 270 grados (90 anti-horario)."""
    rotated = chess.Board()
    rotated.clear_board()
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Convertir coordenadas: (r, c) -> (7-c, r)
            r = chess.square_rank(square)
            c = chess.square_file(square)
            new_r = 7 - c
            new_c = r
            new_square = chess.square(new_c, new_r)
            rotated.set_piece_at(new_square, piece)
    
    rotated.turn = board.turn
    return rotated

def get_all_rotations(board):
    """Devuelve las 4 rotaciones posibles del tablero."""
    rotations = [
        board,                    # 0°
        rotate_board_90(board),   # 90°
        rotate_board_180(board),  # 180°
        rotate_board_270(board)   # 270°
    ]
    return rotations

def board_to_canonical_encoding(board, encoding_func):
    """
    Encuentra la representación canónica (mínima) entre todas las rotaciones.
    """
    rotations = get_all_rotations(board)
    encodings = [encoding_func(rot) for rot in rotations]
    
    # Encontrar la codificación "más pequeña" (lexicográficamente)
    min_encoding = min(encodings)
    return min_encoding

def test_canonical_forms():
    """Prueba básica de canonical forms."""
    print("Testing canonical forms (rotations)...")
    
    # Crear una posición de prueba
    board = chess.Board()
    board.clear_board()
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.D1, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE
    
    print(f"Original position:")
    print(board)
    
    # Obtener todas las rotaciones
    rotations = get_all_rotations(board)
    
    for i, rot in enumerate(rotations):
        print(f"\nRotation {i*90}°:")
        print(rot)
    
    # Test con encoding simple
    def simple_encoding(board):
        """Encoding simple para testing: lista de (pieza, casilla) ordenada."""
        pieces = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                pieces.append((piece.symbol(), square))
        pieces.sort()
        return str(pieces)
    
    canonical = board_to_canonical_encoding(board, simple_encoding)
    print(f"\nCanonical encoding: {canonical}")
    
    return True

if __name__ == "__main__":
    test_canonical_forms()