"""
Fixed canonical forms implementation.
Simpler and more robust version.
"""
import chess
import numpy as np
from typing import List, Tuple

def rotate_board(board: chess.Board, rotation: int) -> chess.Board:
    """
    Rota el tablero: 0=0°, 1=90°, 2=180°, 3=270°
    """
    rotated = chess.Board()
    rotated.clear_board()
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            r = chess.square_rank(square)
            c = chess.square_file(square)
            
            if rotation == 0:  # 0°
                new_r, new_c = r, c
            elif rotation == 1:  # 90°
                new_r, new_c = c, 7 - r
            elif rotation == 2:  # 180°
                new_r, new_c = 7 - r, 7 - c
            elif rotation == 3:  # 270°
                new_r, new_c = 7 - c, r
            else:
                raise ValueError(f"Invalid rotation: {rotation}")
            
            new_square = chess.square(new_c, new_r)
            rotated.set_piece_at(new_square, piece)
    
    # Mantener turno
    rotated.turn = board.turn
    return rotated

def reflect_board_horizontal(board: chess.Board) -> chess.Board:
    """Reflejo horizontal (espejo izquierda-derecha)."""
    reflected = chess.Board()
    reflected.clear_board()
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            r = chess.square_rank(square)
            c = chess.square_file(square)
            new_c = 7 - c
            new_square = chess.square(new_c, r)
            reflected.set_piece_at(new_square, piece)
    
    reflected.turn = board.turn
    return reflected

def get_all_symmetries(board: chess.Board) -> List[chess.Board]:
    """
    Devuelve todas las simetrías del tablero (8 total).
    4 rotaciones × 2 reflexiones.
    """
    symmetries = []
    
    # 4 rotaciones
    for rotation in range(4):
        rotated = rotate_board(board, rotation)
        symmetries.append(rotated)
        
        # Reflexión horizontal de cada rotación
        reflected = reflect_board_horizontal(rotated)
        symmetries.append(reflected)
    
    return symmetries

def board_to_canonical_key(board: chess.Board) -> Tuple:
    """
    Convierte un tablero a una clave canónica única.
    La clave es una tupla que puede ser usada como clave de diccionario.
    
    Formato: (turno, piezas_ordenadas)
    Donde cada pieza es: (color, tipo, fila, columna)
    """
    pieces = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # (color: 0=white, 1=black, tipo, fila, columna)
            key = (
                0 if piece.color == chess.WHITE else 1,
                piece.piece_type,
                chess.square_rank(square),
                chess.square_file(square)
            )
            pieces.append(key)
    
    # Ordenar piezas para consistencia
    # Ordenar por: color, tipo, fila, columna
    pieces.sort()
    
    # Turno: 0=white, 1=black
    turn_key = 0 if board.turn == chess.WHITE else 1
    
    return (turn_key, tuple(pieces))

def find_canonical_form_simple(board: chess.Board) -> Tuple[chess.Board, dict]:
    """
    Encuentra la forma canónica de un tablero usando una clave simple.
    
    Returns:
        - canonical_board: La representación canónica
        - transform_info: Información de la transformación aplicada
    """
    # Obtener todas las simetrías
    symmetries = get_all_symmetries(board)
    
    # Encontrar la simetría con la clave "más pequeña"
    best_key = None
    best_board = None
    best_idx = -1
    
    for idx, sym_board in enumerate(symmetries):
        key = board_to_canonical_key(sym_board)
        
        if best_key is None or key < best_key:
            best_key = key
            best_board = sym_board
            best_idx = idx
    
    # Información de transformación
    transform_info = {
        'rotation': best_idx // 2,  # 0,1,2,3
        'reflected': (best_idx % 2) == 1,  # True si se aplicó reflexión
        'original_to_canonical': best_idx
    }
    
    return best_board, transform_info

def find_canonical_form_with_encoding(board: chess.Board, encoding_func) -> Tuple[chess.Board, dict]:
    """
    Encuentra la forma canónica usando una función de encoding específica.
    Útil cuando queremos usar el mismo encoding que el dataset.
    """
    # Obtener todas las simetrías
    symmetries = get_all_symmetries(board)
    
    # Encontrar la simetría con el encoding "más pequeño"
    best_encoding = None
    best_board = None
    best_idx = -1
    
    for idx, sym_board in enumerate(symmetries):
        encoding = encoding_func(sym_board)
        
        # Convertir encoding a tupla para comparación
        if isinstance(encoding, np.ndarray):
            encoding_tuple = tuple(encoding.tolist())
        else:
            encoding_tuple = tuple(encoding) if hasattr(encoding, '__iter__') else (encoding,)
        
        if best_encoding is None or encoding_tuple < best_encoding:
            best_encoding = encoding_tuple
            best_board = sym_board
            best_idx = idx
    
    # Información de transformación
    transform_info = {
        'rotation': best_idx // 2,  # 0,1,2,3
        'reflected': (best_idx % 2) == 1,  # True si se aplicó reflexión
        'original_to_canonical': best_idx
    }
    
    return best_board, transform_info

def board_to_encoding_key(board: chess.Board, encoding_func) -> Tuple:
    """
    Versión compatible con el código existente.
    Usa la función de encoding para crear una clave.
    """
    encoding = encoding_func(board)
    
    # Convertir a tupla para usar como clave de diccionario
    if isinstance(encoding, np.ndarray):
        return tuple(encoding.tolist())
    elif hasattr(encoding, '__iter__'):
        return tuple(encoding)
    else:
        return (encoding,)

def test_canonical_forms():
    """Prueba completa de canonical forms."""
    print("Testing canonical forms (fixed version)...")
    
    # Crear posición de prueba
    board = chess.Board()
    board.clear_board()
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.D1, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE
    
    print("Original board:")
    print(board)
    
    # Encontrar forma canónica (simple)
    canonical_board, transform_info = find_canonical_form_simple(board)
    
    print("\nCanonical board (simple):")
    print(canonical_board)
    print(f"Transform info: {transform_info}")
    
    # Test con múltiples posiciones simétricas
    print("\n" + "="*50)
    print("Testing symmetry equivalence...")
    
    test_positions = get_all_symmetries(board)[:4]  # Solo primeras 4
    
    canonical_keys = set()
    for i, test_board in enumerate(test_positions):
        canonical_test, _ = find_canonical_form_simple(test_board)
        key = board_to_canonical_key(canonical_test)
        canonical_keys.add(key)
        
        print(f"Position {i} -> Canonical key hash: {hash(key) % 10000:04d}")
    
    print(f"\nAll {len(test_positions)} symmetric positions map to {len(canonical_keys)} canonical form(s).")
    
    if len(canonical_keys) == 1:
        print("✓ SUCCESS: All symmetric positions have the same canonical form!")
    else:
        print("✗ FAIL: Symmetric positions have different canonical forms")
    
    return len(canonical_keys) == 1

if __name__ == "__main__":
    success = test_canonical_forms()
    exit(0 if success else 1)