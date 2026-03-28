"""
Canonical Forms: Encontrar la representación canónica de un tablero
"""
import chess
import numpy as np
from typing import List, Tuple, Literal, Optional

CanonicalMode = Literal["auto", "dihedral", "file_mirror", "none"]


def _board_has_pawns(board: chess.Board) -> bool:
    return any(piece.piece_type == chess.PAWN for piece in board.piece_map().values())

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

def reflect_board_vertical(board: chess.Board) -> chess.Board:
    """Reflejo vertical (espejo arriba-abajo)."""
    reflected = chess.Board()
    reflected.clear_board()
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            r = chess.square_rank(square)
            c = chess.square_file(square)
            new_r = 7 - r
            new_square = chess.square(c, new_r)
            reflected.set_piece_at(new_square, piece)
    
    reflected.turn = board.turn
    return reflected

def get_all_symmetries(board: chess.Board) -> List[chess.Board]:
    """
    Devuelve todas las simetrías del tablero (8 total).
    4 rotaciones × 2 reflexiones.

    ⚠️ Nota (peones):
    Rotaciones/reflexiones verticales no preservan la semántica de los peones.
    Para deduplicación "segura con peones", usa `get_symmetries(board, mode="auto")`.
    """
    return get_symmetries(board, mode="dihedral")


def get_symmetries(board: chess.Board, mode: CanonicalMode = "auto") -> List[chess.Board]:
    """
    Devuelve una lista de tableros transformados por simetrías.

    Modos:
    - "dihedral": 8 simetrías (4 rotaciones × {id, espejo horizontal}). NO seguro con peones.
    - "file_mirror": 2 simetrías ({id, espejo horizontal}). Seguro con peones.
    - "none": solo identidad.
    - "auto": "file_mirror" si hay peones, si no "dihedral".
    """
    if mode == "auto":
        mode = "file_mirror" if _board_has_pawns(board) else "dihedral"

    if mode == "none":
        return [board]

    if mode == "file_mirror":
        return [board, reflect_board_horizontal(board)]

    if mode != "dihedral":
        raise ValueError(f"Invalid canonical mode: {mode!r}")

    symmetries: List[chess.Board] = []

    for rotation in range(4):
        rotated = rotate_board(board, rotation)
        symmetries.append(rotated)

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


def canonical_key(board: chess.Board, mode: CanonicalMode = "auto") -> Tuple:
    """Devuelve la clave canónica mínima dentro del grupo de simetrías permitido."""
    best_key: Optional[Tuple] = None
    for sym_board in get_symmetries(board, mode=mode):
        key = board_to_canonical_key(sym_board)
        if best_key is None or key < best_key:
            best_key = key
    assert best_key is not None
    return best_key


def is_canonical(board: chess.Board, mode: CanonicalMode = "auto") -> bool:
    """True si `board` ya es el representante canónico de su órbita."""
    return board_to_canonical_key(board) == canonical_key(board, mode=mode)

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

def find_canonical_form(board: chess.Board, encoding_func, mode: CanonicalMode = "auto") -> Tuple[chess.Board, dict]:
    """
    Encuentra la forma canónica de un tablero.
    
    Returns:
        - canonical_board: La representación canónica
        - transform_info: Información de la transformación aplicada
    """
    # Obtener simetrías (modo auto = seguro con peones)
    symmetries = get_symmetries(board, mode=mode)
    
    # Encontrar la simetría con la clave "más pequeña"
    best_key = None
    best_board = None
    best_idx = -1
    
    for idx, sym_board in enumerate(symmetries):
        # Usar board_to_canonical_key para comparación consistente
        key = board_to_canonical_key(sym_board)
        
        if best_key is None or key < best_key:
            best_key = key
            best_board = sym_board
            best_idx = idx
    
    # Información de transformación
    if len(symmetries) == 8:
        rotation = best_idx // 2  # 0..3
        reflected = (best_idx % 2) == 1
    elif len(symmetries) == 2:
        rotation = 0
        reflected = best_idx == 1
    else:
        rotation = 0
        reflected = False

    transform_info = {
        'rotation': rotation,
        'reflected': reflected,
        'original_to_canonical': best_idx,
        'mode': mode,
    }
    
    return best_board, transform_info

def test_canonical_forms():
    """Prueba completa de canonical forms."""
    print("Testing canonical forms...")
    
    # Simple encoding for testing
    def simple_encoding(board):
        """Encoding simple para testing."""
        # Create a simple encoding based on piece positions
        encoding = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                encoding.append(piece.piece_type)
                encoding.append(chess.square_rank(square))
                encoding.append(chess.square_file(square))
        return tuple(encoding)
    
    # Crear posición de prueba
    board = chess.Board()
    board.clear_board()
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.D1, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE
    
    print("Original board:")
    print(board)
    
    # Encontrar forma canónica
    canonical_board, transform_info = find_canonical_form(board, simple_encoding)
    
    print("\nCanonical board:")
    print(canonical_board)
    print(f"\nTransform info: {transform_info}")
    
    # Verificar que WDL/DTZ deberían ser iguales
    # (En la práctica, necesitaríamos probar con Syzygy)
    
    # Test con múltiples posiciones simétricas
    print("\n" + "="*50)
    print("Testing symmetry equivalence...")
    
    test_positions = get_all_symmetries(board)[:4]  # Solo primeras 4
    
    canonical_keys = set()
    for i, test_board in enumerate(test_positions):
        canonical_test, _ = find_canonical_form(test_board, simple_encoding)
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
