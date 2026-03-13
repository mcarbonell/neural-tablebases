"""
Encoding invariante a rotaciones para canonical forms
"""
import chess
import numpy as np

def encode_board_invariant(board, use_move_distance=False):
    """
    Encoding geométrico invariante a rotaciones.
    
    Idea: En lugar de coordenadas absolutas, usar:
    1. Distancias entre piezas (ya son invariantes)
    2. Ángulos relativos (invariantes a rotación)
    3. Patrones de conectividad
    
    Para 3 piezas (KQvK):
    - Distancia K_white - Q
    - Distancia K_white - K_black  
    - Distancia Q - K_black
    - Ángulos entre vectores
    - Tipos de pieza
    - Color
    - Side to move
    """
    # Recoger piezas
    pieces = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            pieces.append((piece, square))
    
    # Ordenar: White primero, luego Black; K,Q,R,B,N,P
    pieces.sort(key=lambda x: (
        0 if x[0].color == chess.WHITE else 1,
        x[0].piece_type
    ))
    
    if len(pieces) != 3:
        raise ValueError(f"Expected 3 pieces, got {len(pieces)}")
    
    encoding = []
    
    # 1. Información de piezas (invariante)
    for piece, square in pieces:
        # Tipo de pieza (one-hot) - Order: [K, Q, R, B, N, P]
        piece_type_vec = [0.0] * 6
        type_to_idx = {
            chess.KING: 0,
            chess.QUEEN: 1,
            chess.ROOK: 2,
            chess.BISHOP: 3,
            chess.KNIGHT: 4,
            chess.PAWN: 5
        }
        piece_type_vec[type_to_idx[piece.piece_type]] = 1.0
        encoding.extend(piece_type_vec)
        
        # Color (one-hot)
        color = [1.0, 0.0] if piece.color == chess.WHITE else [0.0, 1.0]
        encoding.extend(color)
    
    # 2. Matriz de distancias (invariante a rotación)
    # Para 3 piezas: 3 distancias
    coords = []
    for _, square in pieces:
        r = chess.square_rank(square)
        c = chess.square_file(square)
        coords.append((r, c))
    
    # Calcular todas las distancias por pares
    distances = []
    for i in range(3):
        for j in range(i+1, 3):
            r1, c1 = coords[i]
            r2, c2 = coords[j]
            
            # Distancia euclidiana (normalizada)
            dx = c2 - c1
            dy = r2 - r1
            dist = np.sqrt(dx*dx + dy*dy) / np.sqrt(98)  # max sqrt(7^2 + 7^2) = sqrt(98)
            
            # Distancia Manhattan (normalizada)
            manhattan = (abs(dx) + abs(dy)) / 14.0
            
            # Distancia Chebyshev (normalizada)
            chebyshev = max(abs(dx), abs(dy)) / 7.0
            
            distances.extend([dist, manhattan, chebyshev])
    
    encoding.extend(distances)
    
    # 3. Ángulos entre vectores (invariante a rotación)
    # Para 3 puntos A, B, C: ángulo ∠ABC
    if len(coords) == 3:
        A = np.array([coords[0][1], coords[0][0]])  # (x, y)
        B = np.array([coords[1][1], coords[1][0]])
        C = np.array([coords[2][1], coords[2][0]])
        
        # Vectores
        BA = A - B
        BC = C - B
        
        # Ángulo entre BA y BC
        dot = np.dot(BA, BC)
        norm_BA = np.linalg.norm(BA)
        norm_BC = np.linalg.norm(BC)
        
        if norm_BA > 0 and norm_BC > 0:
            cos_angle = dot / (norm_BA * norm_BC)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle) / np.pi  # Normalizado a [0, 1]
        else:
            angle = 0.0
        
        encoding.append(angle)
    
    # 4. Side to move (invariante)
    encoding.append(1.0 if board.turn == chess.WHITE else 0.0)
    
    return np.array(encoding, dtype=np.float32)

def test_invariant_encoding():
    """Prueba que el encoding es invariante a rotaciones."""
    print("Testing invariant encoding...")
    
    # Crear posición de prueba
    board = chess.Board()
    board.clear_board()
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.D1, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE
    
    # Funciones de rotación (del archivo anterior)
    def rotate_board_90(board):
        rotated = chess.Board()
        rotated.clear_board()
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                r = chess.square_rank(square)
                c = chess.square_file(square)
                new_r = c
                new_c = 7 - r
                new_square = chess.square(new_c, new_r)
                rotated.set_piece_at(new_square, piece)
        rotated.turn = board.turn
        return rotated
    
    # Obtener encoding original
    encoding_orig = encode_board_invariant(board)
    
    # Rotar y obtener encodings
    board_90 = rotate_board_90(board)
    board_180 = rotate_board_90(board_90)
    board_270 = rotate_board_90(board_180)
    
    encoding_90 = encode_board_invariant(board_90)
    encoding_180 = encode_board_invariant(board_180)
    encoding_270 = encode_board_invariant(board_270)
    
    # Comparar
    print(f"Original encoding shape: {encoding_orig.shape}")
    print(f"Original encoding (first 10): {encoding_orig[:10]}")
    
    # Verificar invariancia
    tolerance = 1e-6
    is_invariant = (
        np.allclose(encoding_orig, encoding_90, atol=tolerance) and
        np.allclose(encoding_orig, encoding_180, atol=tolerance) and
        np.allclose(encoding_orig, encoding_270, atol=tolerance)
    )
    
    print(f"\nEncoding is rotation invariant: {is_invariant}")
    
    if not is_invariant:
        print("\nDifferences:")
        print(f"  Orig vs 90: {np.max(np.abs(encoding_orig - encoding_90))}")
        print(f"  Orig vs 180: {np.max(np.abs(encoding_orig - encoding_180))}")
        print(f"  Orig vs 270: {np.max(np.abs(encoding_orig - encoding_270))}")
    
    return is_invariant

if __name__ == "__main__":
    test_invariant_encoding()