import chess
import chess.syzygy
import numpy as np
import os
import argparse
from typing import List, Tuple

def get_material_config(board: chess.Board) -> str:
    """Returns a string representing the material on the board, e.g., 'KQK'."""
    white_pieces = []
    black_pieces = []
    for pt in chess.PIECE_TYPES:
        for _ in range(len(board.pieces(pt, chess.WHITE))):
            white_pieces.append(chess.piece_symbol(pt).upper())
        for _ in range(len(board.pieces(pt, chess.BLACK))):
            black_pieces.append(chess.piece_symbol(pt).upper())
    
    # Standard format: White pieces then Black pieces, Kings first
    white_pieces.sort(key=lambda x: (x != 'K', x))
    black_pieces.sort(key=lambda x: (x != 'K', x))
    
    return f"{''.join(white_pieces)}v{''.join(black_pieces)}"

def encode_board(board: chess.Board, compact: bool = True, relative: bool = False) -> np.ndarray:
    """
    Encodes the board into a flat array.
    
    Args:
        board: chess.Board object
        compact: If True, uses compact encoding (only pieces present).
                 If False, uses full 768-dim encoding (12 pieces * 64 squares).
        relative: If True, uses relative/geometric encoding (RECOMMENDED).
    
    Returns:
        numpy array with encoding
    """
    if relative:
        # Relative encoding: geometric features + piece info
        return encode_board_relative(board)
    
    if not compact:
        # Full encoding: 768 dimensions (12 pieces * 64 squares)
        encoding = np.zeros(768, dtype=np.float32)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_idx = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
                encoding[piece_idx * 64 + square] = 1.0
        return encoding
    
    # Compact encoding: only encode pieces present
    # For KQvK: 3 pieces * 64 squares = 192 dimensions
    pieces_on_board = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            pieces_on_board.append((piece, square))
    
    # Sort pieces for consistent encoding: White pieces first, then Black
    # Within each color, sort by piece type (K, Q, R, B, N, P)
    pieces_on_board.sort(key=lambda x: (
        0 if x[0].color == chess.WHITE else 1,
        x[0].piece_type
    ))
    
    # Create encoding: num_pieces * 64 dimensions
    num_pieces = len(pieces_on_board)
    encoding = np.zeros(num_pieces * 64, dtype=np.float32)
    
    for idx, (piece, square) in enumerate(pieces_on_board):
        encoding[idx * 64 + square] = 1.0
    
    return encoding

def encode_board_relative(board: chess.Board) -> np.ndarray:
    """
    Relative/geometric encoding that scales to any endgame.
    
    For each piece:
        - Normalized coordinates (x, y): 2 dims
        - Piece type one-hot [K,Q,R,B,N,P]: 6 dims
        - Color [White, Black]: 2 dims
    
    For each pair of pieces:
        - Manhattan distance (normalized): 1 dim
        - Chebyshev distance (normalized): 1 dim
        - Direction vector (dx, dy): 2 dims
    
    Global:
        - Side to move: 1 dim
    
    Total for 3 pieces: 3*10 + 3*4 + 1 = 43 dims
    Total for 4 pieces: 4*10 + 6*4 + 1 = 65 dims
    Total for 5 pieces: 5*10 + 10*4 + 1 = 91 dims
    """
    # Collect pieces
    pieces_on_board = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            pieces_on_board.append((piece, square))
    
    # Sort: White first, then Black; K,Q,R,B,N,P
    pieces_on_board.sort(key=lambda x: (
        0 if x[0].color == chess.WHITE else 1,
        x[0].piece_type
    ))
    
    encoding = []
    
    # 1. Encode each piece
    for piece, square in pieces_on_board:
        # Normalized coordinates [0, 1]
        row = chess.square_rank(square) / 7.0
        col = chess.square_file(square) / 7.0
        encoding.extend([row, col])
        
        # Piece type (one-hot) - Order: [K, Q, R, B, N, P]
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
    
    # 2. Encode pairwise relationships
    num_pieces = len(pieces_on_board)
    for i in range(num_pieces):
        for j in range(i + 1, num_pieces):
            sq1 = pieces_on_board[i][1]
            sq2 = pieces_on_board[j][1]
            
            r1 = chess.square_rank(sq1)
            c1 = chess.square_file(sq1)
            r2 = chess.square_rank(sq2)
            c2 = chess.square_file(sq2)
            
            # Manhattan distance (normalized by max possible: 14)
            manhattan = (abs(r1 - r2) + abs(c1 - c2)) / 14.0
            
            # Chebyshev distance (king moves, normalized by max: 7)
            chebyshev = max(abs(r1 - r2), abs(c1 - c2)) / 7.0
            
            # Direction vector (normalized)
            dx = (c2 - c1) / 7.0
            dy = (r2 - r1) / 7.0
            
            encoding.extend([manhattan, chebyshev, dx, dy])
    
    # 3. Side to move
    encoding.append(1.0 if board.turn == chess.WHITE else 0.0)
    
    return np.array(encoding, dtype=np.float32)

def generate_3piece_dataset(syzygy_path: str, output_dir: str, config: str, compact: bool = True, relative: bool = False):
    """
    Generates a dataset for a specific 3-piece endgame.
    Example configs: 'KQK', 'KRK', 'KPK'
    
    Args:
        syzygy_path: Path to Syzygy tablebases
        output_dir: Output directory for dataset
        config: Endgame configuration (e.g., 'KQvK')
        compact: If True, uses compact encoding (only pieces present)
        relative: If True, uses relative/geometric encoding (RECOMMENDED)
    """
    if not os.path.exists(syzygy_path):
        raise ValueError(f"Syzygy path {syzygy_path} not found.")

    tablebase = chess.syzygy.open_tablebase(syzygy_path)
    
    # Determine pieces based on config
    # Format 'KQvK' means White has K+Q, Black has K
    # Format 'KPvK' means White has K+P, Black has K
    if 'v' in config:
        white_side, black_side = config.split('v')
    else:
        # Simple heuristic for 3-piece: K + OnePiece vs K
        white_side = config[:-1]
        black_side = config[-1]

    print(f"Generating dataset for {white_side} vs {black_side}...")
    print(f"  White pieces: {white_side}")
    print(f"  Black pieces: {black_side}")
    
    positions = []
    labels_wdl = []
    labels_dtz = []

    # Iteration depends on the number of pieces.
    # For 3-piece, we have 64^3 = 262,144 combinations.
    # We'll just iterate through all and check legality.
    
    # Convert piece symbols to piece types
    def symbols_to_pieces(symbols):
        return [chess.Piece(chess.PIECE_SYMBOLS.index(s.lower()), chess.WHITE) for s in symbols]

    w_pieces = symbols_to_pieces(white_side)
    b_pieces = symbols_to_pieces(black_side)
    for p in b_pieces: p.color = chess.BLACK
    
    all_pieces = w_pieces + b_pieces
    num_pieces = len(all_pieces)
    
    import itertools
    
    count = 0
    valid_count = 0
    
    # Brute force all possible square combinations
    for squares in itertools.permutations(chess.SQUARES, num_pieces):
        board = chess.Board(None)
        
        # Place pieces on board
        for i, piece in enumerate(all_pieces):
            board.set_piece_at(squares[i], piece)
        
        # Check if any pawn is on 1st or 8th rank
        invalid_pawn = False
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                rank = chess.square_rank(square)
                if rank == 0 or rank == 7:
                    invalid_pawn = True
                    break
        if invalid_pawn: continue
        
        # We need to test both sides to move
        for turn in [chess.WHITE, chess.BLACK]:
            board.turn = turn
            
            if board.is_valid():
                try:
                    wdl = tablebase.probe_wdl(board)
                    dtz = tablebase.probe_dtz(board)
                    
                    # Store encoded board and labels
                    positions.append(encode_board(board, compact=compact, relative=relative))
                    labels_wdl.append(wdl)
                    labels_dtz.append(dtz)
                    valid_count += 1
                except Exception:
                    pass
        
        count += 1
        if count % 10000 == 0:
            print(f"Checked {count} combinations... Found {valid_count} valid positions.")

    print(f"Found {valid_count} valid positions for {config}.")
    
    output_path = os.path.join(output_dir, f"{config}.npz")
    np.savez_compressed(output_path,
                        x=np.array(positions, dtype=np.float32),
                        wdl=np.array(labels_wdl, dtype=np.int8),
                        dtz=np.array(labels_dtz, dtype=np.int16))
    print(f"Saved to {output_path}")
    tablebase.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--syzygy", type=str, default="syzygy")
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--config", type=str, default="KQvK")
    parser.add_argument("--compact", action="store_true", default=True,
                        help="Use compact encoding (only pieces present)")
    parser.add_argument("--full", action="store_true",
                        help="Use full 768-dim encoding (12 pieces * 64 squares)")
    parser.add_argument("--relative", action="store_true",
                        help="Use relative/geometric encoding (RECOMMENDED for better accuracy)")
    args = parser.parse_args()
    
    # --full flag overrides --compact
    compact = not args.full
    generate_3piece_dataset(args.syzygy, args.data, args.config, compact=compact, relative=args.relative)
