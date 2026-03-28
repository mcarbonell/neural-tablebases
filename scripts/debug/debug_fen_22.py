import chess
from src.rust_engine import RustGnnEngine
import numpy as np

def debug_pos():
    engine = RustGnnEngine()
    fen = "r1bqk2r/ppp1nppp/4p3/n5N1/2BPp3/P1P5/2P2PPP/R1BQK2R w KQkq - 0 1"
    
    board = chess.Board(fen)
    p_ids, tactical, adj = engine.get_features(fen)
    
    print(f"FEN: {fen}")
    print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
    
    mismatches = []
    for sq in range(64):
        w_atks = len(board.attackers(chess.WHITE, sq))
        b_atks = len(board.attackers(chess.BLACK, sq))
        
        rust_w = int(tactical[sq, 0])
        rust_b = int(tactical[sq, 1])
        
        if rust_w != w_atks or rust_b != b_atks:
            mismatches.append((sq, chess.square_name(sq), rust_w, rust_b, w_atks, b_atks))
            
    if not mismatches:
        print("NO mismatches found for attacker counts!")
    else:
        print(f"Found {len(mismatches)} mismatches:")
        for sq, name, rw, rb, ew, eb in mismatches:
            piece = board.piece_at(sq)
            p_str = piece.symbol() if piece else "."
            print(f"  {name} ({p_str}): Rust(W={rw}, B={rb}), Expected(W={ew}, B={eb})")

    # Also check piece IDs just in case
    for sq in range(64):
        piece = board.piece_at(sq)
        expected_id = 0
        if piece:
            expected_id = piece.piece_type + (0 if piece.color == chess.WHITE else 6)
        if p_ids[sq] != expected_id:
            print(f"  Piece ID mismatch at {chess.square_name(sq)}: Rust={p_ids[sq]}, Expected={expected_id}")

    # Check check flag
    is_checked = board.is_check()
    king_sq = board.king(board.turn)
    rust_flag = int(tactical[king_sq, 2])
    rust_checked = (rust_flag & 1) != 0 # Note: I used bit 0 in x88_board.rs
    print(f"Check flag: Rust={rust_checked}, Expected={is_checked}")

if __name__ == "__main__":
    debug_pos()
