import chess
import torch
import numpy as np
from src.rust_engine import RustGnnEngine
import sys

def test_wac():
    engine = RustGnnEngine()
    epd_path = r"C:\Users\mrcm_\Local\proj\ajedrez\simple-chess-ai\tests\WAC.epd"
    
    with open(epd_path, "r") as f:
        lines = f.readlines()[:50] # Test first 50 positions
    
    passed = 0
    failed = 0
    
    for i, line in enumerate(lines):
        # Extract FEN (usually first 6 fields)
        parts = line.split()
        fen = " ".join(parts[:6]) 
        try:
            board = chess.Board(fen)
            # 1. Get Rust features
            p_ids, tactical, adj = engine.get_features(fen)
            
            # 2. Verify Piece IDs
            for sq in range(64):
                piece = board.piece_at(sq)
                expected_id = 0
                if piece:
                    # Map chess.Piece to our ID (1-6 white, 7-12 black)
                    expected_id = piece.piece_type + (0 if piece.color == chess.WHITE else 6)
                
                if p_ids[sq] != expected_id:
                    raise ValueError(f"Piece ID mismatch at {chess.square_name(sq)}: Rust={p_ids[sq]}, Expected={expected_id}")
            
            # 3. Verify Legal Moves (All tactical channels)
            rust_moves = set()
            for src in range(64):
                for dst in range(64):
                    if torch.any(adj[src, dst, :] > 0.5): # All 16 channels
                        rust_moves.add((src, dst))
            
            py_moves = set((m.from_square, m.to_square) for m in board.legal_moves)
            
            # Note: Rust might not handle promotions/castling as separate edges in channel 0 yet, 
            # but let's see if the (from, to) pairs match.
            if rust_moves != py_moves:
                missing = py_moves - rust_moves
                extra = rust_moves - py_moves
                if missing or extra:
                    raise ValueError(f"Legal moves mismatch! Missing: {missing}, Extra: {extra}")

            # 4. Verify Attacker Counts
            for sq in range(64):
                w_atks = len(board.attackers(chess.WHITE, sq))
                b_atks = len(board.attackers(chess.BLACK, sq))
                
                rust_w = int(tactical[sq, 0])
                rust_b = int(tactical[sq, 1])
                
                if rust_w != w_atks or rust_b != b_atks:
                    raise ValueError(f"Attacker count mismatch at {chess.square_name(sq)}: "
                                     f"Rust(W={rust_w}, B={rust_b}), Expected(W={w_atks}, B={b_atks})")
            
            # 5. Verify Checked flag
            is_checked = board.is_check()
            # In our flags, bit 1 (val 2) is Checked for STM
            # Rust provides flags[king_sq] if king is attacked
            king_sq = board.king(board.turn)
            rust_flag = int(tactical[king_sq, 2])
            rust_checked = (rust_flag & 2) != 0
            
            if is_checked != rust_checked:
                raise ValueError(f"Check flag mismatch! Rust={rust_checked}, Expected={is_checked}")

            passed += 1
            print(f"[{i+1}/50] OK: {fen[:30]}...")
            
        except Exception as e:
            failed += 1
            print(f"[{i+1}/50] FAILED: {fen}")
            print(f"  Error: {e}")
            # sys.exit(1) # Stop on first error to debug

    print(f"\nTest Summary: {passed} PASSED, {failed} FAILED")
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    test_wac()
