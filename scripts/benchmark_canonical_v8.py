import os
import sys
import torch
import numpy as np
import chess
import chess.syzygy
import argparse
import random
from typing import List, Tuple

# Set up paths
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "src"))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from model.models_v8 import ChessGnnV8_Pro, build_giant_graph
from search.rust_engine import RustGnnEngine
from data.canonical_forms import find_canonical_form
from benchmark_neural_vs_sf import get_config_pieces

def sample_positions(config, tablebase, count=100):
    pieces = get_config_pieces(config)
    num_pieces = len(pieces)
    samples = []
    
    print(f"  Sampling {count} positions for {config}...")
    while len(samples) < count:
        squares = random.sample(range(64), num_pieces)
        board = chess.Board(None)
        board.castling_rights = 0 # FOR SYZYGY COMPATIBILITY
        
        invalid = False
        for i, piece in enumerate(pieces):
            if piece.piece_type == chess.PAWN:
                rank = chess.square_rank(squares[i])
                if rank == 0 or rank == 7:
                    invalid = True
                    break
            board.set_piece_at(squares[i], piece)
        if invalid: continue
        
        for turn in [chess.WHITE, chess.BLACK]:
            board.turn = turn
            if board.is_valid() and board.is_legal(chess.Move.null()):
                try:
                    wdl = tablebase.probe_wdl(board)
                    # Mapping for Side-to-Move ground truth
                    mapped_wdl = 1 if wdl == 0 else (0 if wdl < 0 else 2)
                    samples.append((board.copy(), mapped_wdl))
                    if len(samples) >= count: break
                except Exception as e:
                    pass
    return samples

def run_test(model_path, configs, count=50):
    device = torch.device('cpu') 
    print(f"Device: {device}", flush=True)
    
    model = ChessGnnV8_Pro().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    
    print("Initialising Rust GNN engine...", flush=True)
    engine = RustGnnEngine()
    tablebase = chess.syzygy.open_tablebase('syzygy')
    
    for config in configs:
        print(f"\nEvaluating {config} (Canonicalized)...", flush=True)
        test_positions = sample_positions(config, tablebase, count)
        
        correct = 0
        correct_baseline = 0
        for i, (board, true_wdl) in enumerate(test_positions):
            if i % 10 == 0:
                print(f"    Evaluating {i}/{len(test_positions)}...", flush=True)
            
            # --- 1. Evaluate CANONICAL VERSION ---
            can_board, _ = find_canonical_form(board, None)
            
            p_ids, tac, edges, cnt = engine.get_raw_features(can_board.fen())
            raw_edges = edges[:cnt]
            e_types = (raw_edges >> 12) & 0xF
            srcs = (raw_edges >> 6) & 0x3F
            dsts = raw_edges & 0x3F
            
            p_ids_t = torch.from_numpy(p_ids.astype(np.int64)).unsqueeze(0).to(device)
            tac_t = (torch.from_numpy(tac.astype(np.float32)).unsqueeze(0) / 8.0).to(device)
            list_srcs = [torch.from_numpy(srcs.astype(np.int64))]
            list_dsts = [torch.from_numpy(dsts.astype(np.int64))]
            list_etypes = [torch.from_numpy(e_types.astype(np.int64))]
            
            flat_pids, flat_tac, g_srcs, g_dsts, g_etypes, B = build_giant_graph(
                p_ids_t, tac_t, list_srcs, list_dsts, list_etypes
            )
            
            with torch.no_grad():
                out_wdl, _, _ = model(flat_pids, flat_tac, g_srcs, g_dsts, g_etypes, B)
                pred = torch.argmax(out_wdl, dim=1).item()
                
            # If canonical_board turn is Black, flip prediction (since model is absolute White-relative)
            # This aligns the internal "White-Win/Draw/White-Loss" with the "Side-To-Move" truth.
            if can_board.turn == chess.BLACK:
                pred = 2 - pred
            if pred == true_wdl:
                correct += 1

            # --- 2. Evaluate NON-CANONICAL (Baseline) ---
            p_ids_b, tac_b, edges_b, cnt_b = engine.get_raw_features(board.fen())
            raw_edges_b = edges_b[:cnt_b]
            e_types_b = (raw_edges_b >> 12) & 0xF
            srcs_b = (raw_edges_b >> 6) & 0x3F
            dsts_b = (raw_edges_b & 0x3F).astype(np.int64)
            
            p_ids_t_b = torch.from_numpy(p_ids_b.astype(np.int64)).unsqueeze(0).to(device)
            tac_t_b = (torch.from_numpy(tac_b.astype(np.float32)).unsqueeze(0) / 8.0).to(device)
            
            flat_pids_b, flat_tac_b, g_srcs_b, g_dsts_b, g_etypes_b, B_b = build_giant_graph(
                p_ids_t_b, tac_t_b, [torch.from_numpy(srcs_b.astype(np.int64))], 
                [torch.from_numpy(dsts_b)], [torch.from_numpy(e_types_b.astype(np.int64))]
            )
            
            with torch.no_grad():
                out_wdl_b, _, _ = model(flat_pids_b, flat_tac_b, g_srcs_b, g_dsts_b, g_etypes_b, B_b)
                pred_b = torch.argmax(out_wdl_b, dim=1).item()
            if board.turn == chess.BLACK:
                pred_b = 2 - pred_b
            if pred_b == true_wdl:
                correct_baseline += 1
                
        print(f"\n[RESULT] {config} ({len(test_positions)} samples)", flush=True)
        print(f"  Baseline Accuracy:  {correct_baseline / len(test_positions) * 100:.2f}%", flush=True)
        print(f"  Canonical Accuracy: {correct / len(test_positions) * 100:.2f}%", flush=True)

if __name__ == "__main__":
    # Small sample for immediate proof of concept
    run_test("data/v8_universal_35M_latest.pth", ["KRvK", "KQvK"], 10)
