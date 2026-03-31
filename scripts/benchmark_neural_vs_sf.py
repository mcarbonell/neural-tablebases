import sys
import os
import torch
import numpy as np
import chess
import chess.syzygy
import time
import argparse
import random
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor

# Add src to sys.path for internal imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from model.models_v8 import build_giant_graph
from search.rust_engine import RustGnnEngine
from data.canonical_forms import is_canonical

# Import StockfishAnalyzer from benchmark_stockfish.py
sys.path.append(os.path.join(os.path.dirname(__file__)))
from benchmark_stockfish import StockfishAnalyzer

# Check for DirectML
try:
    import torch_directml
    HAS_DIRECTML = True
except ImportError:
    HAS_DIRECTML = False

class NeuralEvaluator:
    def __init__(self, model_path, device):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            import torch_directml
            self.device = torch_directml.device()
        except:
            pass
            
        print(f"Using {self.device} for Neural GNN.")
        
        self.engine = RustGnnEngine()
        
        # We unified Vanguard into ChessGnnV8_Pro
        from model.models_v8 import ChessGnnV8_Pro
        self.model = ChessGnnV8_Pro().to(self.device)
        
        print(f"Loading Neural GNN weights from {model_path}...")
            
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        self.model.eval()

    @torch.no_grad()
    def evaluate_batch(self, boards: List[chess.Board]) -> List[int]:
        """Evaluates a batch of boards and returns WDL classes (0=Loss, 1=Draw, 2=Win)."""
        batch_pids = []
        batch_tac = []
        list_srcs = []
        list_dsts = []
        list_etypes = []
        
        for board in boards:
            # Vanguard V8 normalization: Mirror if Black to move
            # This reflects squares, ranks, and swaps piece colors.
            # The model always sees the position as 'Us' (White IDs) vs 'Them' (Black IDs).
            if board.turn == chess.BLACK:
                process_board = board.mirror()
            else:
                process_board = board
            
            p_ids, tac, edges, edge_count = self.engine.get_raw_features(process_board.fen())
            
            raw_edges = edges[:edge_count]
            e_types = (raw_edges >> 12) & 0xF
            srcs = (raw_edges >> 6) & 0x3F
            dsts = raw_edges & 0x3F
            
            batch_pids.append(torch.from_numpy(p_ids.astype(np.int64)))
            batch_tac.append(torch.from_numpy(tac.astype(np.float32)) / 8.0)
            list_srcs.append(torch.from_numpy(srcs.astype(np.int64)))
            list_dsts.append(torch.from_numpy(dsts.astype(np.int64)))
            list_etypes.append(torch.from_numpy(e_types.astype(np.int64)))

        p_ids_t = torch.stack(batch_pids).to(self.device)
        tac_t = torch.stack(batch_tac).to(self.device)
        
        flat_pids, flat_tac, all_srcs, all_dsts, all_etypes, B = build_giant_graph(
            p_ids_t, tac_t, list_srcs, list_dsts, list_etypes
        )
        all_srcs = all_srcs.to(self.device)
        all_dsts = all_dsts.to(self.device)
        all_etypes = all_etypes.to(self.device)
        out_wdl, _, _ = self.model(flat_pids, flat_tac, all_srcs, all_dsts, all_etypes, B)
        
        preds = torch.argmax(out_wdl, dim=1).cpu().numpy()
        
        # No flip needed: the board was mirrored, so prediction is now relative to STM.
        return preds.tolist()

def get_all_endgames(syzygy_path: str) -> List[str]:
    """Finds all available 3 and 4 piece endgames in the syzygy directory."""
    configs = []
    if not os.path.exists(syzygy_path):
        return []
    for filename in os.listdir(syzygy_path):
        if filename.endswith(".rtbw"):
            name = filename.split('.')[0]
            # Syzygy names: 3-piece (KPK), 4-piece (KNPK, etc.)
            if len(name) == 3 or len(name) == 4:
                if name not in configs:
                    configs.append(name)
    return sorted(list(set(configs)))

def get_config_pieces(config: str) -> List[chess.Piece]:
    """Parses config like KRvKP into a list of chess.Piece objects."""
    white_side, black_side = config.replace('V', 'v').split('v')
    w_pieces = [chess.Piece(chess.PIECE_SYMBOLS.index(s.lower()), chess.WHITE) for s in white_side]
    b_pieces = [chess.Piece(chess.PIECE_SYMBOLS.index(s.lower()), chess.BLACK) for s in black_side]
    return w_pieces + b_pieces

def sample_random_positions(config: str, tablebase, count=1000) -> List[Tuple[chess.Board, int]]:
    """Samples random valid canonical positions for a given endgame."""
    pieces = get_config_pieces(config)
    num_pieces = len(pieces)
    
    # We use a simple rejection sampling for speed, as 1000 is small
    samples = []
    seen_fens = set()
    
    attempts = 0
    while len(samples) < count and attempts < count * 50:
        attempts += 1
        squares = random.sample(range(64), num_pieces)
        
        board = chess.Board(None)
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
            if board.is_valid():
                # Enforce legality (no king captures, etc.)
                if board.is_legal(chess.Move.null()): # Basic legal check
                    pass
                
                if is_canonical(board):
                    fen = board.shredder_fen()
                    if fen not in seen_fens:
                        try:
                            wdl = tablebase.probe_wdl(board)
                            # Syzygy WDL: -2=Loss, -1=CursedLoss, 0=Draw, 1=BlessedWin, 2=Win
                            # We map to 3 classes: 0 (Loss), 1 (Draw), 2 (Win)
                            mapped_wdl = 1 if wdl == 0 else (0 if wdl < 0 else 2)
                            samples.append((board.copy(), mapped_wdl))
                            seen_fens.add(fen)
                        except:
                            continue
        if len(samples) >= count:
            break
            
    return samples

def run_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--syzygy", type=str, default="syzygy")
    parser.add_argument("--model", type=str, default="data/v8_universal_35M_latest.pth")
    parser.add_argument("--sf", type=str, default=r"C:\Users\mrcm_\Local\proj\ajedrez\simple-chess-ai\bin\stockfish\stockfish-windows-x86-64-avx2.exe")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--configs", type=str, default=None, help="Comma-separated list of endgames (e.g. KRvK,KPvKP)")
    parser.add_argument("--limit_configs", type=int, default=None, help="Limit number of endgame configurations to test")
    args = parser.parse_args()

    # 1. Init Hardware
    if HAS_DIRECTML and torch_directml.is_available():
        device = torch_directml.device()
        print(f"Using DirectML (Radeon 780M) for Neural GNN.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {device} for Neural GNN.")

    # 2. Init Engines
    neural = NeuralEvaluator(args.model, device)
    stockfish = StockfishAnalyzer(args.sf)
    tablebase = chess.syzygy.open_tablebase(args.syzygy)

    # 3. Identify configs
    if args.configs:
        final_configs = [c.strip() for c in args.configs.split(",")]
    else:
        final_configs = get_all_endgames(args.syzygy)
        if args.limit_configs:
            final_configs = final_configs[:args.limit_configs]

    print(f"Found {len(final_configs)} endgame configurations to benchmark.")

    results = []

    for config in final_configs:
        print(f"\nEvaluating {config}...")
        test_positions = sample_random_positions(config, tablebase, count=args.count)
        if not test_positions:
            print(f"  Skipping {config} (no positions found).")
            continue
        
        boards = [p[0] for p in test_positions]
        ground_truth = [p[1] for p in test_positions]
        
        # Neural Eval
        neural_preds = []
        for i in range(0, len(boards), args.batch_size):
            batch = boards[i : i + args.batch_size]
            neural_preds.extend(neural.evaluate_batch(batch))
        
        # Stockfish Eval
        sf_preds = []
        for board in boards:
            # We use a threshold of 50 centipawns (0.50 pawns) for W/L/D
            score = stockfish.get_eval(board.fen())
            if score is None:
                sf_preds.append(1) # Assume Draw on error
                continue
            
            # eval output is in pawns, map to centipawns
            cp = score * 100
            
            # Stockfish 'eval' is white-relative in SF16+, need to flip if black
            if board.turn == chess.BLACK:
                cp = -cp
                
            if cp > 50:
                mapped = 2 # Win
            elif cp < -50:
                mapped = 0 # Loss
            else:
                mapped = 1 # Draw
            sf_preds.append(mapped)
            
        # Analysis
        n_correct = sum(1 for p, g in zip(neural_preds, ground_truth) if p == g)
        sf_correct = sum(1 for p, g in zip(sf_preds, ground_truth) if p == g)
        
        # Mistakes
        n_fatal = sum(1 for p, g in zip(neural_preds, ground_truth) if (p == 0 and g == 2) or (p == 2 and g == 0))
        sf_fatal = sum(1 for p, g in zip(sf_preds, ground_truth) if (p == 0 and g == 2) or (p == 2 and g == 0))
        
        res_row = {
            "Config": config,
            "Neural_Acc": n_correct / len(boards) * 100,
            "SF_Acc": sf_correct / len(boards) * 100,
            "Neural_Fatal": n_fatal,
            "SF_Fatal": sf_fatal,
            "Total": len(boards)
        }
        results.append(res_row)
        print(f"  Neural Accuracy: {res_row['Neural_Acc']:.2f}% | Fatal: {n_fatal}")
        print(f"  SF Accuracy:     {res_row['SF_Acc']:.2f}% | Fatal: {sf_fatal}")

    # Generate Report
    report_path = "benchmark_results.md"
    with open(report_path, "w") as f:
        f.write("# Benchmark Results: Neural GNN vs Stockfish NNUE (D0)\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Model**: {args.model}\n")
        f.write(f"**Positions per config**: {args.count}\n\n")
        
        f.write("| Config | Neural Acc | SF Acc | Neural Fatal | SF Fatal |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        
        total_n_acc = 0
        total_sf_acc = 0
        total_n_fatal = 0
        total_sf_fatal = 0
        
        for r in results:
            f.write(f"| {r['Config']} | {r['Neural_Acc']:.2f}% | {r['SF_Acc']:.2f}% | {r['Neural_Fatal']} | {r['SF_Fatal']} |\n")
            total_n_acc += r['Neural_Acc']
            total_sf_acc += r['SF_Acc']
            total_n_fatal += r['Neural_Fatal']
            total_sf_fatal += r['SF_Fatal']
            
        avg_n = total_n_acc / len(results)
        avg_sf = total_sf_acc / len(results)
        
        f.write(f"| **AVERAGE** | **{avg_n:.2f}%** | **{avg_sf:.2f}%** | **{total_n_fatal}** | **{total_sf_fatal}** |\n\n")
        
        f.write("## Summary\n")
        if avg_n > avg_sf:
            f.write(f"The Neural GNN outperformed Stockfish NNUE (D0) by **{avg_n - avg_sf:.2f}%** on average.\n")
        else:
            f.write(f"Stockfish NNUE (D0) outperformed the Neural GNN by **{avg_sf - avg_n:.2f}%** on average.\n")

    print(f"\nBenchmark complete! Results saved to {report_path}")
    if 'stockfish' in locals():
        stockfish.close()
    tablebase.close()

if __name__ == "__main__":
    run_benchmark()
