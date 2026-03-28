"""
eval_v8_tablebase.py — Holdout Evaluation of V8-Pro GNN vs Syzygy Tablebases
=============================================================================
Samples random positions from a set of endgame configurations and compares
the V8-Pro GNN WDL prediction against the ground truth from probe_wdl().

Usage:
    python src/eval_v8_tablebase.py --model data/v8_pro_triple_head_best.pth \
        --syzygy syzygy --configs KRvK,KQvK,KPvK,KRvKP --samples 500

Output:
    Per-endgame and per-class accuracy table + overall accuracy.
    Optionally saves a JSON results file.
"""

import chess
import chess.syzygy
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import sys
import json
import random
import time
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# Add src to path
sys.path.append(os.path.dirname(__file__))
from models_v8 import ChessGnnV8_Pro, build_giant_graph
from rust_engine import RustGnnEngine

# Try DirectML
try:
    import torch_directml
    HAS_DIRECTML = True
except ImportError:
    HAS_DIRECTML = False

WDL_SYZYGY_TO_CLASS = {-2: 0, -1: 0, 0: 1, 1: 2, 2: 2}
WDL_LABELS = {0: "Loss", 1: "Draw", 2: "Win"}


def setup_device():
    if HAS_DIRECTML and torch_directml.is_available():
        return torch_directml.device()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: str, device, node_dim: int = 128, num_layers: int = 4) -> ChessGnnV8_Pro:
    model = ChessGnnV8_Pro(node_dim=node_dim, num_layers=num_layers)
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def parse_config(config_name: str) -> Tuple[List[chess.Piece], List[chess.Piece]]:
    """Parse 'KRvKP' type string into white and black piece lists."""
    parts = config_name.replace("V", "v").split("v")
    if len(parts) != 2:
        raise ValueError(f"Invalid config: {config_name}. Expected format like KRvKP.")
    sym_map = {
        'k': chess.KING, 'q': chess.QUEEN, 'r': chess.ROOK,
        'b': chess.BISHOP, 'n': chess.KNIGHT, 'p': chess.PAWN
    }
    w_pieces = [chess.Piece(sym_map[s.lower()], chess.WHITE) for s in parts[0]]
    b_pieces = [chess.Piece(sym_map[s.lower()], chess.BLACK) for s in parts[1]]
    return w_pieces, b_pieces


def sample_positions(
    config: str,
    tablebase: chess.syzygy.Tablebase,
    n_samples: int,
    max_attempts_factor: int = 2000
) -> List[Tuple[chess.Board, int]]:
    """Sample random valid positions for a given endgame config."""
    w_pieces, b_pieces = parse_config(config)
    all_pieces = w_pieces + b_pieces
    n = len(all_pieces)
    results = []
    attempts = 0
    max_attempts = n_samples * max_attempts_factor

    while len(results) < n_samples and attempts < max_attempts:
        attempts += 1
        squares = random.sample(list(chess.SQUARES), n)
        board = chess.Board(None)

        invalid_pawn = False
        for i, piece in enumerate(all_pieces):
            sq = squares[i]
            if piece.piece_type == chess.PAWN and chess.square_rank(sq) in (0, 7):
                invalid_pawn = True
                break
            board.set_piece_at(sq, piece)
        if invalid_pawn:
            continue

        for turn in [chess.WHITE, chess.BLACK]:
            board.turn = turn
            if not board.is_valid():
                continue
            try:
                wdl_raw = tablebase.probe_wdl(board)
                wdl_class = WDL_SYZYGY_TO_CLASS.get(wdl_raw, 1)
                results.append((board.copy(), wdl_class))
                if len(results) >= n_samples:
                    break
            except Exception:
                continue

    return results


@torch.no_grad()
def evaluate_position(
    board: chess.Board,
    model: ChessGnnV8_Pro,
    engine: RustGnnEngine,
    device,
) -> int:
    """Run V8-Pro forward pass and return predicted WDL class (0=Loss,1=Draw,2=Win)."""
    try:
        p_ids, tac, edges, cnt = engine.get_raw_features(board.fen())
    except Exception:
        return 1  # Fallback to Draw on engine error

    p_ids_t = torch.from_numpy(p_ids.astype(np.int64)).unsqueeze(0)   # (1, 64)
    tac_t = torch.from_numpy(tac.astype(np.float32)).unsqueeze(0) / 8.0  # (1, 64, 4)

    # Decode edges
    raw_edges = edges[:cnt]
    e_types = ((raw_edges >> 12) & 0xF).astype(np.int64)
    srcs = ((raw_edges >> 6) & 0x3F).astype(np.int64)
    dsts = (raw_edges & 0x3F).astype(np.int64)

    list_srcs = [torch.from_numpy(srcs)]
    list_dsts = [torch.from_numpy(dsts)]
    list_etypes = [torch.from_numpy(e_types)]

    flat_pids, flat_tac, g_srcs, g_dsts, g_etypes, B = build_giant_graph(
        p_ids_t, tac_t, list_srcs, list_dsts, list_etypes
    )

    flat_pids = flat_pids.to(device)
    flat_tac = flat_tac.to(device)
    g_srcs = g_srcs.to(device)
    g_dsts = g_dsts.to(device)
    g_etypes = g_etypes.to(device)

    wdl_logits, _, _ = model(flat_pids, flat_tac, g_srcs, g_dsts, g_etypes, B)
    return int(wdl_logits.argmax(dim=-1).item())


def run_evaluation(
    model_path: str,
    syzygy_path: str,
    configs: List[str],
    samples_per_config: int,
    node_dim: int = 128,
    num_layers: int = 4,
    output_json: Optional[str] = None,
):
    device = setup_device()
    print(f"Device: {device}")

    print(f"Loading model from {model_path}...")
    model = load_model(model_path, device, node_dim=node_dim, num_layers=num_layers)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded. Parameters: {params:,}")

    print("Initialising Rust GNN engine...")
    engine = RustGnnEngine()

    print(f"Opening Syzygy tablebases at '{syzygy_path}'...")
    tablebase = chess.syzygy.open_tablebase(syzygy_path)

    all_results: Dict[str, Dict] = {}
    grand_correct = 0
    grand_total = 0

    for config in configs:
        print(f"\n{'='*50}")
        print(f"Config: {config}")
        print(f"Sampling {samples_per_config} positions...")

        positions = sample_positions(config, tablebase, samples_per_config)
        if not positions:
            print(f"  WARNING: No valid positions found for {config}. Skipping.")
            continue

        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        t0 = time.time()

        for i, (board, true_wdl) in enumerate(positions):
            pred_wdl = evaluate_position(board, model, engine, device)
            class_total[true_wdl] += 1
            if pred_wdl == true_wdl:
                class_correct[true_wdl] += 1

            if (i + 1) % max(1, len(positions) // 5) == 0:
                elapsed = time.time() - t0
                speed = (i + 1) / elapsed
                print(f"  [{i+1}/{len(positions)}] Speed: {speed:.0f} pos/s", flush=True)

        total = sum(class_total.values())
        correct = sum(class_correct.values())
        acc = correct / total * 100 if total > 0 else 0.0

        print(f"\n  Results for {config} ({total} positions):")
        print(f"  {'Class':<8} {'Correct':>8} {'Total':>8} {'Acc':>8}")
        print(f"  {'-'*38}")
        for cls in sorted(class_total.keys()):
            cls_acc = class_correct[cls] / class_total[cls] * 100 if class_total[cls] > 0 else 0
            print(f"  {WDL_LABELS[cls]:<8} {class_correct[cls]:>8} {class_total[cls]:>8} {cls_acc:>7.2f}%")
        print(f"  {'-'*38}")
        print(f"  {'TOTAL':<8} {correct:>8} {total:>8} {acc:>7.2f}%")

        grand_correct += correct
        grand_total += total

        all_results[config] = {
            "total": total,
            "correct": correct,
            "accuracy": acc,
            "per_class": {
                WDL_LABELS[c]: {
                    "correct": class_correct[c],
                    "total": class_total[c],
                    "accuracy": class_correct[c] / class_total[c] * 100 if class_total[c] > 0 else 0.0
                }
                for c in sorted(class_total.keys())
            }
        }

    tablebase.close()

    grand_acc = grand_correct / grand_total * 100 if grand_total > 0 else 0.0
    print(f"\n{'='*50}")
    print(f"GRAND TOTAL: {grand_correct}/{grand_total} = {grand_acc:.2f}% WDL Accuracy")
    print(f"{'='*50}")

    summary = {
        "model": model_path,
        "syzygy": syzygy_path,
        "configs": configs,
        "samples_per_config": samples_per_config,
        "grand_accuracy": grand_acc,
        "grand_correct": grand_correct,
        "grand_total": grand_total,
        "per_config": all_results
    }

    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {output_json}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Holdout Evaluation of V8-Pro GNN vs Syzygy")
    parser.add_argument("--model", type=str, default="data/v8_pro_triple_head_best.pth",
                        help="Path to model .pth file")
    parser.add_argument("--syzygy", type=str, default="syzygy",
                        help="Path to Syzygy tablebase directory")
    parser.add_argument("--configs", type=str, default="KRvK,KQvK,KPvK",
                        help="Comma-separated list of endgame configs (e.g. KRvK,KQvKR)")
    parser.add_argument("--samples", type=int, default=500,
                        help="Number of positions to sample per config")
    parser.add_argument("--node_dim", type=int, default=128,
                        help="GNN node embedding dimension (must match model)")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of GNN layers (must match model)")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save JSON results (e.g. data/logs/eval_v8.json)")
    args = parser.parse_args()

    configs = [c.strip() for c in args.configs.split(",")]
    run_evaluation(
        model_path=args.model,
        syzygy_path=args.syzygy,
        configs=configs,
        samples_per_config=args.samples,
        node_dim=args.node_dim,
        num_layers=args.num_layers,
        output_json=args.output,
    )


if __name__ == "__main__":
    main()
