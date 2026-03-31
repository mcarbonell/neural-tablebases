"""
searcher_v8.py — GNN-Search Integration for V8-Pro Architecture
================================================================
Implements GnnSearcher: a Minimax/Alpha-Beta searcher that uses
ChessGnnV8_Pro as its evaluation function (analogous to NeuralSearcher
in search_poc.py but adapted for the graph-based V8 interface).

Key hypothesis: A 1-2 ply Minimax over the GNN corrects local
inconsistencies and pushes WDL-Acc from ~99% to ~100%.

Usage:
    # Benchmark depth=0 vs depth=1 vs depth=2 on 4-piece endgames
    python src/searcher_v8.py \
        --model data/v8_pro_triple_head_best.pth \
        --syzygy syzygy \
        --configs KRvK,KQvK,KPvK,KRvKP \
        --samples 200 \
        --depths 0,1,2 \
        --output data/logs/gnn_search_benchmark.json
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

sys.path.append(os.path.dirname(__file__))
from model.models_v8 import ChessGnnV8_Pro, build_giant_graph
from search.rust_engine import RustGnnEngine

try:
    import torch_directml
    HAS_DIRECTML = True
except ImportError:
    HAS_DIRECTML = False

WDL_SYZYGY_TO_CLASS = {-2: 0, -1: 0, 0: 1, 1: 2, 2: 2}
WDL_LABELS = {0: "Loss", 1: "Draw", 2: "Win"}
NUM_WDL_CLASSES = 3


def setup_device():
    if HAS_DIRECTML and torch_directml.is_available():
        return torch_directml.device()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GnnSearcher:
    """
    GNN-based Minimax searcher for chess endgames.

    Uses ChessGnnV8_Pro as the leaf evaluation function and runs
    Alpha-Beta pruning to depth D. The key innovation vs NeuralSearcher
    is that the evaluation is topology-aware (graph features), giving
    richer tactical context to each node evaluation.
    """

    def __init__(
        self,
        model_path: str,
        syzygy_path: str,
        node_dim: int = 128,
        num_layers: int = 4,
        device=None
    ):
        self.device = device or setup_device()

        print(f"[GnnSearcher] Device: {self.device}")
        print(f"[GnnSearcher] Loading model: {model_path}")

        self.model = ChessGnnV8_Pro(node_dim=node_dim, num_layers=num_layers)
        state = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        params = sum(p.numel() for p in self.model.parameters())
        print(f"[GnnSearcher] Model loaded. Parameters: {params:,}")

        print("[GnnSearcher] Initialising Rust engine...")
        self.engine = RustGnnEngine()

        print(f"[GnnSearcher] Opening Syzygy at '{syzygy_path}'...")
        self.tablebase = chess.syzygy.open_tablebase(syzygy_path)

        # Cache for GNN evaluations within a single search tree
        self._eval_cache: Dict[str, float] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _invert_score(self, score: float) -> float:
        """Convert score from one player's perspective to the other (Loss<->Win)."""
        return float(NUM_WDL_CLASSES - 1) - score

    @torch.no_grad()
    def _gnn_eval(self, board: chess.Board) -> float:
        """
        Single-position GNN forward pass.
        Returns expected WDL value in [0, 2] from White's perspective
        (0=White loses, 1=Draw, 2=White wins).
        """
        fen = board.fen()

        # Cache lookup
        if fen in self._eval_cache:
            self._cache_hits += 1
            return self._eval_cache[fen]
        self._cache_misses += 1

        try:
            p_ids, tac, edges, cnt = self.engine.get_raw_features(fen)
        except Exception:
            # Fallback to Draw if Rust engine fails
            return 1.0

        p_ids_t = torch.from_numpy(p_ids.astype(np.int64)).unsqueeze(0)
        tac_t = torch.from_numpy(tac.astype(np.float32)).unsqueeze(0) / 8.0

        raw_edges = edges[:cnt]
        e_types = ((raw_edges >> 12) & 0xF).astype(np.int64)
        srcs = ((raw_edges >> 6) & 0x3F).astype(np.int64)
        dsts = (raw_edges & 0x3F).astype(np.int64)

        flat_pids, flat_tac, g_srcs, g_dsts, g_etypes, B = build_giant_graph(
            p_ids_t, tac_t,
            [torch.from_numpy(srcs)],
            [torch.from_numpy(dsts)],
            [torch.from_numpy(e_types)]
        )

        flat_pids = flat_pids.to(self.device)
        flat_tac = flat_tac.to(self.device)
        g_srcs = g_srcs.to(self.device)
        g_dsts = g_dsts.to(self.device)
        g_etypes = g_etypes.to(self.device)

        wdl_logits, _, _ = self.model(flat_pids, flat_tac, g_srcs, g_dsts, g_etypes, B)
        probs = F.softmax(wdl_logits, dim=-1).squeeze(0)

        # Expected WDL value from side-to-move perspective
        score_stm = sum(probs[i].item() * i for i in range(NUM_WDL_CLASSES))

        # Convert to White's perspective for Minimax
        if board.turn == chess.BLACK:
            score_white = self._invert_score(score_stm)
        else:
            score_white = score_stm

        self._eval_cache[fen] = score_white
        return score_white

    def evaluate(self, board: chess.Board) -> float:
        """
        Evaluate a position — handles terminal states and GNN eval.
        Returns score in White's perspective.
        """
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                return float(NUM_WDL_CLASSES - 1)  # White wins = 2
            elif result == "0-1":
                return 0.0  # Black wins (White loses)
            else:
                return 1.0  # Draw

        return self._gnn_eval(board)

    def minimax(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        is_maximizing: bool
    ) -> float:
        """Alpha-Beta Minimax search."""
        if depth == 0 or board.is_game_over():
            return self.evaluate(board)

        if is_maximizing:
            best = -float('inf')
            for move in board.legal_moves:
                board.push(move)
                val = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                best = max(best, val)
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
            return best
        else:
            best = float('inf')
            for move in board.legal_moves:
                board.push(move)
                val = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                best = min(best, val)
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return best

    def get_wdl_prediction(self, board: chess.Board, depth: int) -> int:
        """
        Get WDL class prediction (0=Loss, 1=Draw, 2=Win from side-to-move perspective).
        """
        self._eval_cache.clear()  # Fresh cache per position search

        score_white = self.minimax(
            board, depth,
            -float('inf'), float('inf'),
            board.turn == chess.WHITE
        )
        wdl_white = int(round(score_white))
        wdl_white = max(0, min(NUM_WDL_CLASSES - 1, wdl_white))

        # Convert back to side-to-move perspective
        if board.turn == chess.BLACK:
            return int(round(self._invert_score(wdl_white)))
        return wdl_white

    def close(self):
        self.tablebase.close()


def parse_config(config_name: str) -> Tuple[List[chess.Piece], List[chess.Piece]]:
    parts = config_name.replace("V", "v").split("v")
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
) -> List[Tuple[chess.Board, int]]:
    w_pieces, b_pieces = parse_config(config)
    all_pieces = w_pieces + b_pieces
    results = []
    attempts = 0

    while len(results) < n_samples and attempts < n_samples * 2000:
        attempts += 1
        squares = random.sample(list(chess.SQUARES), len(all_pieces))
        board = chess.Board(None)
        invalid = False
        for i, piece in enumerate(all_pieces):
            if piece.piece_type == chess.PAWN and chess.square_rank(squares[i]) in (0, 7):
                invalid = True
                break
            board.set_piece_at(squares[i], piece)
        if invalid:
            continue
        for turn in [chess.WHITE, chess.BLACK]:
            board.turn = turn
            if board.is_valid():
                try:
                    wdl_raw = tablebase.probe_wdl(board)
                    wdl_class = WDL_SYZYGY_TO_CLASS.get(wdl_raw, 1)
                    results.append((board.copy(), wdl_class))
                    if len(results) >= n_samples:
                        break
                except Exception:
                    continue
    return results


def benchmark(
    model_path: str,
    syzygy_path: str,
    configs: List[str],
    samples_per_config: int,
    depths: List[int],
    output_json: Optional[str] = None,
):
    """
    Compare GNN accuracy at different Minimax search depths.
    This is the core validation experiment for the paper.
    """
    searcher = GnnSearcher(model_path, syzygy_path)

    all_results = {}
    grand_stats = {d: {"correct": 0, "total": 0} for d in depths}

    for config in configs:
        print(f"\n{'='*55}")
        print(f"Config: {config} | Depths: {depths}")
        print(f"Sampling {samples_per_config} positions...")

        positions = sample_positions(config, searcher.tablebase, samples_per_config)
        if not positions:
            print(f"  WARNING: No valid positions for {config}. Skipping.")
            continue

        print(f"  Sampled {len(positions)} positions. Evaluating...")

        depth_stats: Dict[int, Dict[str, int]] = {
            d: {"correct": 0, "total": 0} for d in depths
        }
        t0 = time.time()

        for i, (board, true_wdl) in enumerate(positions):
            if (i + 1) % max(1, len(positions) // 5) == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(positions)}] {elapsed:.1f}s elapsed", flush=True)

            for d in depths:
                pred = searcher.get_wdl_prediction(board, d)
                depth_stats[d]["total"] += 1
                if pred == true_wdl:
                    depth_stats[d]["correct"] += 1

        print(f"\n  Results for {config} ({len(positions)} positions):")
        print(f"  {'Depth':<8} {'Correct':>9} {'Total':>7} {'Acc':>8}")
        print(f"  {'-'*36}")
        for d in depths:
            s = depth_stats[d]
            acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
            print(f"  depth={d}  {s['correct']:>9} {s['total']:>7} {acc:>7.2f}%")
            grand_stats[d]["correct"] += s["correct"]
            grand_stats[d]["total"] += s["total"]

        all_results[config] = {
            str(d): {
                "correct": depth_stats[d]["correct"],
                "total": depth_stats[d]["total"],
                "accuracy": depth_stats[d]["correct"] / max(1, depth_stats[d]["total"]) * 100
            }
            for d in depths
        }

    print(f"\n{'='*55}")
    print(f"GRAND TOTAL ACROSS ALL CONFIGS:")
    print(f"  {'Depth':<8} {'Correct':>9} {'Total':>7} {'Acc':>8}")
    print(f"  {'-'*36}")
    for d in depths:
        s = grand_stats[d]
        acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"  depth={d}  {s['correct']:>9} {s['total']:>7} {acc:>7.2f}%")
    print(f"{'='*55}")

    summary = {
        "model": model_path,
        "configs": configs,
        "samples_per_config": samples_per_config,
        "depths": depths,
        "grand_stats": {
            str(d): {
                "correct": grand_stats[d]["correct"],
                "total": grand_stats[d]["total"],
                "accuracy": grand_stats[d]["correct"] / max(1, grand_stats[d]["total"]) * 100
            }
            for d in depths
        },
        "per_config": all_results
    }

    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {output_json}")

    searcher.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description="GNN-Search Benchmark for V8-Pro")
    parser.add_argument("--model", type=str, default="data/v8_pro_triple_head_best.pth")
    parser.add_argument("--syzygy", type=str, default="syzygy")
    parser.add_argument("--configs", type=str, default="KRvK,KQvK,KPvK")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--depths", type=str, default="0,1,2")
    parser.add_argument("--node_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    configs = [c.strip() for c in args.configs.split(",")]
    depths = [int(d) for d in args.depths.split(",")]

    benchmark(
        model_path=args.model,
        syzygy_path=args.syzygy,
        configs=configs,
        samples_per_config=args.samples,
        depths=depths,
        output_json=args.output,
    )


if __name__ == "__main__":
    main()
