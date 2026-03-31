"""
export_onnx_v8.py — ONNX Export + Latency Benchmark for V8-Pro GNN
====================================================================
Exports ChessGnnV8_Pro to ONNX format and benchmarks inference latency,
including the full pipeline: Rust feature extraction + model forward pass.

Note on GNN ONNX export:
  torch.onnx.export works for the GNN if the graph structure (edge indices)
  is treated as dynamic inputs. We export with fixed batch_size=1 and dynamic
  edge counts, suitable for online inference in the GNN-Search pipeline.

Usage:
    # Export model
    python src/export_onnx_v8.py \
        --model data/v8_pro_triple_head_best.pth \
        --output data/v8_pro_triple_head.onnx

    # Export + latency benchmark
    python src/export_onnx_v8.py \
        --model data/v8_pro_triple_head_best.pth \
        --output data/v8_pro_triple_head.onnx \
        --benchmark --n_warmup 20 --n_bench 200 \
        --fen "8/8/8/4k3/8/4K3/4Q3/8 w - - 0 1"
"""

import torch
import numpy as np
import argparse
import os
import sys
import time
import json

sys.path.append(os.path.dirname(__file__))
from model.models_v8 import ChessGnnV8_Pro, build_giant_graph
from search.rust_engine import RustGnnEngine


# ──────────────────────────────────────────────────────────────────────────
# Export
# ──────────────────────────────────────────────────────────────────────────

def _make_dummy_inputs(num_edges: int = 32):
    """Create dummy graph inputs for a single board position."""
    B = 1
    p_ids = torch.zeros(B, 64, dtype=torch.long)
    tac   = torch.zeros(B, 64, 4, dtype=torch.float32)

    srcs   = torch.randint(0, 64, (num_edges,), dtype=torch.long)
    dsts   = torch.randint(0, 64, (num_edges,), dtype=torch.long)
    etypes = torch.randint(0, 16, (num_edges,), dtype=torch.long)

    flat_pids, flat_tac, g_srcs, g_dsts, g_etypes, B_out = build_giant_graph(
        p_ids, tac, [srcs], [dsts], [etypes]
    )
    return flat_pids, flat_tac, g_srcs, g_dsts, g_etypes, B_out


def export_v8_onnx(
    model_path: str,
    output_path: str,
    node_dim: int = 128,
    num_layers: int = 4,
    opset: int = 17,
):
    """Export ChessGnnV8_Pro to ONNX."""
    print(f"Loading model from {model_path}...")
    model = ChessGnnV8_Pro(node_dim=node_dim, num_layers=num_layers)
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Dummy inputs
    flat_pids, flat_tac, g_srcs, g_dsts, g_etypes, B = _make_dummy_inputs(num_edges=64)

    print(f"Exporting to {output_path} (opset={opset})...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Wrap model to accept batch_size as tensor (ONNX doesn't support Python int inputs)
    class OnnxWrapper(torch.nn.Module):
        def __init__(self, gnn):
            super().__init__()
            self.gnn = gnn
        def forward(self, p_ids, tac, srcs, dsts, etypes):
            # batch_size inferred from p_ids: total_nodes / 64
            B = p_ids.shape[0] // 64
            return self.gnn(p_ids, tac, srcs, dsts, etypes, B)

    wrapper = OnnxWrapper(model)
    wrapper.eval()

    dummy = (flat_pids, flat_tac, g_srcs, g_dsts, g_etypes)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            output_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["p_ids", "node_tac", "srcs", "dsts", "etypes"],
            output_names=["wdl_logits", "dtz", "eval_score"],
            dynamic_axes={
                "p_ids":       {0: "total_nodes"},
                "node_tac":    {0: "total_nodes"},
                "srcs":        {0: "total_edges"},
                "dsts":        {0: "total_edges"},
                "etypes":      {0: "total_edges"},
                "wdl_logits":  {0: "batch_size"},
                "dtz":         {0: "batch_size"},
                "eval_score":  {0: "batch_size"},
            },
            verbose=False,
        )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Export successful! File size: {size_mb:.2f} MB")
    return output_path


# ──────────────────────────────────────────────────────────────────────────
# Latency Benchmark
# ──────────────────────────────────────────────────────────────────────────

def benchmark_pytorch(model_path: str, fens: list, n_warmup: int = 10, n_bench: int = 100,
                      node_dim: int = 128, num_layers: int = 4):
    """Measure PyTorch CPU inference latency: Rust extraction + GNN forward."""
    model = ChessGnnV8_Pro(node_dim=node_dim, num_layers=num_layers)
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()

    engine = RustGnnEngine()

    def _eval_one(fen: str):
        p_ids, tac, edges, cnt = engine.get_raw_features(fen)
        raw = edges[:cnt]
        e_types = ((raw >> 12) & 0xF).astype(np.int64)
        srcs    = ((raw >>  6) & 0x3F).astype(np.int64)
        dsts    = (raw & 0x3F).astype(np.int64)
        p_ids_t = torch.from_numpy(p_ids.astype(np.int64)).unsqueeze(0)
        tac_t   = torch.from_numpy(tac.astype(np.float32)).unsqueeze(0) / 8.0
        fpi, ft, gs, gd, ge, B = build_giant_graph(
            p_ids_t, tac_t,
            [torch.from_numpy(srcs)], [torch.from_numpy(dsts)], [torch.from_numpy(e_types)]
        )
        with torch.no_grad():
            model(fpi, ft, gs, gd, ge, B)

    print(f"\nPyTorch CPU Benchmark: {n_warmup} warmup + {n_bench} measured calls")
    print(f"FENs used: {len(fens)}")

    # Warmup
    for i in range(n_warmup):
        _eval_one(fens[i % len(fens)])

    # Measure
    times = []
    for i in range(n_bench):
        t0 = time.perf_counter()
        _eval_one(fens[i % len(fens)])
        times.append((time.perf_counter() - t0) * 1000)  # ms

    times_arr = np.array(times)
    print(f"  Mean:   {times_arr.mean():.3f} ms")
    print(f"  Median: {np.median(times_arr):.3f} ms")
    print(f"  P95:    {np.percentile(times_arr, 95):.3f} ms")
    print(f"  Min:    {times_arr.min():.3f} ms")
    print(f"  Max:    {times_arr.max():.3f} ms")
    print(f"  Target: < 10ms  {'✅ PASS' if times_arr.mean() < 10 else '❌ FAIL (needs optimization)'}")
    return {
        "mean_ms": float(times_arr.mean()),
        "median_ms": float(np.median(times_arr)),
        "p95_ms": float(np.percentile(times_arr, 95)),
        "min_ms": float(times_arr.min()),
        "max_ms": float(times_arr.max()),
        "n_bench": n_bench,
    }


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

DEFAULT_FENS = [
    "8/8/8/4k3/8/4K3/4Q3/8 w - - 0 1",   # KQvK
    "8/8/8/8/8/4K3/4R3/4k3 w - - 0 1",   # KRvK
    "8/8/4k3/8/4K3/8/4P3/8 w - - 0 1",   # KPvK
    "8/8/8/3k4/8/3K4/3Q3/8 b - - 0 1",   # KQvK Black-to-move
    "8/8/8/8/8/3K4/3R4/3k4 b - - 0 1",   # KRvK Black-to-move
]


def main():
    parser = argparse.ArgumentParser(description="ONNX Export + Latency Benchmark for V8-Pro GNN")
    parser.add_argument("--model", type=str, default="data/v8_pro_triple_head_best.pth")
    parser.add_argument("--output", type=str, default="data/v8_pro_triple_head.onnx")
    parser.add_argument("--node_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--benchmark", action="store_true", default=False,
                        help="Run PyTorch CPU latency benchmark after export")
    parser.add_argument("--n_warmup", type=int, default=20)
    parser.add_argument("--n_bench", type=int, default=200)
    parser.add_argument("--fen", type=str, default=None,
                        help="Custom FEN to include in benchmark (optional)")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Save benchmark JSON to this path")
    args = parser.parse_args()

    # Export
    export_v8_onnx(
        args.model, args.output,
        node_dim=args.node_dim, num_layers=args.num_layers, opset=args.opset
    )

    # Benchmark
    if args.benchmark:
        fens = DEFAULT_FENS.copy()
        if args.fen:
            fens.insert(0, args.fen)
        results = benchmark_pytorch(
            args.model, fens,
            n_warmup=args.n_warmup,
            n_bench=args.n_bench,
            node_dim=args.node_dim,
            num_layers=args.num_layers,
        )
        results["model"] = args.model
        results["onnx_output"] = args.output
        results["onnx_size_mb"] = os.path.getsize(args.output) / (1024 * 1024)

        if args.output_json:
            os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
            with open(args.output_json, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nBenchmark results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
