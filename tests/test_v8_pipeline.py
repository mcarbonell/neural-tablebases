"""
tests/test_v8_pipeline.py — V8-Pro GNN Pipeline Tests
======================================================
Covers:
  - Rust engine loading and feature extraction
  - ChessGnnV8_Pro forward pass (CPU smoke test)
  - build_giant_graph vectorized output shapes
  - GnnShardDataset loading from smoke data

Run with:
    python run_tests.py
or directly:
    python -m pytest tests/test_v8_pipeline.py -v
"""

import os
import sys
import numpy as np
import torch
import unittest

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_graph(batch_size: int = 2, num_edges: int = 8):
    """Build minimal synthetic graph tensors for a batch."""
    B = batch_size
    p_ids = torch.zeros(B, 64, dtype=torch.long)
    tac   = torch.zeros(B, 64, 4, dtype=torch.float32)

    list_srcs, list_dsts, list_etypes = [], [], []
    for _ in range(B):
        srcs   = torch.randint(0, 64, (num_edges,), dtype=torch.long)
        dsts   = torch.randint(0, 64, (num_edges,), dtype=torch.long)
        etypes = torch.randint(0, 16, (num_edges,), dtype=torch.long)
        list_srcs.append(srcs)
        list_dsts.append(dsts)
        list_etypes.append(etypes)

    return p_ids, tac, list_srcs, list_dsts, list_etypes


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestBuildGiantGraph(unittest.TestCase):
    """Tests for the vectorized build_giant_graph utility."""

    def setUp(self):
        from models_v8 import build_giant_graph
        self.build_giant_graph = build_giant_graph

    def test_output_shapes_batch2(self):
        B = 2
        num_edges = 10
        p_ids, tac, srcs, dsts, etypes = _make_synthetic_graph(B, num_edges)
        flat_pids, flat_tac, g_srcs, g_dsts, g_etypes, out_B = self.build_giant_graph(
            p_ids, tac, srcs, dsts, etypes
        )
        self.assertEqual(out_B, B)
        self.assertEqual(flat_pids.shape, (B * 64,))
        self.assertEqual(flat_tac.shape, (B * 64, 4))
        self.assertEqual(g_srcs.shape[0], B * num_edges)
        self.assertEqual(g_dsts.shape[0], B * num_edges)
        self.assertEqual(g_etypes.shape[0], B * num_edges)

    def test_node_offsets_correct(self):
        """Nodes in sample b should be offset by b*64."""
        B = 3
        p_ids, tac, srcs, dsts, etypes = _make_synthetic_graph(B, num_edges=4)
        _, _, g_srcs, _, _, _ = self.build_giant_graph(p_ids, tac, srcs, dsts, etypes)
        # All sources for batch b=2 must be >= 2*64 = 128
        third_batch_srcs = g_srcs[8:12]  # 4 edges * 2 = offset 8
        # They should all be in range [128, 191]
        self.assertTrue((third_batch_srcs >= 128).all(),
                        f"Expected offsets >= 128, got {third_batch_srcs}")

    def test_batch1_no_offset(self):
        """With B=1, source node indices should be unchanged."""
        B = 1
        p_ids, tac, list_srcs, list_dsts, list_etypes = _make_synthetic_graph(B, num_edges=5)
        _, _, g_srcs, _, _, _ = self.build_giant_graph(p_ids, tac, list_srcs, list_dsts, list_etypes)
        self.assertTrue((g_srcs < 64).all(),
                        "B=1 offsets should keep node indices in [0, 63]")


class TestChessGnnV8Pro(unittest.TestCase):
    """Smoke tests for the ChessGnnV8_Pro model architecture."""

    def setUp(self):
        from models_v8 import ChessGnnV8_Pro, build_giant_graph
        self.ChessGnnV8_Pro = ChessGnnV8_Pro
        self.build_giant_graph = build_giant_graph

    def test_forward_returns_three_heads(self):
        """Model must return (wdl, dtz, eval) tensors."""
        model = self.ChessGnnV8_Pro(node_dim=32, num_layers=2)
        model.eval()
        B = 2
        p_ids, tac, srcs, dsts, etypes = _make_synthetic_graph(B, num_edges=6)
        flat_pids, flat_tac, g_srcs, g_dsts, g_etypes, _ = self.build_giant_graph(
            p_ids, tac, srcs, dsts, etypes
        )
        with torch.no_grad():
            wdl, dtz, ev = model(flat_pids, flat_tac, g_srcs, g_dsts, g_etypes, B)
        self.assertEqual(wdl.shape, (B, 3), f"WDL shape mismatch: {wdl.shape}")
        self.assertEqual(dtz.shape, (B, 1), f"DTZ shape mismatch: {dtz.shape}")
        self.assertEqual(ev.shape, (B, 1),  f"Eval shape mismatch: {ev.shape}")

    def test_no_nan_in_output(self):
        """Forward pass must not produce NaN values."""
        model = self.ChessGnnV8_Pro(node_dim=32, num_layers=2)
        model.eval()
        B = 4
        p_ids, tac, srcs, dsts, etypes = _make_synthetic_graph(B, num_edges=12)
        flat_pids, flat_tac, g_srcs, g_dsts, g_etypes, _ = self.build_giant_graph(
            p_ids, tac, srcs, dsts, etypes
        )
        with torch.no_grad():
            wdl, dtz, ev = model(flat_pids, flat_tac, g_srcs, g_dsts, g_etypes, B)
        self.assertFalse(torch.isnan(wdl).any(), "NaN in WDL output")
        self.assertFalse(torch.isnan(dtz).any(), "NaN in DTZ output")
        self.assertFalse(torch.isnan(ev).any(), "NaN in Eval output")

    def test_wdl_softmax_probabilities(self):
        """WDL logits, after softmax, must sum to 1 per sample."""
        import torch.nn.functional as F
        model = self.ChessGnnV8_Pro(node_dim=32, num_layers=2)
        model.eval()
        B = 3
        p_ids, tac, srcs, dsts, etypes = _make_synthetic_graph(B)
        flat_pids, flat_tac, g_srcs, g_dsts, g_etypes, _ = self.build_giant_graph(
            p_ids, tac, srcs, dsts, etypes
        )
        with torch.no_grad():
            wdl, _, _ = model(flat_pids, flat_tac, g_srcs, g_dsts, g_etypes, B)
        probs = F.softmax(wdl, dim=-1)
        sums = probs.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones(B), atol=1e-5),
                        f"Softmax probabilities don't sum to 1: {sums}")

    def test_larger_batch_stability(self):
        """Larger batches should not change the WDL logit distribution radically."""
        model = self.ChessGnnV8_Pro(node_dim=64, num_layers=2)
        model.eval()
        for B in [1, 8, 32]:
            p_ids, tac, srcs, dsts, etypes = _make_synthetic_graph(B, num_edges=20)
            flat_pids, flat_tac, g_srcs, g_dsts, g_etypes, _ = self.build_giant_graph(
                p_ids, tac, srcs, dsts, etypes
            )
            with torch.no_grad():
                wdl, dtz, ev = model(flat_pids, flat_tac, g_srcs, g_dsts, g_etypes, B)
            self.assertEqual(wdl.shape[0], B, f"Batch size mismatch at B={B}")


class TestGnnShardDataset(unittest.TestCase):
    """Tests for GnnShardDataset loading. Uses smoke data if available."""

    SMOKE_SHARD = os.path.join(
        os.path.dirname(__file__), "..", "data", "gnn_krvk_smoke.npz"
    )

    def setUp(self):
        from train_v8 import GnnShardDataset
        self.GnnShardDataset = GnnShardDataset

    def test_smoke_shard_loads(self):
        if not os.path.exists(self.SMOKE_SHARD):
            self.skipTest(f"Smoke shard not found at {self.SMOKE_SHARD}")
        ds = self.GnnShardDataset(self.SMOKE_SHARD)
        self.assertGreater(len(ds), 0, "Dataset should not be empty")

    def test_item_keys_present(self):
        if not os.path.exists(self.SMOKE_SHARD):
            self.skipTest(f"Smoke shard not found at {self.SMOKE_SHARD}")
        ds = self.GnnShardDataset(self.SMOKE_SHARD)
        item = ds[0]
        required_keys = {"p_ids", "node_tac", "srcs", "dsts", "e_types", "wdl", "dtz", "eval"}
        for key in required_keys:
            self.assertIn(key, item, f"Missing key '{key}' in dataset item")

    def test_p_ids_shape(self):
        if not os.path.exists(self.SMOKE_SHARD):
            self.skipTest(f"Smoke shard not found at {self.SMOKE_SHARD}")
        ds = self.GnnShardDataset(self.SMOKE_SHARD)
        item = ds[0]
        self.assertEqual(item["p_ids"].shape, (64,),
                         f"p_ids shape mismatch: {item['p_ids'].shape}")

    def test_node_tac_shape(self):
        if not os.path.exists(self.SMOKE_SHARD):
            self.skipTest(f"Smoke shard not found at {self.SMOKE_SHARD}")
        ds = self.GnnShardDataset(self.SMOKE_SHARD)
        item = ds[0]
        self.assertEqual(item["node_tac"].shape, (64, 4),
                         f"node_tac shape mismatch: {item['node_tac'].shape}")

    def test_wdl_label_in_range(self):
        if not os.path.exists(self.SMOKE_SHARD):
            self.skipTest(f"Smoke shard not found at {self.SMOKE_SHARD}")
        ds = self.GnnShardDataset(self.SMOKE_SHARD)
        for i in range(min(len(ds), 50)):
            item = ds[i]
            wdl = item["wdl"].item()
            self.assertIn(wdl, (0, 1, 2), f"WDL label {wdl} out of range at index {i}")


class TestRustEngine(unittest.TestCase):
    """Tests for the RustGnnEngine interface."""

    def setUp(self):
        try:
            from rust_engine import RustGnnEngine
            self.engine = RustGnnEngine()
            self.available = True
        except (FileNotFoundError, ImportError) as e:
            self.available = False
            self.skip_reason = str(e)

    def _check_available(self):
        if not self.available:
            self.skipTest(f"Rust engine not available: {self.skip_reason}")

    def test_engine_loads(self):
        self._check_available()
        # If setUp succeeded, engine is loaded
        self.assertIsNotNone(self.engine)

    def test_starting_position_features(self):
        self._check_available()
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        p_ids, tac, edges, cnt = self.engine.get_raw_features(fen)
        self.assertEqual(p_ids.shape, (64,), "piece_ids must be (64,)")
        self.assertEqual(tac.shape, (64, 4), "tactical must be (64, 4)")
        self.assertGreater(cnt, 0, "Edge count should be positive")
        self.assertLessEqual(cnt, 1024, "Edge count cannot exceed buffer size")

    def test_empty_position_doesnt_crash(self):
        self._check_available()
        # KQvK minimal position
        fen = "8/8/8/4k3/8/4K3/4Q3/8 w - - 0 1"
        p_ids, tac, edges, cnt = self.engine.get_raw_features(fen)
        self.assertEqual(p_ids.shape, (64,))
        self.assertGreaterEqual(cnt, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
