import torch
import numpy as np
import sys
import chess

sys.path.append('src')
from search.rust_engine import RustGnnEngine
from model.models_v8 import ChessGnnV8_Pro, build_giant_graph
from data.canonical_forms import get_all_symmetries

device = torch.device('cpu')
model = ChessGnnV8_Pro().to(device)
model.load_state_dict(torch.load('data/v8_universal_35M_latest.pth', map_location='cpu', weights_only=False))
model.eval()

engine = RustGnnEngine()

# Test position: KRvK
fen = "8/8/8/8/5k2/8/4R3/4K3 w - - 0 1"
board = chess.Board(fen)
symmetries = get_all_symmetries(board)

print(f"Testing {len(symmetries)} symmetries of FEN: {fen}")

results = []
for i, sym_board in enumerate(symmetries):
    p_ids, tac, edges, cnt = engine.get_raw_features(sym_board.fen())
    raw_edges = edges[:cnt]
    e_types = (raw_edges >> 12) & 0xF
    srcs = (raw_edges >> 6) & 0x3F
    dsts = raw_edges & 0x3F
    
    list_pids = torch.from_numpy(p_ids.astype(np.int64)).unsqueeze(0)
    list_tac = torch.from_numpy(tac.astype(np.float32)).unsqueeze(0) / 8.0
    list_srcs = [torch.from_numpy(srcs.astype(np.int64))]
    list_dsts = [torch.from_numpy(dsts.astype(np.int64))]
    list_etypes = [torch.from_numpy(e_types.astype(np.int64))]
    
    flat_pids, flat_tac, g_srcs, g_dsts, g_etypes, B = build_giant_graph(
        list_pids, list_tac, list_srcs, list_dsts, list_etypes
    )
    
    with torch.no_grad():
        wdl, _, _ = model(flat_pids, flat_tac, g_srcs, g_dsts, g_etypes, B)
        # Handle perspective: the model is absolute (White-relative)
        # But for symmetry testing, we just see if the logits are stable
        # Wait, if we reflect the board, the piece positions change.
        # If the model is a GNN, it should be invariant IF the edges and node IDs are consistent.
        results.append(torch.softmax(wdl, dim=-1).numpy())

results = np.array(results)
print("Logit variance across symmetries:")
print(np.var(results, axis=0))
print("Max diff across symmetries:")
print(np.max(results, axis=0) - np.min(results, axis=0))
