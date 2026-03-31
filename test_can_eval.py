import sys, os, torch, chess, chess.syzygy, numpy as np
sys.path.append(os.path.abspath('src'))
from model.models_v8 import ChessGnnV8_Pro, build_giant_graph
from search.rust_engine import RustGnnEngine
from data.canonical_forms import find_canonical_form

device = torch.device('cpu')
model = ChessGnnV8_Pro().to(device)
model.load_state_dict(torch.load('data/v8_universal_35M_latest.pth', map_location='cpu', weights_only=False))
model.eval()

engine = RustGnnEngine()
tb = chess.syzygy.open_tablebase('syzygy')

# Original board: KRvK non-canonical
fen = "8/8/8/8/5k2/8/4R3/4K3 w - - 0 1"
board = chess.Board(fen)
board.castling_rights = 0
true_wdl = tb.probe_wdl(board) # 2
print(f"Original FEN: {fen}, True WDL: {true_wdl}")

# Canonical version
can_board, _ = find_canonical_form(board, None)
print(f"Canonical FEN: {can_board.fen()}")

p_ids, tac, edges, cnt = engine.get_raw_features(can_board.fen())
p_ids_t = torch.from_numpy(p_ids.astype(np.int64)).unsqueeze(0)
tac_t = torch.from_numpy(tac.astype(np.float32)).unsqueeze(0) / 8.0
raw_edges = edges[:cnt]
list_srcs = [torch.from_numpy(((raw_edges >> 6) & 0x3F).astype(np.int64))]
list_dsts = [torch.from_numpy((raw_edges & 0x3F).astype(np.int64))]
list_etypes = [torch.from_numpy(((raw_edges >> 12) & 0xF).astype(np.int64))]

flat_pids, flat_tac, g_srcs, g_dsts, g_etypes, B = build_giant_graph(p_ids_t, tac_t, list_srcs, list_dsts, list_etypes)

with torch.no_grad():
    out_wdl, _, _ = model(flat_pids, flat_tac, g_srcs, g_dsts, g_etypes, B)
    pred = torch.argmax(out_wdl, dim=1).item()

if can_board.turn == chess.BLACK: pred = 2 - pred
print(f"Canonical Prediction: {pred}")
