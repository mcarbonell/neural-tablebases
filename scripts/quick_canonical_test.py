import sys, os, torch, chess, chess.syzygy, numpy as np, random
sys.path.append(os.path.abspath('src'))
from model.models_v8 import ChessGnnV8_Pro, build_giant_graph
from search.rust_engine import RustGnnEngine
from data.canonical_forms import find_canonical_form

def get_eval(board, model, engine, device):
    p_ids, tac, edges, cnt = engine.get_raw_features(board.fen())
    p_ids_t = torch.from_numpy(p_ids.astype(np.int64)).unsqueeze(0).to(device)
    tac_t = (torch.from_numpy(tac.astype(np.float32)).unsqueeze(0) / 8.0).to(device)
    raw_edges = edges[:cnt]
    list_srcs = [torch.from_numpy(((raw_edges >> 6) & 0x3F).astype(np.int64))]
    list_dsts = [torch.from_numpy((raw_edges & 0x3F).astype(np.int64))]
    list_etypes = [torch.from_numpy(((raw_edges >> 12) & 0xF).astype(np.int64))]
    flat_pids, flat_tac, g_srcs, g_dsts, g_etypes, B = build_giant_graph(p_ids_t, tac_t, list_srcs, list_dsts, list_etypes)
    g_srcs, g_dsts, g_etypes = g_srcs.to(device), g_dsts.to(device), g_etypes.to(device)
    with torch.no_grad():
        out_wdl, _, _ = model(flat_pids, flat_tac, g_srcs, g_dsts, g_etypes, B)
        pred = torch.argmax(out_wdl, dim=1).item()
    if board.turn == chess.BLACK: pred = 2 - pred
    return pred

device = torch.device('cpu')
model = ChessGnnV8_Pro().to(device)
model.load_state_dict(torch.load('data/v8_universal_35M_latest.pth', map_location='cpu', weights_only=False))
model.eval()
engine = RustGnnEngine()
tb = chess.syzygy.open_tablebase('syzygy')

# Set of 20 random KQvK positions
positions = []
while len(positions) < 20:
    board = chess.Board(None)
    board.set_piece_at(random.choice(range(64)), chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(random.choice(range(64)), chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(random.choice(range(64)), chess.Piece(chess.KING, chess.BLACK))
    board.turn = random.choice([chess.WHITE, chess.BLACK])
    board.castling_rights = 0
    if board.is_valid() and not board.is_checkmate() and not board.is_stalemate():
        try:
            wdl = tb.probe_wdl(board)
            true_wdl = 1 if wdl == 0 else (0 if wdl < 0 else 2)
            positions.append((board.copy(), true_wdl))
        except: pass

print(f"Testing 20 KQvK positions...")
correct_raw = 0
correct_can = 0
for board, true_wdl in positions:
    try:
        pr = get_eval(board, model, engine, device)
        if pr == true_wdl: correct_raw += 1
        can_board, _ = find_canonical_form(board, None)
        pc = get_eval(can_board, model, engine, device)
        if pc == true_wdl: correct_can += 1
    except: pass

print(f"RESULT:")
print(f"  Baseline Accuracy: {correct_raw/20*100:.1f}%")
print(f"  Canonical Accuracy: {correct_can/20*100:.1f}%")
