import torch
import torch.nn.functional as F
import numpy as np
import sys
import chess
import chess.syzygy

sys.path.append('src')
from search.rust_engine import RustGnnEngine
from data.canonical_forms import is_canonical

def test_everything():
    from model.v8_vanguard import ChessGnnV8_Vanguard, build_giant_graph_vanguard
    device = torch.device('cpu')
    model = ChessGnnV8_Vanguard().to(device)
    model.load_state_dict(torch.load('data/v8_universal_35M_latest.pth', map_location='cpu', weights_only=False))
    model.eval()
    
    tablebase = chess.syzygy.open_tablebase('syzygy')
    sys.path.append('scripts')
    from benchmark_neural_vs_sf import sample_random_positions
    test_positions = sample_random_positions("KRvK", tablebase, count=100)
    engine = RustGnnEngine()    
    
    batch_pids, batch_tac, list_srcs, list_dsts, list_etypes = [], [], [], [], []
    ground_truth = []
    
    for board, wdl in test_positions:
        ground_truth.append(wdl)
        p_ids, tac, edges, edge_count = engine.get_raw_features(board.fen())
        raw_edges = edges[:edge_count]
        e_types = (raw_edges >> 12) & 0xF
        srcs = (raw_edges >> 6) & 0x3F
        dsts = raw_edges & 0x3F
        batch_pids.append(torch.from_numpy(p_ids.astype(np.int64)))
        batch_tac.append(torch.from_numpy(tac.astype(np.float32)))
        list_srcs.append(torch.from_numpy(srcs.astype(np.int64)))
        list_dsts.append(torch.from_numpy(dsts.astype(np.int64)))
        list_etypes.append(torch.from_numpy(e_types.astype(np.int64)))

    p_ids_t = torch.stack(batch_pids).to(device)
    tac_t = torch.stack(batch_tac).to(device)
    flat_pids, flat_tac, all_srcs, all_dsts, all_etypes, B = build_giant_graph_vanguard(
        p_ids_t, tac_t, list_srcs, list_dsts, list_etypes
    )

    best_acc = 0.0
    best_config = ""
    target_acc = 0.99
    
    for shift in [True, False]:
        for coord_scale in [1.0, 7.0, 3.5]:
            for coord_shift in [0.0, 1.0]:
                for flip_rf in [False, True]:
                    for tac_scale in [1.0, 8.0]:
                        for perm in range(6):
                            def modified_forward(p_ids, node_tac, srcs, dsts, etypes, B):
                                remap_ids = p_ids.clone()
                                if shift:
                                    remap_ids = torch.where(remap_ids == 0, torch.tensor(1, device=device), remap_ids + 1)
                                h_piece = model.embed(remap_ids)
                                
                                ranks = torch.arange(8, device=device).repeat_interleave(8).float()
                                files = torch.arange(8, device=device).repeat(8).float()
                                ranks = ranks / coord_scale - coord_shift
                                files = files / coord_scale - coord_shift
                                
                                if flip_rf: coords = torch.stack([files, ranks], dim=-1)
                                else:       coords = torch.stack([ranks, files], dim=-1)
                                
                                h_coord = model.coord_proj(coords).repeat(B, 1)
                                
                                tac_in = node_tac / tac_scale
                                
                                if perm == 0:   h = torch.cat([h_piece, h_coord, tac_in], dim=-1)
                                elif perm == 1: h = torch.cat([h_piece, tac_in, h_coord], dim=-1)
                                elif perm == 2: h = torch.cat([h_coord, h_piece, tac_in], dim=-1)
                                elif perm == 3: h = torch.cat([h_coord, tac_in, h_piece], dim=-1)
                                elif perm == 4: h = torch.cat([tac_in, h_piece, h_coord], dim=-1)
                                elif perm == 5: h = torch.cat([tac_in, h_coord, h_piece], dim=-1)
                                    
                                h = F.gelu(model.node_proj(h))
                                for layer in model.gnn_blocks:
                                    h = h + layer(h, srcs, dsts, etypes, B)
                                h_reshaped = h.view(B, 64, model.node_dim)
                                pooled = torch.cat([torch.mean(h_reshaped, dim=1), torch.max(h_reshaped, dim=1)[0]], dim=-1)
                                return model.wdl_head(pooled)

                            with torch.no_grad():
                                out_wdl = modified_forward(flat_pids, flat_tac, all_srcs, all_dsts, all_etypes, B)
                                preds = torch.argmax(out_wdl, dim=1).cpu().numpy()
                                
                            acc = sum(1 for p, g in zip(preds, ground_truth) if p == g) / len(test_positions)
                            if acc > best_acc:
                                best_acc = acc
                                best_config = f"perm={perm}, shift={shift}, c_scale={coord_scale}, c_shift={coord_shift}, flip_rf={flip_rf}, t_scale={tac_scale}"
                            if acc >= target_acc:
                                print(f"MATCH FOUND: {best_config} with acc {acc}")
                                return
    print(f"Max acc: {best_acc} for config: {best_config}")

test_everything()
