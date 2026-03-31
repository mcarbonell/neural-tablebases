import torch
import torch.nn.functional as F
import numpy as np
import sys
import chess
import chess.syzygy

sys.path.append('src')
from search.rust_engine import RustGnnEngine

def test_permutation(permutation_idx, shift, scale_coords, flip_coords):
    from model.v8_vanguard import ChessGnnV8_Vanguard, build_giant_graph_vanguard
    
    device = torch.device('cpu')
    model = ChessGnnV8_Vanguard().to(device)
    model.load_state_dict(torch.load('data/v8_universal_35M_latest.pth', map_location='cpu', weights_only=False))
    model.eval()
    
    def modified_forward(self, p_ids, node_tac, srcs, dsts, etypes, B):
        remap_ids = p_ids.clone()
        if shift:
            remap_ids = torch.where(remap_ids == 0, torch.tensor(1, device=device), remap_ids + 1)
            
        h_piece = self.embed(remap_ids)
        
        ranks = torch.arange(8, device=device).repeat_interleave(8).float()
        files = torch.arange(8, device=device).repeat(8).float()
        if scale_coords:
            ranks = ranks / 3.5 - 1.0
            files = files / 3.5 - 1.0
        else:
            ranks = ranks / 7.0
            files = files / 7.0
            
        if flip_coords:
            coords = torch.stack([files, ranks], dim=-1)
        else:
            coords = torch.stack([ranks, files], dim=-1)
            
        h_coord = self.coord_proj(coords).repeat(B, 1)
        
        if permutation_idx == 0:
            h = torch.cat([h_piece, h_coord, node_tac], dim=-1)
        elif permutation_idx == 1:
            h = torch.cat([h_piece, node_tac, h_coord], dim=-1)
        elif permutation_idx == 2:
            h = torch.cat([h_coord, h_piece, node_tac], dim=-1)
        elif permutation_idx == 3:
            h = torch.cat([h_coord, node_tac, h_piece], dim=-1)
        elif permutation_idx == 4:
            h = torch.cat([node_tac, h_piece, h_coord], dim=-1)
        elif permutation_idx == 5:
            h = torch.cat([node_tac, h_coord, h_piece], dim=-1)
            
        h = F.gelu(self.node_proj(h))
        
        for layer in self.gnn_blocks:
            h = h + layer(h, srcs, dsts, etypes, B)
            
        h_reshaped = h.view(B, 64, self.node_dim)
        pooled = torch.cat([torch.mean(h_reshaped, dim=1), torch.max(h_reshaped, dim=1)[0]], dim=-1)
        
        wdl = self.wdl_head(pooled)
        dtz = self.dtz_head(pooled)
        return wdl, dtz, None
        
    import types
    model.forward = types.MethodType(modified_forward, model)
    
    engine = RustGnnEngine()    
    tablebase = chess.syzygy.open_tablebase('syzygy')
    
    sys.path.append('scripts')
    from benchmark_neural_vs_sf import sample_random_positions
    test_positions = sample_random_positions("KRvK", tablebase, count=50) # use 50 for speed
    
    boards = [p[0] for p in test_positions]
    ground_truth = [p[1] for p in test_positions]
    
    batch_pids = []
    batch_tac = []
    list_srcs = []
    list_dsts = []
    list_etypes = []
    
    for board in boards:
        p_ids, tac, edges, edge_count = engine.get_raw_features(board.fen())
        raw_edges = edges[:edge_count]
        e_types = (raw_edges >> 12) & 0xF
        srcs = (raw_edges >> 6) & 0x3F
        dsts = raw_edges & 0x3F
        
        batch_pids.append(torch.from_numpy(p_ids.astype(np.int64)))
        batch_tac.append(torch.from_numpy(tac.astype(np.float32)) / 8.0)
        list_srcs.append(torch.from_numpy(srcs.astype(np.int64)))
        list_dsts.append(torch.from_numpy(dsts.astype(np.int64)))
        list_etypes.append(torch.from_numpy(e_types.astype(np.int64)))

    p_ids_t = torch.stack(batch_pids).to(device)
    # WHAT IF WE DON'T DIVIDE TAC BY 8?
    # Actually wait. Let's just try TAC * 1 / 8 vs just TAC
    tac_t = torch.stack(batch_tac).to(device)
    
    flat_pids, flat_tac, all_srcs, all_dsts, all_etypes, B = build_giant_graph_vanguard(
        p_ids_t, tac_t, list_srcs, list_dsts, list_etypes
    )

    with torch.no_grad():
        out_wdl, _, _ = model(flat_pids, flat_tac, all_srcs, all_dsts, all_etypes, B)
        preds = torch.argmax(out_wdl, dim=1).cpu().numpy()
        
    acc = sum(1 for p, g in zip(preds, ground_truth) if p == g) / len(boards)
    tablebase.close()
    return acc

for shift in [True, False]:
    for scale in [True, False]:
        for flip in [True, False]:
            for i in range(6):
                try:
                    acc = test_permutation(i, shift, scale, flip)
                    if acc > 0.8:
                        print(f"BINGO! Shift={shift}, Scale={scale}, Flip={flip}, Perm={i}: Acc = {acc*100:.2f}%")
                except Exception as e:
                    pass
print("Done")
