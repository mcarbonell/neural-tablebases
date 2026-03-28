import torch
import torch.nn as nn
import torch.nn.functional as F

class RGNNLayerVectorized(nn.Module):
    """
    Relational Graph Neural Network Layer (Vectorized).
    """
    def __init__(self, in_dim, out_dim, num_relations=16, dropout=0.1):
        super().__init__()
        self.num_relations = num_relations
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Xavier initialization
        self.rel_weights = nn.Parameter(torch.randn(num_relations, in_dim, out_dim) * (2.0 / (in_dim + out_dim))**0.5)
        self.self_weight = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, srcs, dsts, etypes, batch_size):
        # 1. Self propagation
        out = self.self_weight(x)
        
        # 2. Relation-wise message passing
        for r in range(self.num_relations):
            mask = (etypes == r)
            if not mask.any(): continue
            
            msgs = x[srcs[mask]] @ self.rel_weights[r]
            out.index_add_(0, dsts[mask], msgs)

        out = self.bn(out)
        out = F.gelu(out)
        return self.dropout(out)

class GlobalAttentionPooling(nn.Module):
    def __init__(self, node_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(node_dim, node_dim // 2),
            nn.GELU(),
            nn.Linear(node_dim // 2, 1)
        )

    def forward(self, x, batch_size):
        scores = self.attention(x) 
        scores = scores.view(batch_size, 64)
        weights = F.softmax(scores, dim=1).view(batch_size * 64, 1)
        
        weighted_nodes = x * weights
        weighted_nodes = weighted_nodes.view(batch_size, 64, -1)
        return torch.sum(weighted_nodes, dim=1)

class ChessGnnV8_Pro(nn.Module):
    """
    V8-Pro Architecture - Optimized Universal Chess Evaluator.
    """
    def __init__(self, node_dim=128, num_layers=4, dropout=0.1):
        super().__init__()
        self.node_dim = node_dim
        
        self.piece_embed = nn.Embedding(14, 32)
        self.init_lin = nn.Linear(32 + 4, node_dim)
        
        self.layers = nn.ModuleList([
            RGNNLayerVectorized(node_dim, node_dim, dropout=dropout) for _ in range(num_layers)
        ])
        
        self.pool = GlobalAttentionPooling(node_dim)
        self.ln_post = nn.LayerNorm(node_dim)
        
        # Head 1: WDL (Win/Draw/Loss)
        self.wdl_head = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, 3)
        )
        
        # Head 2: DTZ (Depth/Endgame)
        self.dtz_head = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, node_dim // 2),
            nn.GELU(),
            nn.Linear(node_dim // 2, 1)
        )
        
        # Head 3: EVAL (Centipawn Intuition)
        self.eval_head = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, node_dim // 2),
            nn.GELU(),
            nn.Linear(node_dim // 2, 1)
        )

    def forward(self, p_ids, node_tac, srcs, dsts, etypes, batch_size):
        h_piece = self.piece_embed(p_ids)
        h = torch.cat([h_piece, node_tac], dim=-1)
        h = F.gelu(self.init_lin(h))
        
        for layer in self.layers:
            h = h + layer(h, srcs, dsts, etypes, batch_size)
            
        global_features = self.pool(h, batch_size)
        global_features = self.ln_post(global_features) # Stabilize features
        
        wdl = self.wdl_head(global_features)
        dtz = self.dtz_head(global_features)
        eval_score = self.eval_head(global_features)
        
        return wdl, dtz, eval_score

def build_giant_graph(batch_pids, batch_tac, list_srcs, list_dsts, list_etypes):
    """Build batched graph by offsetting node indices. Vectorized (no Python loops)."""
    B = len(batch_pids)
    flat_pids = batch_pids.view(-1)
    flat_tac = batch_tac.view(-1, 4)
    # Compute per-sample offset (b * 64) without a Python loop
    offsets = torch.arange(B, dtype=torch.long) * 64
    all_srcs = torch.cat([s + off for s, off in zip(list_srcs, offsets)])
    all_dsts = torch.cat([d + off for d, off in zip(list_dsts, offsets)])
    all_etypes = torch.cat(list_etypes)
    return flat_pids, flat_tac, all_srcs, all_dsts, all_etypes, B
