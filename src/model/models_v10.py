import torch
import torch.nn as nn
import torch.nn.functional as F

class VanguardDenseRelLayerV10(nn.Module):
    """
    Relational GNN Layer (3D Flattened) for Vanguard V10.
    """
    def __init__(self, channels=128, num_relations=16):
        super().__init__()
        self.channels = channels
        self.num_relations = num_relations
        self.rel_weights = nn.Parameter(torch.Tensor(num_relations, channels, channels))
        self.self_weight = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
        nn.init.xavier_uniform_(self.rel_weights)

    def forward(self, h, adj):
        B = h.shape[0]
        out = self.self_weight(h)
        # Flattening for DML efficiency
        h_flat = h.unsqueeze(1).expand(B, 16, 64, self.channels).reshape(B * 16, 64, self.channels)
        w_flat = self.rel_weights.unsqueeze(0).expand(B, 16, self.channels, self.channels).reshape(B * 16, self.channels, self.channels)
        h_rel_flat = torch.bmm(h_flat, w_flat)
        adj_flat = adj.view(B * 16, 64, 64)
        messages_flat = torch.bmm(adj_flat, h_rel_flat)
        messages = messages_flat.view(B, 16, 64, self.channels).sum(dim=1)
        return F.gelu(self.norm(out + messages))

class VanguardV10(nn.Module):
    """
    Vanguard V10: Geometric Fusion GNN.
    - Piece Embedding (32)
    - Coordinates (16)
    - Tactical Info (4)
    - Pawn Progress (1) <-- NEW (Geometric Fusion)
    """
    def __init__(self, node_dim=128, num_layers=4):
        super().__init__()
        self.piece_embed = nn.Embedding(14, 32)
        self.coord_proj = nn.Linear(2, 16)
        # Input: 32 (piece) + 16 (coord) + 4 (tac) + 1 (pawn prog) = 53
        self.node_init = nn.Linear(53, node_dim)
        
        self.gnn_layers = nn.ModuleList([
            VanguardDenseRelLayerV10(node_dim, num_relations=16) 
            for _ in range(num_layers)
        ])
        
        self.pool_dim = node_dim * 2
        self.wdl_head = nn.Sequential(
            nn.Linear(self.pool_dim, 128), nn.GELU(), nn.Linear(128, 3)
        )
        self.dtz_head = nn.Sequential(
            nn.Linear(self.pool_dim, 128), nn.GELU(), nn.Linear(128, 1)
        )

    def forward(self, p_ids, tac, pawn_prog, adj, batch_size):
        """
        p_ids: [B, 64]
        tac: [B, 64, 4]
        pawn_prog: [B, 64, 1]
        adj: [B, 16, 64, 64]
        """
        h_piece = self.piece_embed(p_ids)
        
        # Static Grid Coords (Normalized)
        y = (torch.arange(64, device=p_ids.device)).view(1, 64).repeat(batch_size, 1) // 8
        x = (torch.arange(64, device=p_ids.device)).view(1, 64).repeat(batch_size, 1) % 8
        coords = (torch.stack([x.float(), y.float()], dim=2) / 7.0) - 1.0
        h_coord = self.coord_proj(coords)
        
        # Fusion: [Piece, Coord, Tac, PawnProg]
        h_in = torch.cat([h_piece, h_coord, tac / 8.0, pawn_prog], dim=2)
        h = F.gelu(self.node_init(h_in))
        
        for layer in self.gnn_layers:
            h = layer(h, adj)
            
        h_pool = torch.cat([h.mean(dim=1), h.max(dim=1)[0]], dim=1)
        return self.wdl_head(h_pool), self.dtz_head(h_pool)

def reconstruct_dense_adj(l_src, l_dst, l_et, B, device='cpu'):
    all_src = torch.cat(l_src)
    all_dst = torch.cat(l_dst)
    all_et = torch.cat(l_et)
    all_b = torch.cat([torch.full((len(s),), i, dtype=torch.long) for i, s in enumerate(l_src)])
    adj = torch.zeros((B, 16, 64, 64), device=device)
    adj[all_b, all_et, all_src, all_dst] = 1.0
    return adj
