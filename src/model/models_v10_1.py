import torch
import torch.nn as nn
import torch.nn.functional as F

class VanguardWeightedRelLayerV10_1(nn.Module):
    """
    Relational GNN Layer with HEAVY GEOMETRIC INFUSION for Vanguard V10.1.
    Matches the information density of the classic MLP by injecting 
    weighted adjacency (dx, dy, Manhattan, Chebyshev).
    """
    def __init__(self, channels=128, num_relations=16):
        super().__init__()
        self.channels = channels
        self.num_relations = num_relations
        # Learnable weights for each logical relation
        self.rel_weights = nn.Parameter(torch.Tensor(num_relations, channels, channels))
        self.self_weight = nn.Linear(channels, channels)
        
        # Spatial Gate: Learns to modulate relations based on geometric weights (4 channels)
        # Input: 4 geometric features -> Output: 16 scaling factors
        self.spatial_gate = nn.Sequential(
            nn.Linear(4, 16),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(channels)
        nn.init.xavier_uniform_(self.rel_weights)

    def forward(self, h, adj_20):
        """
        h: [B, 64, channels]
        adj_20: [B, 20, 64, 64] 
                 (0-15: Logical Binary, 16-19: Geometric Weights)
        """
        B = h.shape[0]
        logical = adj_20[:, :16] # [B, 16, 64, 64]
        geo = adj_20[:, 16:]     # [B, 4, 64, 64]
        
        # 1. Transform spatial weights into relation gates
        # geo_flat: [B, 4, 64*64] -> [B, 64, 64, 4]
        geo_perm = geo.permute(0, 2, 3, 1) # [B, 64, 64, 4]
        gates = self.spatial_gate(geo_perm).permute(0, 3, 1, 2) # [B, 16, 64, 64]
        
        # 2. Apply Spatial Gates to Logical Adjacency
        weighted_adj = logical * gates # [B, 16, 64, 64]
        
        # 3. Relational Message Passing (Flattened for DML)
        out = self.self_weight(h)
        h_flat = h.unsqueeze(1).expand(B, 16, 64, self.channels).reshape(B * 16, 64, self.channels)
        w_flat = self.rel_weights.unsqueeze(0).expand(B, 16, self.channels, self.channels).reshape(B * 16, self.channels, self.channels)
        h_rel_flat = torch.bmm(h_flat, w_flat)
        
        adj_flat = weighted_adj.view(B * 16, 64, 64)
        messages_flat = torch.bmm(adj_flat, h_rel_flat)
        
        messages = messages_flat.view(B, 16, 64, self.channels).sum(dim=1)
        return F.gelu(self.norm(out + messages))

class VanguardV10_1(nn.Module):
    """
    Vanguard V10.1: Weighted Relational Graphs.
    Matches the MLP information density (~45 features) within a GNN.
    - Perspective Normalization (handled in loader)
    - Pawn Progress (1)
    - Weighted Adjacency (dx, dy, Manhattan, Chebyshev)
    """
    def __init__(self, node_dim=128, num_layers=4):
        super().__init__()
        self.piece_embed = nn.Embedding(14, 32)
        self.coord_proj = nn.Linear(2, 16)
        # Input: 32 (piece) + 16 (coord) + 4 (tac) + 1 (pawn prog) = 53
        self.node_init = nn.Linear(53, node_dim)
        
        self.gnn_layers = nn.ModuleList([
            VanguardWeightedRelLayerV10_1(node_dim, num_relations=16) 
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
        adj: [B, 20, 64, 64] (Weighted)
        """
        h_piece = self.piece_embed(p_ids)
        
        # Static Grid Coords (Normalized -1..1)
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

def reconstruct_dense_adj_v10_1(l_src, l_dst, l_et, l_weights, B, device='cpu'):
    """
    Reconstructs a [B, 20, 64, 64] adjacency tensor.
    l_weights: List of tensors [E_i, 4] containing (dx, dy, Manhattan, Chebyshev)
    """
    all_src = torch.cat(l_src)
    all_dst = torch.cat(l_dst)
    all_et = torch.cat(l_et)
    all_weights = torch.cat(l_weights) # [Total_E, 4]
    
    all_b = torch.cat([torch.full((len(s),), i, dtype=torch.long) for i, s in enumerate(l_src)])
    
    # 20 Channels: 16 Logical + 4 Geometric
    adj = torch.zeros((B, 20, 64, 64), device=device)
    
    # Populate logical channels (Binary)
    adj[all_b, all_et, all_src, all_dst] = 1.0
    
    # Populate geometric channels (Weighted)
    # We populate ALL 4 channels for EVERY edge (regardless of et)
    # This might overwrite if multiple relations exist between same nodes,
    # but the distances are the same for the same nodes.
    for i in range(4):
        adj[all_b, 16 + i, all_src, all_dst] = all_weights[:, i]
        
    return adj
