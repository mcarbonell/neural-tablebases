import torch
import torch.nn as nn
import torch.nn.functional as F

class VanguardV11_FieldTheory(nn.Module):
    """
    Vanguard V11: Square-Centric Field Theory GNN.
    Nodes: 64 Fixed (A1 to H8)
    Edges: 16-channel Adjacency Tensor from Rust Engine (handles occlusion).
    """
    def __init__(self, node_dim=128, tactical_dim=4, layers=8):
        super().__init__()
        self.node_dim = node_dim
        self.layers = layers
        
        # 1. Input Projectors
        self.piece_embed = nn.Embedding(14, node_dim // 2)
        self.tac_proj = nn.Linear(tactical_dim, node_dim // 2)
        
        # 2. Field Update Layers (GNN Layers)
        # Each layer processes message passing across 16 connectivity channels
        self.gnn_layers = nn.ModuleList([
            FieldInfluenceLayer(node_dim, channels=16) for _ in range(layers)
        ])
        
        # 3. Decision Heads (WDL / DTZ)
        self.wdl_head = nn.Sequential(
            nn.Linear(node_dim * 64, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Linear(256, 3)
        )
        # ... (DTZ head similarly)

    def forward(self, piece_ids, tactical_features, adj_t):
        """
        piece_ids: [Batch, 64]
        tactical_features: [Batch, 64, 4]
        adj_t: [Batch, 64, 64, 16] (Connectivity from Rust)
        """
        # Embed pieces and project tactical info
        p_emb = self.piece_embed(piece_ids) # [B, 64, dim/2]
        t_emb = self.tac_proj(tactical_features) # [B, 64, dim/2]
        
        # Initial square state
        x = torch.cat([p_emb, t_emb], dim=-1) # [Batch, 64, node_dim]
        
        # Propagation phase (N layers = N plies)
        for gnn in self.gnn_layers:
            x = gnn(x, adj_t)
            
        # Global Readout
        x_flat = x.view(-1, 64 * self.node_dim)
        return self.wdl_head(x_flat)

class FieldInfluenceLayer(nn.Module):
    """
    Propagates messages through 16 types of connectivity channels.
    """
    def __init__(self, node_dim, channels=16):
        super().__init__()
        self.node_dim = node_dim
        self.channels = channels
        
        # Message transform for each channel
        self.channel_weights = nn.Parameter(torch.ones(channels))
        self.msg_net = nn.Linear(node_dim, node_dim)
        
        self.norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(0.05)

    def forward(self, x, adj_t):
        """
        x: [B, 64, node_dim]
        adj_t: [B, 64, 64, 16]
        """
        B, N, D = x.shape
        # Transform node state to message
        msg = self.msg_net(x) # [B, 64, D]
        
        # Aggregate across all 16 channels
        # adj_t sum over channels: [B, 64, 64]
        # Weighted mixture of connectivity
        combined_adj = torch.sum(adj_t * self.channel_weights.view(1, 1, 1, -1), dim=-1)
        
        # Square-to-Square message pass (MatMul)
        # updated_msg = Adj @ Msg
        out = torch.bmm(combined_adj, msg) # [B, 64, D]
        
        return self.norm(x + self.dropout(out))

# Helper to generate the static move-patterns (adjacency matrices)
def get_move_adjacency():
    # Will return tensors for [KingMoves, KnightMoves, Diagonal, Orthogonal]
    # Size each: [64, 64]
    pass
