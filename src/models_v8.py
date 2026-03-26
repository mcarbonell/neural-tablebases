import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GNNLayer(nn.Module):
    """
    Multi-Channel GNN Layer. 
    Processes different types of edges (Mobility, Attack, X-Ray) with separate weights.
    """
    def __init__(self, in_features, out_features, num_channels=4):
        super(GNNLayer, self).__init__()
        # Separate transforms for each edge channel
        self.channel_linears = nn.ModuleList([
            nn.Linear(in_features, out_features) for _ in range(num_channels)
        ])
        self.norm = nn.LayerNorm(out_features)
        self.shortcut = nn.Identity() if in_features == out_features else nn.Linear(in_features, out_features)
        
    def forward(self, x, adj_channels):
        """
        x: [Batch, 64, in_features]
        adj_channels: [Batch, 64, 64, num_channels]
        """
        out = 0
        for i, linear in enumerate(self.channel_linears):
            # Aggregation per channel: support = X @ W, then result = Adj @ support
            support = linear(x) 
            # Slice the specific channel adjacency matrix
            channel_adj = adj_channels[:, :, :, i] 
            out = out + torch.matmul(channel_adj, support)
        
        # Residual + Norm
        return F.relu(self.norm(out + self.shortcut(x)))

class ChessGNN(nn.Module):
    """
    Evolution 8.1 (GNN-Fusion): 
    Integrates tactical node features and multi-channel edges from Rust engine.
    """
    def __init__(self, node_features=32, tactical_features=8, hidden_dim=128, num_layers=4):
        super(ChessGNN, self).__init__()
        self.piece_emb = nn.Embedding(13, node_features)
        self.coord_proj = nn.Linear(2, 8)
        
        input_dim = node_features + tactical_features + 8
        
        self.layers = nn.ModuleList([
            GNNLayer(input_dim if i == 0 else hidden_dim, hidden_dim, num_channels=4)
            for i in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.wdl_head = nn.Linear(hidden_dim * 2, 3)
        self.dtz_head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x_piece_ids, x_tactical, adj_channels):
        """
        x_piece_ids: [B, 64]
        x_tactical: [B, 64, 8] (From Rust engine: attack counts, flags, etc.)
        adj_channels: [B, 64, 64, 4]
        """
        batch_size = x_piece_ids.size(0)
        device = x_piece_ids.device
        
        # 1. Base Node Features
        h_piece = self.piece_emb(x_piece_ids)
        
        # 2. Geometric Features
        ranks = torch.arange(8, device=device).repeat_interleave(8).float() / 7.0
        files = torch.arange(8, device=device).repeat(8).float() / 7.0
        coords = torch.stack([ranks, files], dim=-1).unsqueeze(0).expand(batch_size, -1, -1)
        h_coord = self.coord_proj(coords)
        
        # 3. Concatenate all features (Neural + Tactical + Geo)
        h = torch.cat([h_piece, x_tactical, h_coord], dim=-1)
        
        # 4. Message Passing through Tactical Channels
        for layer in self.layers:
            h = layer(h, adj_channels)
            
        h = self.final_norm(h)
        
        # 5. Global Valuation Pooling
        pooled = torch.cat([torch.mean(h, dim=1), torch.max(h, dim=1)[0]], dim=-1)
        
        return self.wdl_head(pooled), self.dtz_head(pooled)

def get_gnn_model(num_layers=3, hidden_dim=64):
    return ChessGNN(num_layers=num_layers, hidden_dim=hidden_dim)
