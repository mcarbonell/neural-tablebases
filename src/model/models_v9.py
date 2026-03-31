import torch
import torch.nn as nn
import torch.nn.functional as F

class VanguardDenseRelLayer(nn.Module):
    """
    Relational GNN Layer (3D Flattened) for Vanguard V9.
    Optimizado para DirectML (Radeon 780M) usando torch.bmm en 3D.
    Aplanamos los 16 canales en la dimensión de batch para máxima velocidad.
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
        """
        h: [B, 64, channels]
        adj: [B, 16, 64, 64]
        """
        B = h.shape[0]
        
        # 1. Contribución propia (Auto-activación)
        out = self.self_weight(h)
        
        # 2. Paso de Mensajes 3D (Aplanado para DirectML)
        # H: [B, 64, C] -> Expansión: [B, 16, 64, C] -> Aplanado: [B*16, 64, C]
        h_flat = h.unsqueeze(1).expand(B, 16, 64, self.channels).reshape(B * 16, 64, self.channels)
        
        # Pesos: [16, C, C] -> Expansión: [B*16, C, C]
        w_flat = self.rel_weights.unsqueeze(0).expand(B, 16, self.channels, self.channels).reshape(B * 16, self.channels, self.channels)
        
        # Transformación de Nodos (3D BMM): [B*16, 64, C] @ [B*16, C, C]
        h_rel_flat = torch.bmm(h_flat, w_flat)
        
        # Adyacencia: [B, 16, 64, 64] -> Aplanado: [B*16, 64, 64]
        adj_flat = adj.view(B * 16, 64, 64)
        
        # Mensajes (3D BMM): [B*16, 64, 64] @ [B*16, 64, C]
        messages_flat = torch.bmm(adj_flat, h_rel_flat)
        
        # Recuperar canales y sumar: [B, 16, 64, C] -> [B, 64, C]
        messages = messages_flat.view(B, 16, 64, self.channels).sum(dim=1)
        
        # 3. Residual & Norm & Activation
        return F.gelu(self.norm(out + messages))

class VanguardV9(nn.Module):
    def __init__(self, node_dim=128, num_layers=4):
        super().__init__()
        self.piece_embed = nn.Embedding(14, 32)
        self.coord_proj = nn.Linear(2, 16)
        self.node_init = nn.Linear(52, node_dim)
        self.gnn_layers = nn.ModuleList([
            VanguardDenseRelLayer(node_dim, num_relations=16) 
            for _ in range(num_layers)
        ])
        self.pool_dim = node_dim * 2
        self.wdl_head = nn.Sequential(
            nn.Linear(self.pool_dim, 128), nn.GELU(), nn.Linear(128, 3)
        )
        self.dtz_head = nn.Sequential(
            nn.Linear(self.pool_dim, 128), nn.GELU(), nn.Linear(128, 1)
        )

    def forward(self, p_ids, tac, adj, batch_size):
        h_piece = self.piece_embed(p_ids)
        y = (torch.arange(64, device=p_ids.device)).view(1, 64).repeat(batch_size, 1) // 8
        x = (torch.arange(64, device=p_ids.device)).view(1, 64).repeat(batch_size, 1) % 8
        coords = (torch.stack([x.float(), y.float()], dim=2) / 7.0) - 1.0
        h_coord = self.coord_proj(coords)
        h_in = torch.cat([h_piece, h_coord, tac / 8.0], dim=2)
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
