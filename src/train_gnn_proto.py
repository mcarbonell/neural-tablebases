import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
import time
from torch.utils.data import Dataset, DataLoader
# torch_directml import moved to train function

class GnnDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.p_ids = data['p_ids']
        self.node_tac = data['node_tac']
        self.edges = data['edges']
        self.edge_counts = data['edge_counts']
        self.wdl = data['wdl']
        self.dtz = data['dtz']
        
    def __len__(self):
        return len(self.p_ids)
    
    def __getitem__(self, idx):
        # Convert to tensors
        p_ids = torch.from_numpy(self.p_ids[idx].astype(np.int64))
        node_tac = torch.from_numpy(self.node_tac[idx].astype(np.float32)) / 8.0 # Normalizing atk counts?
        
        # Edges: (1024,) u16
        # Need to decode: (type, src, dst)
        count = int(self.edge_counts[idx])
        raw_edges = self.edges[idx][:count]
        
        e_types = (raw_edges >> 12) & 0xF
        srcs = (raw_edges >> 6) & 0x3F
        dsts = raw_edges & 0x3F
        
        # We'll use these to build a sparse adjacency in the model or here
        return {
            "p_ids": p_ids,
            "node_tac": node_tac,
            "srcs": torch.from_numpy(srcs.astype(np.int64)),
            "dsts": torch.from_numpy(dsts.astype(np.int64)),
            "e_types": torch.from_numpy(e_types.astype(np.int64)),
            "wdl": torch.tensor(np.sign(self.wdl[idx]).astype(np.int64) + 1, dtype=torch.long), # Maps -2,-1,0,1,2 -> 0,1,2
            "dtz": torch.tensor(abs(self.dtz[idx]), dtype=torch.float32)
        }

def collate_gnn(batch):
    """
    Collate function to handle variable number of edges per position.
    """
    p_ids = torch.stack([b["p_ids"] for b in batch])
    node_tac = torch.stack([b["node_tac"] for b in batch])
    wdl = torch.stack([b["wdl"] for b in batch])
    dtz = torch.stack([b["dtz"] for b in batch])
    
    # For edges, we need to offset the indices for batching if we use global sparse ops
    # But for a simple "Hello World" with loops or manual indexing:
    srcs = [b["srcs"] for b in batch]
    dsts = [b["dsts"] for b in batch]
    e_types = [b["e_types"] for b in batch]
    
    return p_ids, node_tac, srcs, dsts, e_types, wdl, dtz

class SimpleGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_relations=16):
        super().__init__()
        self.num_relations = num_relations
        self.rel_weights = nn.Parameter(torch.randn(num_relations, in_dim, out_dim) * 0.1)
        self.self_weight = nn.Linear(in_dim, out_dim)
        
    def forward(self, x, batch_srcs, batch_dsts, batch_etypes):
        # x: (B, 64, in_dim)
        B, N, D = x.size()
        out = self.self_weight(x)
        
        # Message passing
        for b in range(B):
            src = batch_srcs[b]
            dst = batch_dsts[b]
            et = batch_etypes[b]
            
            # Group edges by type for this position
            for r in range(self.num_relations):
                mask = (et == r)
                if mask.any():
                    # x[b, src[mask]] is (num_edges_of_type, in_dim)
                    # self.rel_weights[r] is (in_dim, out_dim)
                    msgs = x[b, src[mask]] @ self.rel_weights[r]
                    out[b].index_add_(0, dst[mask], msgs)
                    
        return F.relu(out)

class ChessGnnV8(nn.Module):
    def __init__(self, node_dim=64):
        super().__init__()
        self.embed = nn.Embedding(14, 16) # 0-13 piece IDs
        self.node_lin = nn.Linear(16 + 4, node_dim) # Embed + Tactical node features
        
        self.gnn1 = SimpleGNNLayer(node_dim, node_dim)
        self.gnn2 = SimpleGNNLayer(node_dim, node_dim)
        
        self.wdl_head = nn.Linear(node_dim * 64, 3)
        self.dtz_head = nn.Linear(node_dim * 64, 1)
        
    def forward(self, p_ids, node_tac, srcs, dsts, etypes):
        # p_ids: (B, 64)
        # node_tac: (B, 64, 4)
        
        h = torch.cat([self.embed(p_ids), node_tac], dim=-1) # (B, 64, 20)
        h = F.relu(self.node_lin(h)) # (B, 64, node_dim)
        
        h = self.gnn1(h, srcs, dsts, etypes)
        h = self.gnn2(h, srcs, dsts, etypes)
        
        # Flatten for the heads (Pooling would be better for general chess, but ok for tablebases)
        flat = h.view(h.size(0), -1)
        
        wdl = self.wdl_head(flat)
        dtz = self.dtz_head(flat)
        return wdl, dtz

def train_gnn():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/gnn_krvk_smoke.npz")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    # Hardware
    try:
        import torch_directml
        device = torch_directml.device() if torch_directml.is_available() else torch.device("cpu")
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data
    dataset = GnnDataset(args.data)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_gnn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_gnn)
    
    # Model
    model = ChessGnnV8().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Starting V8 GNN Proto Training on {len(train_ds)} positions...")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        start = time.time()
        for p_ids, node_tac, srcs, dsts, etypes, wdl, dtz in train_loader:
            # Move to device
            p_ids = p_ids.to(device)
            node_tac = node_tac.to(device)
            wdl = wdl.to(device)
            dtz = dtz.to(device)
            # srcs, dsts, etypes are lists of tensors, move individually
            srcs = [s.to(device) for s in srcs]
            dsts = [d.to(device) for d in dsts]
            etypes = [e.to(device) for e in etypes]
            
            optimizer.zero_grad()
            out_wdl, out_dtz = model(p_ids, node_tac, srcs, dsts, etypes)
            
            loss_wdl = F.cross_entropy(out_wdl, wdl)
            loss_dtz = F.mse_loss(out_dtz.squeeze(), dtz)
            loss = loss_wdl + 0.1 * loss_dtz
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, pred = out_wdl.max(1)
            correct += pred.eq(wdl).sum().item()
            total += wdl.size(0)
            
        elapsed = time.time() - start
        acc = correct / total
        
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc*100:.2f}% | Time: {elapsed:.1f}s", flush=True)
        
    print("Training complete! V8 Hello World success.")

if __name__ == "__main__":
    train_gnn()
