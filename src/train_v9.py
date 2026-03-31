import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
import time
import sys
import glob
from torch.utils.data import Dataset, DataLoader

# Internal Imports
sys.path.append(os.path.dirname(__file__))
from model.models_v9 import VanguardV9, reconstruct_dense_adj

# Setup Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    import torch_directml
    device = torch_directml.device()
    print(f"Using DirectML device: {device}")
except ImportError:
    print(f"Using device: {device}")

def setup_logger(model_name):
    """Sets up mandatory project logging with Elapsed Time tracking."""
    os.makedirs("data/logs", exist_ok=True)
    log_path = os.path.join("data/logs", f"train_{model_name}.log")
    
    start_time = time.time()
    start_date = time.strftime('%Y-%m-%d %H:%M:%S')
    
    cmd_string = " ".join(sys.argv)
    header = f"LAUNCH COMMAND: python {cmd_string}\n"
    header += f"LAUNCH TIME: {start_date}\n"
    header += "="*50 + "\n"
    
    with open(log_path, "a") as f:
        f.write(header)
        
    def log_print(msg):
        print(msg)
        elapsed = time.time() - start_time
        # Format as H:M:S
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        timestamp = f"{h:02d}:{m:02d}:{s:02d}"
        
        with open(log_path, "a") as f:
            f.write(f"{timestamp} | {msg}\n")
    
    return log_print

class GnnShardDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.p_ids = data['p_ids']
        self.node_tac = data['node_tac']
        self.edges = data['edges']
        self.edge_counts = data['edge_counts']
        self.wdl_raw = data['wdl']
        self.dtz_raw = data['dtz']
        
    def __len__(self):
        return len(self.p_ids)
    
    def __getitem__(self, idx):
        count = int(self.edge_counts[idx])
        raw_edges = self.edges[idx][:count]
        e_types = (raw_edges >> 12) & 0xF
        srcs = (raw_edges >> 6) & 0x3F
        dsts = raw_edges & 0x3F
        
        # Normalize WDL from Tablebase (-2..2) to (0..2)
        raw_wdl = self.wdl_raw[idx]
        if raw_wdl > 0: wdl = 2
        elif raw_wdl < 0: wdl = 0
        else: wdl = 1
        
        return (
            torch.from_numpy(self.p_ids[idx].astype(np.int64)),
            torch.from_numpy(self.node_tac[idx].astype(np.float32)),
            torch.from_numpy(srcs.astype(np.int64)),
            torch.from_numpy(dsts.astype(np.int64)),
            torch.from_numpy(e_types.astype(np.int64)),
            torch.tensor(wdl, dtype=torch.long),
            torch.tensor(self.dtz_raw[idx], dtype=torch.float32)
        )

def collate_gnn(batch):
    p_ids, tac, l_src, l_dst, l_et, wdl, dtz = zip(*batch)
    
    p_ids_t = torch.stack(p_ids)
    tac_t = torch.stack(tac)
    wdl_t = torch.stack(wdl)
    dtz_t = torch.stack(dtz)
    
    # Dense Adjacency reconstruction (small 64x64 matrices)
    # Perform on CPU in worker processes to keep GPU fed.
    adj_t = reconstruct_dense_adj(l_src, l_dst, l_et, len(p_ids), device='cpu')
    
    return p_ids_t, tac_t, adj_t, wdl_t, dtz_t, len(p_ids)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--name", type=str, default="v9_smoke")
    args = parser.parse_args()
    
    log = setup_logger(args.name)
    log(f"Starting Training V9 (FLATTENED 3D): {args.name}")
    
    # 1. Model
    model = VanguardV9().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion_wdl = nn.CrossEntropyLoss()
    criterion_dtz = nn.MSELoss()
    
    # 2. Data
    shards = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    log(f"Found {len(shards)} shards.")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        total_mae = 0
        samples = 0
        
        # Track deltas for logging visibility
        last_loss, last_acc, last_mae = 0, 0, 0
        
        for shard in shards:
            dataset = GnnShardDataset(shard)
            # Parallel Loading with 4 Workers and Pin Memory for PCIe speed.
            loader = DataLoader(
                dataset, 
                batch_size=args.batch_size, 
                shuffle=True, 
                collate_fn=collate_gnn,
                num_workers=4,
                pin_memory=True
            )
            
            epoch_start_time = time.time()
            epoch_samples = 0
            
            for i, (p, t, adj, wdl, dtz, B) in enumerate(loader):
                # Transfer to GPU
                p, t, adj, wdl, dtz = [x.to(device) for x in [p, t, adj, wdl, dtz]]
                
                optimizer.zero_grad()
                out_wdl, out_dtz = model(p, t, adj, B)
                
                # Balanced Loss
                loss_wdl = criterion_wdl(out_wdl, wdl)
                loss_dtz = criterion_dtz(out_dtz.squeeze(), dtz / 100.0)
                loss = loss_wdl + 0.1 * loss_dtz
                
                loss.backward()
                optimizer.step()
                
                # Metrics
                curr_loss = loss.item()
                total_loss += curr_loss
                
                preds = torch.argmax(out_wdl, dim=1)
                curr_acc = (preds == wdl).sum().item() / B
                total_acc += (preds == wdl).sum().item()
                
                curr_mae = torch.abs(out_dtz.squeeze() * 100.0 - dtz).mean().item()
                total_mae += curr_mae * B
                
                samples += B
                epoch_samples += B
                
                if i % 10 == 0:
                    avg_acc = total_acc / samples
                    avg_mae = total_mae / samples
                    
                    # Calculate Deltas (variation since last log line)
                    if last_loss == 0: # First log of epoch, no delta yet
                        d_loss, d_acc, d_mae = 0, 0, 0
                    else:
                        d_loss = curr_loss - last_loss
                        d_acc = avg_acc - last_acc
                        d_mae = avg_mae - last_mae
                    
                    last_loss, last_acc, last_mae = curr_loss, avg_acc, avg_mae
                    
                    elapsed = time.time() - epoch_start_time
                    speed = epoch_samples / elapsed if elapsed > 0 else 0
                    lr = optimizer.param_groups[0]['lr']
                    
                    log(f"Epoch {epoch} | Batch {i} | Loss: {curr_loss:.4f} ({d_loss:+.4f}) | Acc: {avg_acc:.4f} ({d_acc:+.4f}) | MAE: {avg_mae:.2f} ({d_mae:+.2f}) | Speed: {speed:.0f} pos/s | LR: {lr:.2e}")
                    
        # Checkpoint summary
        avg_acc = total_acc / samples
        avg_mae = total_mae / samples
        log(f"--- EPOCA {epoch} FINALIZADA ---")
        log(f"  Accuracy: {avg_acc:.4f}")
        log(f"  DTZ-MAE: {avg_mae:.2f} moves")
        
        os.makedirs("data/checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"data/checkpoints/{args.name}_epoch_{epoch}.pth")
        
    log("Training Complete.")

if __name__ == "__main__":
    train()
