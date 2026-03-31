import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import time
import argparse
import sys
import chess

from model.models_v10 import VanguardV10, reconstruct_dense_adj

# Setup device: AMD/DirectML on Windows, else CPU/CUDA
if sys.platform == "win32":
    import torch_directml
    device = torch_directml.device()
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class GnnShardDatasetV10(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.p_ids = data['p_ids']
        self.node_tac = data['node_tac']
        self.edges = data['edges'] # [N, 1024] Encoded uint16
        self.edge_counts = data['edge_counts']
        self.wdl = data['wdl']
        self.dtz = data['dtz']
        self.turns = data['turn']

    def __len__(self):
        return len(self.p_ids)

    def __getitem__(self, idx):
        p_ids = self.p_ids[idx]
        tac = self.node_tac[idx]
        turn = self.turns[idx]
        wdl = self.wdl[idx]
        dtz = self.dtz[idx]
        
        # Decode edges from packed uint16
        count = int(self.edge_counts[idx])
        encoded = self.edges[idx][:count]
        
        l_et = (encoded >> 12) & 0xF
        l_src = (encoded >> 6) & 0x3F
        l_dst = encoded & 0x3F
        
        # Perspective Normalization (V10 Improvement)
        if turn == 0: # Black to move (0=Black, 1=White)
            p_ids_sq = p_ids.reshape(8, 8)
            p_ids_flipped = np.flipud(p_ids_sq)
            
            def swap_color(p):
                if p == 0: return 0
                if p <= 6: return p + 6
                return p - 6
            
            p_ids = np.vectorize(swap_color)(p_ids_flipped).flatten()
            
            def flip_sq(s):
                r, f = s // 8, s % 8
                return (7 - r) * 8 + f
            
            l_src = np.array([flip_sq(s) for s in l_src])
            l_dst = np.array([flip_sq(s) for s in l_dst])
        
        # Pawn Progress Feature (Geometric Fusion)
        pawn_prog = np.zeros(64, dtype=np.float32)
        for sq in range(64):
            p = p_ids[sq]
            if p == 1: # White Pawn (always STM)
                pawn_prog[sq] = (sq // 8) / 7.0
            elif p == 7: # Black Pawn
                pawn_prog[sq] = (7 - (sq // 8)) / 7.0

        wdl_cat = 1
        if wdl > 0: wdl_cat = 2
        elif wdl < 0: wdl_cat = 0

        return (torch.from_numpy(p_ids).long(), 
                torch.from_numpy(tac).float(), 
                torch.from_numpy(pawn_prog).float().unsqueeze(1),
                torch.from_numpy(l_src).long(),
                torch.from_numpy(l_dst).long(),
                torch.from_numpy(l_et).long(),
                torch.tensor(wdl_cat, dtype=torch.long), 
                torch.tensor(dtz, dtype=torch.float32))

        # Pawn Progress Feature (Geometric Fusion)
        pawn_prog = np.zeros(64, dtype=np.float32)
        for sq in range(64):
            p = p_ids[sq]
            if p == 1: # White Pawn (which is always us after flip)
                pawn_prog[sq] = (sq // 8) / 7.0
            elif p == 7: # Black Pawn (enemy)
                pawn_prog[sq] = (7 - (sq // 8)) / 7.0

        # WDL to categorical [0, 1, 2]
        # Syzygy: -2..-1 -> 0, 0 -> 1, 1..2 -> 2
        wdl_cat = 1
        if wdl > 0: wdl_cat = 2
        elif wdl < 0: wdl_cat = 0

        return (torch.from_numpy(p_ids).long(), 
                torch.from_numpy(tac).float(), 
                torch.from_numpy(pawn_prog).float().unsqueeze(1),
                torch.tensor(src, dtype=torch.long),
                torch.tensor(dst, dtype=torch.long),
                torch.tensor(et, dtype=torch.long),
                torch.tensor(wdl_cat, dtype=torch.long), 
                torch.tensor(dtz, dtype=torch.float32))

def collate_gnn_v10(batch):
    p_ids, tac, pawn_prog, l_src, l_dst, l_et, wdl, dtz = zip(*batch)
    
    p_ids_t = torch.stack(p_ids)
    tac_t = torch.stack(tac)
    pawn_prog_t = torch.stack(pawn_prog)
    wdl_t = torch.stack(wdl)
    dtz_t = torch.stack(dtz)
    
    adj_t = reconstruct_dense_adj(l_src, l_dst, l_et, len(p_ids), device='cpu')
    
    return p_ids_t, tac_t, pawn_prog_t, adj_t, wdl_t, dtz_t, len(p_ids)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--name", type=str, default="v10_smoke")
    args = parser.parse_args()
    
    log = setup_logger(args.name)
    log(f"Starting Vanguard V10 (GEOMETRIC FUSION GNN): {args.name}")
    
    # 1. Model
    model = VanguardV10().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion_wdl = nn.CrossEntropyLoss()
    criterion_dtz = nn.MSELoss()
    
    # 2. Data
    shards = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    log(f"Found {len(shards)} shards.")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss, total_acc, total_mae, samples = 0, 0, 0, 0
        last_loss, last_acc, last_mae = 0, 0, 0
        
        for shard in shards:
            dataset = GnnShardDatasetV10(shard)
            loader = DataLoader(
                dataset, 
                batch_size=args.batch_size, 
                shuffle=True, 
                collate_fn=collate_gnn_v10,
                num_workers=4,
                pin_memory=True
            )
            
            epoch_start_time = time.time()
            epoch_samples = 0
            
            for i, (p, tac, prog, adj, wdl, dtz, B) in enumerate(loader):
                p, tac, prog, adj, wdl, dtz = [x.to(device) for x in [p, tac, prog, adj, wdl, dtz]]
                
                optimizer.zero_grad()
                out_wdl, out_dtz = model(p, tac, prog, adj, B)
                
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
                    if last_loss == 0: d_loss, d_acc, d_mae = 0, 0, 0
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

if __name__ == "__main__":
    train()
