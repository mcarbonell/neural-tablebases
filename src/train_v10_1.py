import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import time
import argparse
from tqdm import tqdm
import sys
import chess

from model.models_v10_1 import VanguardV10_1, reconstruct_dense_adj_v10_1

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
    
    with open(log_path, "w") as f: # Overwrite for new version
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

class GnnShardDatasetV10_1(Dataset):
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
        
        # Calculate Geometric Weights (V10.1 Innovation) - Vectorized
        # Features: [dx, dy, Manhattan, Chebyshev]
        rs, fs = l_src // 8, l_src % 8
        rd, fd = l_dst // 8, l_dst % 8
        
        dx = (rd.astype(np.int32) - rs.astype(np.int32))
        dy = (fd.astype(np.int32) - fs.astype(np.int32))
        
        weights = np.zeros((len(l_src), 4), dtype=np.float32)
        weights[:, 0] = dx / 7.0
        weights[:, 1] = dy / 7.0
        weights[:, 2] = (np.abs(dx) + np.abs(dy)) / 14.0
        weights[:, 3] = np.maximum(np.abs(dx), np.abs(dy)) / 7.0

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
                torch.from_numpy(weights).float(),
                torch.tensor(wdl_cat, dtype=torch.long), 
                torch.tensor(dtz, dtype=torch.float32))

def collate_gnn_v10_1(batch):
    p_ids, tac, pawn_prog, l_src, l_dst, l_et, l_weights, wdl, dtz = zip(*batch)
    
    p_ids_t = torch.stack(p_ids)
    tac_t = torch.stack(tac)
    pawn_prog_t = torch.stack(pawn_prog)
    wdl_t = torch.stack(wdl)
    dtz_t = torch.stack(dtz)
    
    adj_t = reconstruct_dense_adj_v10_1(l_src, l_dst, l_et, l_weights, len(p_ids), device='cpu')
    
    return p_ids_t, tac_t, pawn_prog_t, adj_t, wdl_t, dtz_t, len(p_ids)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs", type=str, required=True, help="Comma-separated list of data directories")
    parser.add_argument("--epochs", type=int, default=100) # Increased for 99.9% goal
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--name", type=str, default="v10_1_weighted")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--workers", type=int, default=4, help="Number of data loader workers")
    args = parser.parse_args()
    
    log = setup_logger(args.name)
    log(f"Starting Vanguard V10.1 (WEIGHTED GEOMETRIC INFUSION): {args.name}")
    
    # 1. Model
    model = VanguardV10_1().to(device)
    log(f"Model architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Total trainable parameters: {total_params:,}")
    
    if args.resume:
        log(f"Resuming from checkpoint: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5, verbose=True)
    criterion_wdl = nn.CrossEntropyLoss()
    criterion_dtz = nn.MSELoss()
    
    # 2. Data
    # Collect shards from multiple directories (e.g. 4P and 5P datasets)
    shards = []
    for d in args.data_dirs.split(','):
        d_clean = d.strip()
        found = glob.glob(os.path.join(d_clean, "*.npz"))
        print(f"Found {len(found)} shards in {d_clean}")
        shards.extend(found)
    shards = sorted(shards)
    
    if not shards:
        print("No shards found in any directory!")
        return
    
    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss, total_acc, total_mae, samples = 0, 0, 0, 0
        epoch_start_time = time.time()
        last_log_metrics = None
        
        # Shuffle shards each epoch for balanced universal training
        epoch_shards = shards.copy()
        np.random.shuffle(epoch_shards)
        
        for shard in epoch_shards:
            dataset = GnnShardDatasetV10_1(shard)
            loader = DataLoader(
                dataset, 
                batch_size=args.batch_size, 
                shuffle=True, 
                num_workers=args.workers,
                pin_memory=True,
                collate_fn=collate_gnn_v10_1
            )
            
            pbar = tqdm(loader, desc=f"Epoch {epoch} | Shard {os.path.basename(shard)}", leave=False)
            for batch_idx, (p_ids, tac, pawn_prog, adj, wdl, dtz, B) in enumerate(pbar):
                p_ids, tac, pawn_prog, adj = p_ids.to(device), tac.to(device), pawn_prog.to(device), adj.to(device)
                wdl, dtz = wdl.to(device), dtz.to(device).float()
                
                optimizer.zero_grad()
                out_wdl, out_dtz = model(p_ids, tac, pawn_prog, adj, B)
                
                # Target WDL is 0, 1, 2 (mapped in collate).
                loss_wdl = criterion_wdl(out_wdl, wdl)
                loss_dtz = criterion_dtz(out_dtz.squeeze(), dtz)
                loss = loss_wdl + 0.1 * loss_dtz
                
                loss.backward()
                optimizer.step()
                
                # Metrics (global epoch accumulation)
                samples += B
                total_loss += loss.item() * B
                
                preds = torch.argmax(out_wdl, dim=1)
                total_acc += (preds == wdl).sum().item()
                total_mae += torch.abs(out_dtz.squeeze() - dtz).sum().item()
                
                if batch_idx % 10 == 0:
                    speed = samples / (time.time() - epoch_start_time + 1e-6)
                    pbar.set_postfix({"Acc": f"{total_acc/samples:.4f}", "MAE": f"{total_mae/samples:.2f}", "S": f"{speed:.0f}"})
                
                # Periodic DISK logging every 100 batches (as per GEMINI.md rules)
                if batch_idx % 100 == 0:
                    avg_loss = total_loss / samples
                    avg_acc = total_acc / samples
                    avg_mae = total_mae / samples
                    lr = optimizer.param_groups[0]['lr']
                    speed = samples / (time.time() - epoch_start_time + 1e-6)
                    
                    # Calculate deltas (+/-) from PREVIOUS log line
                    if last_log_metrics:
                        d_loss = avg_loss - last_log_metrics['loss']
                        d_acc = avg_acc - last_log_metrics['acc']
                        d_mae = avg_mae - last_log_metrics['mae']
                    else:
                        d_loss, d_acc, d_mae = 0, 0, 0
                    
                    last_log_metrics = {'loss': avg_loss, 'acc': avg_acc, 'mae': avg_mae}
                    
                    log(f"Epoch {epoch} | Batch {batch_idx} | Loss: {avg_loss:.4f} ({d_loss:+.4f}) | Acc: {avg_acc:.4f} ({d_acc:+.4f}) | MAE: {avg_mae:.2f} ({d_mae:+.2f}) | Speed: {speed:.0f} pos/s | LR: {lr:.2e}")

            # End of Shard: Update BEST model
            shard_acc = total_acc / samples
            if shard_acc > best_acc:
                best_acc = shard_acc
                os.makedirs("data/checkpoints", exist_ok=True)
                torch.save(model.state_dict(), f"data/checkpoints/{args.name}_BEST.pth")
                log(f"New GLOBAL BEST after shard {os.path.basename(shard)}: {best_acc:.4f}. Saved.")

        # End of Epoch
        avg_acc = total_acc / samples
        scheduler.step(avg_acc)
        log(f"Epoch {epoch} finished. Global Acc: {avg_acc:.4f}")
        torch.save(model.state_dict(), f"data/checkpoints/{args.name}_epoch_{epoch}.pth")

if __name__ == "__main__":
    train()
