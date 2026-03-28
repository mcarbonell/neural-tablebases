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
from models_v8 import ChessGnnV8_Pro, build_giant_graph

# Try to import DirectML for AMD
try:
    import torch_directml
    HAS_DIRECTML = True
except ImportError:
    HAS_DIRECTML = False

def setup_logger(model_name):
    """Sets up mandatory project logging as per GEMINI.md rules."""
    os.makedirs("data/logs", exist_ok=True)
    log_path = os.path.join("data/logs", f"train_{model_name}.log")
    
    # Save the exact command used to launch
    cmd_string = " ".join(sys.argv)
    header = f"LAUNCH COMMAND: python {cmd_string}\n"
    header += "="*50 + "\n"
    
    # Check if we append or start new
    mode = "a" if os.path.exists(log_path) else "w"
    with open(log_path, mode) as f:
        f.write(header if mode == "w" else f"\n\nRESUMING: {header}")
        
    def log_print(msg):
        print(msg)
        with open(log_path, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}\n")
    
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
        
        # Detect if we have Lichess-style evals (0..2) vs Tablebase (-2..2)
        self.is_lichess = np.max(self.wdl_raw) <= 2 and np.min(self.wdl_raw) >= 0
        
    def __len__(self):
        return len(self.p_ids)
    
    def __getitem__(self, idx):
        count = int(self.edge_counts[idx])
        raw_edges = self.edges[idx][:count]
        
        e_types = (raw_edges >> 12) & 0xF
        srcs = (raw_edges >> 6) & 0x3F
        dsts = raw_edges & 0x3F
        
        # Mapping WDL correctly
        wdl_val = self.wdl_raw[idx]
        if not self.is_lichess:
            wdl_mapped = np.sign(wdl_val).astype(np.int64) + 1
        else:
            wdl_mapped = wdl_val.astype(np.int64)
            
        dtz_val = self.dtz_raw[idx]
        eval_score = dtz_val if self.is_lichess else 0.0
        dtz_score = 0.0 if self.is_lichess else abs(dtz_val)
        
        return {
            "p_ids": torch.from_numpy(self.p_ids[idx].astype(np.int64)),
            "node_tac": torch.from_numpy(self.node_tac[idx].astype(np.float32)) / 8.0,
            "srcs": torch.from_numpy(srcs.astype(np.int64)),
            "dsts": torch.from_numpy(dsts.astype(np.int64)),
            "e_types": torch.from_numpy(e_types.astype(np.int64)),
            "wdl": torch.tensor(wdl_mapped, dtype=torch.long),
            "dtz": torch.tensor(dtz_score, dtype=torch.float32),
            "eval": torch.tensor(eval_score, dtype=torch.float32)
        }

def collate_vectorized(batch):
    B = len(batch)
    batch_pids = torch.stack([b["p_ids"] for b in batch])
    batch_tac = torch.stack([b["node_tac"] for b in batch])
    batch_wdl = torch.stack([b["wdl"] for b in batch])
    batch_dtz = torch.stack([b["dtz"] for b in batch])
    batch_eval = torch.stack([b["eval"] for b in batch])
    
    list_srcs = [b["srcs"] for b in batch]
    list_dsts = [b["dsts"] for b in batch]
    list_etypes = [b["e_types"] for b in batch]
    
    flat_pids, flat_tac, g_srcs, g_dsts, g_etypes, _ = build_giant_graph(
        batch_pids, batch_tac, list_srcs, list_dsts, list_etypes
    )
    
    return flat_pids, flat_tac, g_srcs, g_dsts, g_etypes, B, batch_wdl, batch_dtz, batch_eval

def train_v8():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/v8_shards")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--model_name", type=str, default="v8_pro_triple_head")
    args = parser.parse_args()
    
    log = setup_logger(args.model_name)
    
    if HAS_DIRECTML and torch_directml.is_available():
        device = torch_directml.device()
        log(f"Using DirectML (Radeon 780M) Acceleration.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"Using device: {device}")
        
    model = ChessGnnV8_Pro(node_dim=128, num_layers=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Add Cosine Annealing Scheduler for smooth convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    # Rule 4.4: Load latest checkpoint if available
    checkpoint_path = f"data/{args.model_name}_latest.pth"
    if os.path.exists(checkpoint_path):
        log(f"Loading weights from existing checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    
    log(f"Starting GNN-V8-Pro Triple Head Training: {args.model_name}")
    log(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # H1-A: Track best model across epochs
    best_wdl_acc = 0.0
    best_path = f"data/{args.model_name}_best.pth"

    for epoch in range(args.epochs):
        model.train()
        cur_lr = optimizer.param_groups[0]['lr']
        log(f"Epoch {epoch+1} starting. Current LR: {cur_lr:.6f}")
        
        # Init metrics
        total_correct_wdl = 0
        total_mae_dtz = 0
        total_mae_eval = 0
        total_pos = 0
        total_samples_dtz = 0
        total_samples_eval = 0
        
        # Refresh and shuffle shards each epoch to include newly generated data
        shards = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
        if not shards:
            log(f"WARNING: No shards found in {args.data_dir} at Epoch {epoch+1}")
            time.sleep(30)
            continue
            
        np.random.shuffle(shards)
        log(f"Epoch {epoch+1} starting with {len(shards)} shards.")
        
        last_acc = 0
        last_mae_eval = 0
        
        for s_idx, shard_path in enumerate(shards):
            ds = GnnShardDataset(shard_path)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, 
                              collate_fn=collate_vectorized, num_workers=2)
            
            start_shard = time.time()
            shard_pos = 0 # Track pos in current shard for speed
            
            for batch_idx, (p_ids, tac, srcs, dsts, etypes, B, wdl, dtz, ev) in enumerate(loader):
                p_ids, tac = p_ids.to(device), tac.to(device)
                srcs, dsts, etypes = srcs.to(device), dsts.to(device), etypes.to(device)
                wdl, dtz, ev = wdl.to(device), dtz.to(device), ev.to(device)
                
                optimizer.zero_grad()
                out_wdl, out_dtz, out_eval = model(p_ids, tac, srcs, dsts, etypes, B)
                
                # Triple Loss Calculation
                loss_wdl = F.cross_entropy(out_wdl, wdl)
                loss_dtz = F.mse_loss(out_dtz.squeeze(), dtz) if not ds.is_lichess else torch.tensor(0.0, device=device)
                loss_eval = F.mse_loss(out_eval.squeeze(), ev) if ds.is_lichess else torch.tensor(0.0, device=device)
                
                # Weighted Loss
                loss = loss_wdl + 0.1 * loss_dtz + 1.0 * loss_eval
                
                loss.backward()
                optimizer.step()
                
                # Metrics Calculation
                _, pred_wdl = out_wdl.max(1)
                total_correct_wdl += pred_wdl.eq(wdl).sum().item()
                
                if not ds.is_lichess:
                    total_mae_dtz += F.l1_loss(out_dtz.squeeze(), dtz).item() * B
                    total_samples_dtz += B
                else:
                    total_mae_eval += F.l1_loss(out_eval.squeeze(), ev).item() * B
                    total_samples_eval += B
                
                total_pos += B
                shard_pos += B
                
                if batch_idx % 100 == 0:
                    speed = shard_pos / (time.time() - start_shard + 1e-6)
                    cur_acc = total_correct_wdl / total_pos * 100
                    cur_mae_eval = total_mae_eval / (total_samples_eval + 1e-6)
                    
                    # Calculate Deltas
                    delta_acc = cur_acc - last_acc if last_acc > 0 else 0
                    delta_mae = cur_mae_eval - last_mae_eval if last_mae_eval > 0 else 0
                    
                    acc_str = f"{cur_acc:.2f}% ({delta_acc:+.2f}%)"
                    mae_str = f"{cur_mae_eval:.4f} ({delta_mae:+.4f})"
                    
                    log(f"E{epoch+1} | S{s_idx+1}/{len(shards)} | B{batch_idx} | "
                        f"L:{loss.item():.4f} | WDL-Acc:{acc_str} | "
                        f"Eval-MAE:{mae_str} | "
                        f"Spd:{speed:.1f} pos/s")
                    
                    last_acc = cur_acc
                    last_mae_eval = cur_mae_eval

            torch.save(model.state_dict(), f"data/{args.model_name}_latest.pth")

        # H1-A: Save best model checkpoint
        epoch_acc = total_correct_wdl / max(total_pos, 1) * 100
        if epoch_acc > best_wdl_acc:
            best_wdl_acc = epoch_acc
            torch.save(model.state_dict(), best_path)
            log(f"★ NEW BEST saved: {best_path} | WDL-Acc: {best_wdl_acc:.2f}%")

        log(f"Epoch {epoch+1} complete. WDL-Acc: {epoch_acc:.2f}% | Best: {best_wdl_acc:.2f}%")

        # Rule 4.5: Update LR
        scheduler.step()

if __name__ == "__main__":
    train_v8()
