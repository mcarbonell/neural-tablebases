import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import argparse
import time
from models import get_model_for_endgame
from datetime import datetime

class StreamingShardedDataset(Dataset):
    """
    Loads chunks of data (shards) from a directory one by one.
    To be efficient, we load the whole shard into memory, then return items.
    """
    def __init__(self, shards_dir):
        self.shards_dir = shards_dir
        self.shard_files = sorted([f for f in os.listdir(shards_dir) if f.endswith('.npz')])
        if not self.shard_files:
            raise ValueError(f"No .npz shards found in {shards_dir}")
            
        # Get metadata from first shard
        first_shard = np.load(os.path.join(shards_dir, self.shard_files[0]))
        self.input_size = first_shard['x'].shape[1]
        
        # Calculate total size
        self.total_size = 0
        self.shard_lengths = []
        for f in self.shard_files:
            # We peek into the NPZ metadata without loading the whole array (if possible)
            data = np.load(os.path.join(shards_dir, f), mmap_mode='r')
            length = data['x'].shape[0]
            self.total_size += length
            self.shard_lengths.append(length)
            
        print(f"Streaming dataset initialized: {len(self.shard_files)} shards, {self.total_size:,} total positions.")
        
        self.current_shard_idx = -1
        self.x_data = None
        self.wdl_data = None
        self.dtz_data = None
        self.shard_offsets = np.cumsum([0] + self.shard_lengths)[:-1]

    def __len__(self):
        return self.total_size

    def _load_shard(self, idx):
        if self.current_shard_idx == idx:
            return
        
        shard_path = os.path.join(self.shards_dir, self.shard_files[idx])
        # print(f"Loading shard {idx}: {self.shard_files[idx]}...")
        data = np.load(shard_path)
        self.x_data = torch.from_numpy(data['x']).float()
        self.wdl_data = torch.from_numpy(data['wdl']).long()
        self.dtz_data = torch.from_numpy(data['dtz']).float().unsqueeze(1)
        self.current_shard_idx = idx

    def __getitem__(self, idx):
        # Find which shard this global index belongs to
        shard_idx = np.searchsorted(self.shard_offsets, idx, side='right') - 1
        self._load_shard(shard_idx)
        
        local_idx = idx - self.shard_offsets[shard_idx]
        return self.x_data[local_idx], self.wdl_data[local_idx], self.dtz_data[local_idx]

def train_large_scale(args):
    # Setup Device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        try:
            import torch_directml
            device = torch_directml.device()
            print(f"Using DirectML: {device}")
        except:
            pass
    
    print(f"Starting Large-Scale Training on {device}")
    
    # Dataset
    full_dataset = StreamingShardedDataset(args.data_dir)
    
    # Simple Split (for safety, avoid huge shuffles)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices = np.arange(train_size)
    val_indices = np.arange(train_size, len(full_dataset))
    
    train_loader = DataLoader(
        torch.utils.data.Subset(full_dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=False, # ¡CRÍTICO! Debe ser False. Los shards ya están barajados internamente. Si es True, causa saltos de disco masivos y OOM.
        num_workers=0, # Higher is complex with on-demand loading
        pin_memory=False
    )
    
    val_loader = DataLoader(
        torch.utils.data.Subset(full_dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False
    )

    # Model
    model = get_model_for_endgame(
        model_type=args.model,
        num_pieces=args.num_pieces,
        num_wdl_classes=3,
        input_size=full_dataset.input_size
    ).to(device)

    if args.load_path and os.path.exists(args.load_path):
        print(f"Loading weights from {args.load_path} for continuous learning...")
        try:
            state_dict = torch.load(args.load_path, map_location=device)
            model.load_state_dict(state_dict, strict=True)
            print("Weights loaded successfully!")
        except Exception as e:
            print(f"WARNING: Could not load weights from {args.load_path}: {e}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    wdl_criterion = nn.CrossEntropyLoss()
    dtz_criterion = nn.MSELoss()

    best_val_acc = 0.0
    
    print(f"Entering training loop... (Target: {args.epochs} epochs)")
    
    for epoch in range(args.epochs):
        model.train()
        start_time = time.time()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for i, (x_batch, wdl_batch, dtz_batch) in enumerate(train_loader):
            x_batch, wdl_batch, dtz_batch = x_batch.to(device), wdl_batch.to(device), dtz_batch.to(device)
            
            # WDL labels: convert -2,0,2 to 0,1,2
            wdl_labels = (wdl_batch + 2) // 2 
            
            optimizer.zero_grad()
            wdl_logits, dtz_out = model(x_batch)
            
            loss_wdl = wdl_criterion(wdl_logits, wdl_labels)
            loss_dtz = dtz_criterion(dtz_out, dtz_batch)
            
            loss = loss_wdl + 0.1 * loss_dtz
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(wdl_logits, 1)
            total += wdl_labels.size(0)
            correct += (predicted == wdl_labels).sum().item()
            
            if i % 100 == 0:
                elapsed = time.time() - start_time
                throughput = total / elapsed if elapsed > 0 else 0
                print(f"Epoch {epoch+1} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {100*correct/total:2f}% | Speed: {throughput:.0f} pos/s")

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x_batch, wdl_batch, dtz_batch in val_loader:
                x_batch, wdl_batch = x_batch.to(device), wdl_batch.to(device)
                wdl_labels = (wdl_batch + 2) // 2
                logits, _ = model(x_batch)
                _, pred = torch.max(logits, 1)
                val_total += wdl_labels.size(0)
                val_correct += (pred == wdl_labels).sum().item()
        
        val_acc = val_correct / val_total
        print(f"END EPOCH {epoch+1} | Val Acc: {val_acc:.4f} | Time: {time.time() - start_time:.2f}s")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = f"data/models/{args.model_name}_large_scale_best.pth"
            os.makedirs("data/models", exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to directory with .npz shards")
    parser.add_argument("--num_pieces", type=int, default=5, help="Number of pieces in the endgame")
    parser.add_argument("--batch_size", type=int, default=16384, help="Large batch size for GPU utilization")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--model", type=str, default="mlp")
    parser.add_argument("--model_name", type=str, default="krpvkp_v5")
    parser.add_argument("--load_path", type=str, default=None, help="Load weights from checkpoint")
    args = parser.parse_args()
    
    train_large_scale(args)
