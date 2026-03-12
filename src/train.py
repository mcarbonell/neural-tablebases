import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import argparse
import time
from datetime import datetime
from models import MLP, SIREN, get_model_for_endgame
from collections import Counter

class TablebaseDataset(Dataset):
    def __init__(self, data_path):
        print(f"Loading dataset from {data_path}...")
        data = np.load(data_path)
        self.x = torch.from_numpy(data['x']).float()
        
        # Map WDL to 3 classes: {-2 -> 0, 0 -> 1, 2 -> 2}
        wdl_raw = data['wdl']
        wdl_mapped = np.zeros_like(wdl_raw)
        wdl_mapped[wdl_raw == -2] = 0  # Loss
        wdl_mapped[wdl_raw == 0] = 1   # Draw
        wdl_mapped[wdl_raw == 2] = 2   # Win
        
        self.wdl = torch.from_numpy(wdl_mapped).long()
        self.dtz = torch.from_numpy(data['dtz']).float()
        
        # Store input size for model initialization
        self.input_size = self.x.shape[1]
        
        # Detect encoding type
        # Relative encoding: 43 dims for 3 pieces, 65 for 4, 91 for 5
        # Compact encoding: 192 dims for 3 pieces, 256 for 4, 320 for 5
        if self.input_size == 43:
            self.num_pieces = 3
            self.use_relative_encoding = True
        elif self.input_size == 65:
            self.num_pieces = 4
            self.use_relative_encoding = True
        elif self.input_size == 91:
            self.num_pieces = 5
            self.use_relative_encoding = True
        else:
            # Compact encoding: num_pieces * 64
            self.num_pieces = self.input_size // 64
            self.use_relative_encoding = False
        
        # Calculate class weights for balanced training
        unique, counts = np.unique(wdl_mapped, return_counts=True)
        total = len(wdl_mapped)
        self.class_weights = torch.FloatTensor([total / (len(unique) * c) for c in counts])
        
        encoding_type = "relative/geometric" if self.use_relative_encoding else "compact one-hot"
        print(f"Dataset loaded: {len(self.x)} positions.")
        print(f"Input size: {self.input_size} ({self.num_pieces} pieces, {encoding_type} encoding)")
        print(f"WDL distribution: Loss={counts[0]}, Draw={counts[1]}, Win={counts[2]}")
        print(f"Class weights: {self.class_weights.numpy()}")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.wdl[idx], self.dtz[idx]

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", f"train_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    def log(message):
        print(message)
        with open(log_file, "a") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

    log(f"Starting training session: {args.model}")
    log(f"Args: {args}")

    dataset = TablebaseDataset(args.data_path)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Use adaptive model based on endgame complexity (3 classes)
    model = get_model_for_endgame(args.model, dataset.num_pieces, num_wdl_classes=3, 
                                   use_relative_encoding=dataset.use_relative_encoding).to(device)
    log(f"Model architecture: {model}")
    log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Use class weights from dataset for imbalanced classes
    class_weights = dataset.class_weights.to(device)
    log(f"Class weights: {class_weights.cpu().numpy()}")
    
    # CrossEntropy with class weights and reduction='none' for hard example weighting
    criterion_wdl = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
    criterion_dtz = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Scheduler: More conservative ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=20, factor=0.7, min_lr=1e-5
    )

    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # Overfitting Loop: Track hard examples
    hard_examples = []
    hard_example_weight = args.hard_weight  # Weight for hard examples
    hard_examples_count = 0  # Track total hard examples found

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        correct_wdl = 0
        epoch_hard_examples = 0
        
        for batch_idx, (x, wdl, dtz) in enumerate(train_loader):
            x, wdl, dtz = x.to(device), wdl.to(device), dtz.to(device)
            
            optimizer.zero_grad()
            wdl_logits, dtz_pred = model(x)
            
            loss_wdl = criterion_wdl(wdl_logits, wdl)
            
            # Less aggressive Hard Example Mining
            with torch.no_grad():
                predictions = wdl_logits.argmax(1)
                is_wrong = (predictions != wdl).float()
                
                # Reduced dynamic weighting: max 2x instead of 3x
                epoch_factor = min(epoch / 100.0, 2.0)  # Max 2x weight increase, slower ramp
                weights = is_wrong * (hard_example_weight * (1 + epoch_factor)) + 1.0
                
                # Track hard examples less frequently
                if args.hard_mining and batch_idx % 20 == 0:  # Every 20 batches instead of 10
                    wrong_indices = torch.where(is_wrong == 1)[0]
                    epoch_hard_examples += len(wrong_indices)
                    for idx in wrong_indices:
                        hard_examples.append({
                            'x': x[idx].cpu(),
                            'wdl': wdl[idx].cpu(),
                            'dtz': dtz[idx].cpu()
                        })
            
            # Combined loss: WDL + DTZ
            loss = (loss_wdl * weights).mean() + 0.1 * criterion_dtz(dtz_pred.squeeze(), dtz)
            
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            correct_wdl += (wdl_logits.argmax(1) == wdl).sum().item()
        
        hard_examples_count += epoch_hard_examples

        # Validation
        model.eval()
        val_correct = 0
        val_loss = 0
        with torch.no_grad():
            for x, wdl, dtz in val_loader:
                x, wdl, dtz = x.to(device), wdl.to(device), dtz.to(device)
                wdl_logits, dtz_pred = model(x)
                val_correct += (wdl_logits.argmax(1) == wdl).sum().item()
                val_loss += criterion_wdl(wdl_logits, wdl).mean().item()

        val_acc = val_correct / val_size
        val_loss_avg = val_loss / len(val_loader)
        epoch_duration = time.time() - epoch_start_time
        
        log(f"Epoch {epoch+1}/{args.epochs} - Time: {epoch_duration:.2f}s - "
            f"Train Loss: {total_loss/len(train_loader):.4f} - Val Loss: {val_loss_avg:.4f} - "
            f"Train Acc: {correct_wdl/train_size:.4f} - Val Acc: {val_acc:.4f} - "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} - Hard Examples: {epoch_hard_examples}")

        # Save Best Model (based on val_loss for better generalization)
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            save_path = os.path.join("data", f"{args.model}_best.pth")
            torch.save(model.state_dict(), save_path)
            log(f"New best validation loss! Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.4f}. Model saved to {save_path}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            log(f"Early stopping triggered after {patience_counter} epochs without improvement")
            break

        # Periodic checkpoint
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), os.path.join("data", f"{args.model}_checkpoint_e{epoch+1}.pth"))
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Overfitting Loop: Re-train on hard examples every N epochs
        if args.hard_mining and (epoch + 1) % args.hard_mining_freq == 0 and len(hard_examples) > 0:
            log(f"Overfitting Loop: Re-training on {len(hard_examples)} hard examples...")
            model.train()
            
            # Create DataLoader from hard examples
            hard_x = torch.stack([ex['x'] for ex in hard_examples]).to(device)
            hard_wdl = torch.stack([ex['wdl'] for ex in hard_examples]).to(device)
            hard_dtz = torch.stack([ex['dtz'] for ex in hard_examples]).to(device)
            
            # Train on hard examples with same learning rate (not 2x)
            hard_optimizer = optim.Adam(model.parameters(), lr=args.lr)
            for _ in range(args.hard_mining_epochs):
                hard_optimizer.zero_grad()
                wdl_logits, dtz_pred = model(hard_x)
                loss = criterion_wdl(wdl_logits, hard_wdl).mean() + 0.1 * criterion_dtz(dtz_pred.squeeze(), hard_dtz)
                loss.backward()
                hard_optimizer.step()
            
            # Clear hard examples list
            hard_examples = []
            log(f"Overfitting Loop completed")

    # Save Final Model
    save_path = os.path.join("data", f"{args.model}_final.pth")
    torch.save(model.state_dict(), save_path)
    log(f"Final model saved to {save_path}")
    log(f"Total hard examples processed: {hard_examples_count}")
    log(f"Best validation accuracy: {best_val_acc:.4f}")
    log(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["mlp", "siren"], default="mlp")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--hard_weight", type=float, default=2.0,
                        help="Weight multiplier for hard examples (reduced from 5.0)")
    parser.add_argument("--hard_mining", action="store_true", default=True,
                        help="Enable hard example mining")
    parser.add_argument("--hard_mining_freq", type=int, default=50,
                        help="Frequency of hard example re-training (epochs, increased from 10)")
    parser.add_argument("--hard_mining_epochs", type=int, default=3,
                        help="Number of epochs to train on hard examples (reduced from 5)")
    args = parser.parse_args()
    train(args)
