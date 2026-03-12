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

class TablebaseDataset(Dataset):
    def __init__(self, data_path):
        print(f"Loading dataset from {data_path}...")
        data = np.load(data_path)
        self.x = torch.from_numpy(data['x']).float()
        # WDL maps {-2, -1, 0, 1, 2} -> {0, 1, 2, 3, 4} for CrossEntropy
        self.wdl = torch.from_numpy(data['wdl']).long() + 2
        self.dtz = torch.from_numpy(data['dtz']).float()
        
        # Store input size for model initialization
        self.input_size = self.x.shape[1]
        
        # Count pieces from input size (compact encoding: num_pieces * 64)
        self.num_pieces = self.input_size // 64
        
        print(f"Dataset loaded: {len(self.x)} positions.")
        print(f"Input size: {self.input_size} ({self.num_pieces} pieces)")

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

    # Use adaptive model based on endgame complexity
    model = get_model_for_endgame(args.model, dataset.num_pieces, num_wdl_classes=5).to(device)
    log(f"Model architecture: {model}")
    log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # CrossEntropy with reduction='none' for weighting
    criterion_wdl = nn.CrossEntropyLoss(reduction='none')
    criterion_dtz = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Scheduler: ReduceLROnPlateau for adaptive learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.5, min_lr=1e-6)

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    # Overfitting Loop: Track hard examples
    hard_examples = []
    hard_example_weight = args.hard_weight  # Weight for hard examples

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        correct_wdl = 0
        
        for batch_idx, (x, wdl, dtz) in enumerate(train_loader):
            x, wdl, dtz = x.to(device), wdl.to(device), dtz.to(device)
            
            optimizer.zero_grad()
            wdl_logits, dtz_pred = model(x)
            
            loss_wdl = criterion_wdl(wdl_logits, wdl)
            
            # Aggressive Hard Example Mining
            with torch.no_grad():
                predictions = wdl_logits.argmax(1)
                is_wrong = (predictions != wdl).float()
                
                # Dynamic weighting: wrong examples get much higher weight
                # Weight increases as training progresses (more aggressive overfitting)
                epoch_factor = min(epoch / 50.0, 3.0)  # Max 3x weight increase
                weights = is_wrong * (hard_example_weight * (1 + epoch_factor)) + 1.0
                
                # Track hard examples for potential re-sampling
                if args.hard_mining and batch_idx % 10 == 0:
                    wrong_indices = torch.where(is_wrong == 1)[0]
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
            f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            save_path = os.path.join("data", f"{args.model}_best.pth")
            torch.save(model.state_dict(), save_path)
            log(f"New best validation accuracy! Model saved to {save_path}")
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
            
            # Train on hard examples with higher learning rate
            hard_optimizer = optim.Adam(model.parameters(), lr=args.lr * 2)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["mlp", "siren"], default="mlp")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--hard_weight", type=float, default=5.0,
                        help="Weight multiplier for hard examples")
    parser.add_argument("--hard_mining", action="store_true", default=True,
                        help="Enable aggressive hard example mining")
    parser.add_argument("--hard_mining_freq", type=int, default=10,
                        help="Frequency of hard example re-training (epochs)")
    parser.add_argument("--hard_mining_epochs", type=int, default=5,
                        help="Number of epochs to train on hard examples")
    args = parser.parse_args()
    train(args)
