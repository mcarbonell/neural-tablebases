import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import time

# Import local tools
sys.path.append(os.path.dirname(__file__))
from data.dataset_v8 import ChessGNNDataset
from model.models_v8 import get_gnn_model

def train_v8_prototype(data_path: str, epochs: int = 20, batch_size: int = 128):
    print(f"Loading Dataset: {data_path}...")
    dataset = ChessGNNDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Initializing V8 GNN Prototype...")
    model = get_gnn_model(num_layers=3, hidden_dim=64)
    # Check for DirectML or CUDA
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"Training on device: {device}")
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_wdl = nn.CrossEntropyLoss()
    criterion_dtz = nn.MSELoss()
    
    print(f"\nTraining Loop:")
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        total_loss = 0
        correct = 0
        total = 0
        
        model.train()
        for x_ids, adj, y_wdl, y_dtz in dataloader:
            x_ids, adj, y_wdl, y_dtz = x_ids.to(device), adj.to(device), y_wdl.to(device), y_dtz.to(device)
            
            optimizer.zero_grad()
            wdl_logits, dtz_val = model(x_ids, adj)
            
            loss_wdl = criterion_wdl(wdl_logits, y_wdl)
            loss_dtz = criterion_dtz(dtz_val, y_dtz)
            
            loss = loss_wdl + 0.1 * loss_dtz
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(wdl_logits.data, 1)
            total += y_wdl.size(0)
            correct += (predicted == y_wdl).sum().item()
            
        elapsed = time.time() - start_time
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch:02d}/{epochs}] | Loss: {total_loss/len(dataloader):.4f} | Acc: {accuracy:.2f}% | Time: {elapsed:.1f}s")

if __name__ == "__main__":
    smoke_path = "data/smoke/KQvK_gnn_smoke.npz"
    if os.path.exists(smoke_path):
        train_v8_prototype(smoke_path)
    else:
        print(f"Smoke dataset not found at {smoke_path}. Run generate_gnn_dataset.py first.")
