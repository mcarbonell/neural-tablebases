import torch
import torch.nn as nn
from src.models import get_model_for_endgame
from src.train_large_scale import StreamingShardedDataset
from torch.utils.data import DataLoader
import os
import glob
import time

def validate_model(model_name, data_dir, device='cpu'):
    # Determine input size from checkpoint
    model_path = f"data/models/{model_name}_best.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    state_dict = torch.load(model_path, map_location=device)
    # The first layer weight in our MLP backbone has shape [hidden_size, input_size]
    # Check for both possible key patterns
    if 'backbone.0.weight' in state_dict:
        input_size = state_dict['backbone.0.weight'].shape[1]
    elif 'layers.0.weight' in state_dict:
        input_size = state_dict['layers.0.weight'].shape[1]
    else:
        # Fallback to scanning for any weight with 2D shape if others fail
        input_size = next(v.shape[1] for k, v in state_dict.items() if len(v.shape) == 2)
    
    print(f"Detected input size from checkpoint: {input_size}")

    # Initialize model (using num_pieces=4 as proxy for architecture)
    model = get_model_for_endgame(num_pieces=4, model_type='mlp', input_size=input_size)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load data
    full_dataset = StreamingShardedDataset(data_dir)
    
    # Validation split is the last 10% (matching train_large_scale.py)
    val_size = int(0.1 * len(full_dataset))
    val_indices = range(len(full_dataset) - val_size, len(full_dataset))
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)
    
    print(f"Validating on {val_size:,} positions (Last 10% of dataset)")
    
    loader = DataLoader(val_subset, batch_size=16384, shuffle=False)

    total = 0
    correct = 0
    total_loss = 0.0
    criterion_wdl = nn.CrossEntropyLoss()
    criterion_dtz = nn.L1Loss()

    start_time = time.time()
    
    with torch.no_grad():
        for i, (x, y_target, y_dtz) in enumerate(loader):
            x, y_target, y_dtz = x.to(device), y_target.to(device), y_dtz.to(device)
            
            # WDL labels: convert -2,0,2 to 0,1,2
            y_wdl = (y_target + 2) // 2
            
            out_wdl, out_dtz = model(x)
            
            # WDL Accuracy
            _, predicted = torch.max(out_wdl, 1)
            total += y_wdl.size(0)
            correct += (predicted == y_wdl).sum().item()
            
            # Loss for tracking
            loss_wdl = criterion_wdl(out_wdl, y_wdl)
            loss_dtz = criterion_dtz(out_dtz, y_dtz)
            total_loss += (loss_wdl.item() + loss_dtz.item())
            
            if i % 50 == 0:
                print(f"Batch {i} | Acc: {100*correct/total:.2f}% | Current Loss: {total_loss/(i+1):.4f}")
            
            # If we want it to be REALLY fast, stop after 500 batches (~8M positions)
            if i >= 200: 
                break

    print(f"\nFINAL VALIDATION RESULTS:")
    print(f"Accuracy: {100*correct/total:.4f}%")
    print(f"Avg Loss: {total_loss/(i+1):.4f}")
    print(f"Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mlp_universal_v6_tactical_large_scale")
    parser.add_argument("--data", type=str, default="data/shards_v6")
    parser.add_argument("--device", type=str, default="cpu") # Use cpu for high stability in validation
    args = parser.parse_args()
    
    # Try to use DirectML if available and not explicitly cpu
    if args.device != 'cpu':
        try:
            import torch_directml
            args.device = torch_directml.device()
            print("Using DirectML for validation")
        except:
            print("DirectML not found, using CPU")
            args.device = 'cpu'

    validate_model(args.model, args.data, args.device)
