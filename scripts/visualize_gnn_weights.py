import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys

# Add src to path for model import
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.train_v8 import ChessGnnV8

# Standard Tactical Channels (Match with Rust engine)
RELATIONS = [
    "White-Move", "Black-Move", "White-Attack", "Black-Attack",
    "White-Capture", "Black-Capture", "White-Check", "Black-Check",
    "Promotion", "Castling", "En-Passant", "X-Ray",
    "Defense", "Covered", "Protected", "Pinned"
]

def visualize_model(model_path):
    # Load model
    print(f"Loading V8 model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # We need to create a model with the same architecture to load the weights
    # Default is 4 layers, 128 dim
    model = ChessGnnV8(node_dim=128, gnn_layers=4)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # 1. Analyze Relation Weights (Importance of Tactical Channels)
    # Layer 1 is usually the most interpretive
    gnn1 = model.gnn_blocks[0]
    weights = gnn1.rel_weights.data # (16, in_dim, out_dim)
    
    # Calculate magnitude per relation (L2 norm)
    importance = torch.norm(weights, dim=(1, 2)).numpy()
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(RELATIONS)), importance, color='teal', alpha=0.7)
    plt.xticks(range(len(RELATIONS)), RELATIONS, rotation=45, ha='right')
    plt.title("V8 GNN: Tactical Channel Importance (Layer 1)")
    plt.ylabel("Weight Magnitude (L2 Norm)")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Highlight the top channel
    max_idx = np.argmax(importance)
    bars[max_idx].set_color('orange')
    
    plt.tight_layout()
    plt.savefig("data/gnn_relation_importance.png")
    print("Saved relation importance plot to data/gnn_relation_importance.png")
    
    # 2. Piece Embedding Analysis
    embeddings = model.embed.weight.data.numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(embeddings, aspect='auto', cmap='viridis')
    plt.title("V8 Piece Embeddings")
    plt.xlabel("Embedding Dim")
    plt.ylabel("Piece ID (0=Empty, 1=WP, 2=WN, ... 6=WK)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("data/gnn_piece_embeddings.png")
    print("Saved piece embeddings plot to data/gnn_piece_embeddings.png")
    
    # plt.show() # Removed for headless

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="data/v8_vanguard_run_latest.pth")
    args = parser.parse_args()
    
    if os.path.exists(args.model):
        visualize_model(args.model)
    else:
        print(f"Model {args.model} not found.")
