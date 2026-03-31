import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))
from model.models_v8 import ChessGnnV8_Pro

device = torch.device("cpu")
model = ChessGnnV8_Pro(node_dim=128, num_layers=4)

models_to_try = [
    "data/v8_vanguard_run_gpu_best.pth",
    "data/v8_pro_universal_latest.pth",
    "data/v8_universal_35M_latest.pth",
    "data/v8_pro_fast_latest.pth"
]

for m in models_to_try:
    path = os.path.join(os.getcwd(), m)
    if os.path.exists(path):
        print(f"Checking {m}...")
        try:
            model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
            print(f"  SUCCESS! {m} matches ChessGnnV8_Pro.")
        except Exception as e:
            # Print specific mismatch if any
            err_msg = str(e)
            print(f"  FAILED: {err_msg[:200]}...")
    else:
        print(f"File {m} not found.")
