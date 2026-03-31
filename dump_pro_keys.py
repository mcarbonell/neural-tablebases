import torch
import os

path = "data/v8_pro_universal_latest.pth"
checkpoint = torch.load(path, map_location='cpu', weights_only=False)

print(f"Keys for {path}:")
for k in sorted(checkpoint.keys()):
    print(k)
