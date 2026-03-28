import torch
import time

try:
    import torch_directml
    device = torch_directml.device()
    print("Testing on DirectML")
except ImportError:
    device = torch.device("cpu")
    print("Testing on CPU")

# Data
B, N, D = 64, 64, 128
x = torch.randn(B * N, D, device=device)
src_idx = torch.randint(0, B * N, (10000,), device=device)
dst_idx = torch.randint(0, B * N, (10000,), device=device)
vals = torch.randn(10000, D, device=device)

print(f"Testing index_add_ on {device}...")
out1 = torch.zeros(B * N, D, device=device)
try:
    start = time.time()
    for _ in range(100):
        out1.index_add_(0, dst_idx, vals)
    print(f"  index_add_ time: {(time.time() - start)*1000:.2f}ms")
except Exception as e:
    print(f"  index_add_ failed: {e}")

print(f"Testing scatter_add_ on {device}...")
out2 = torch.zeros(B * N, D, device=device)
try:
    # We need to expand dst_idx to match vals shape for scatter_add
    # dst_idx: (E,) -> dst_idx_exp: (E, D)
    dst_idx_exp = dst_idx.unsqueeze(-1).expand(-1, D)
    start = time.time()
    for _ in range(100):
        out2.scatter_add_(0, dst_idx_exp, vals)
    print(f"  scatter_add_ time: {(time.time() - start)*1000:.2f}ms")
except Exception as e:
    print(f"  scatter_add_ failed: {e}")

# Check parity
print(f"Diff: {torch.abs(out1 - out2).max().item()}")
