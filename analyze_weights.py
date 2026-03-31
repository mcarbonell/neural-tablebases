import torch
sd = torch.load('data/v8_universal_35M_latest.pth', map_location='cpu', weights_only=False)

w = sd['node_proj.weight']

print("node_proj.weight shape:", w.shape)
print("Chunk variances along dim 1:")
for chunk_size, name in [(32, 'Piece'), (16, 'Coord'), (4, 'Tac')]:
    print(f"If {name} is first ({chunk_size}):", w[:, :chunk_size].var().item())
    
# Let's slice it into the 3 possible sizes: 32, 16, 4. 
# Find out the actual order
vars_32 = []
vars_16 = []
vars_4 = []

# piece could be 0:32, 16:48, 20:52, 4:36, etc.
# Check 0:32
c1 = w[:, 0:32].var().item()
c2 = w[:, 32:48].var().item()
c3 = w[:, 48:52].var().item()
print("Order [32, 16, 4]:", c1, c2, c3)

c1 = w[:, 0:16].var().item()
c2 = w[:, 16:48].var().item()
c3 = w[:, 48:52].var().item()
print("Order [16, 32, 4]:", c1, c2, c3)

c1 = w[:, 0:16].var().item()
c2 = w[:, 16:20].var().item()
c3 = w[:, 20:52].var().item()
print("Order [16, 4, 32]:", c1, c2, c3)

c1 = w[:, 0:4].var().item()
c2 = w[:, 4:20].var().item()
c3 = w[:, 20:52].var().item()
print("Order [4, 16, 32]:", c1, c2, c3)

c1 = w[:, 0:4].var().item()
c2 = w[:, 4:36].var().item()
c3 = w[:, 36:52].var().item()
print("Order [4, 32, 16]:", c1, c2, c3)

