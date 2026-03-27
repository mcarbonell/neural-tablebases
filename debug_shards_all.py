import numpy as np
import os

for i in [1, 5, 10, 20]:
    shard_path = f"data/lichess_v8_shards/lichess_v8_{i:03d}.npz"
    if os.path.exists(shard_path):
        data = np.load(shard_path)
        wdl = data['wdl']
        unique, counts = np.unique(wdl, return_counts=True)
        dist = dict(zip(unique, counts))
        print(f"Shard {i} distribution: {dist}")
    else:
        print(f"Shard {i} not found.")
