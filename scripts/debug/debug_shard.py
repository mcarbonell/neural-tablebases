import numpy as np
import os

shard_path = "data/lichess_v8_shards/lichess_v8_000.npz"
if os.path.exists(shard_path):
    data = np.load(shard_path)
    wdl = data['wdl']
    unique, counts = np.unique(wdl, return_counts=True)
    dist = dict(zip(unique, counts))
    print(f"Distribution of WDL in Shard 0: {dist}")
    
    dtz_eval = data['dtz']
    print(f"Range of Eval: min={np.min(dtz_eval):.4f}, max={np.max(dtz_eval):.4f}, mean={np.mean(dtz_eval):.4f}")
else:
    print(f"File not found: {shard_path}")
