import numpy as np
import os

def check(path):
    if not os.path.exists(path):
        print(f"File {path} does not exist.")
        return
    print(f"Checking {path} (Size: {os.path.getsize(path)/(1024*1024):.2f} MB)")
    data = np.load(path)
    for key in data.keys():
        shape = data[key].shape
        print(f"  {key}: {shape}, dtype: {data[key].dtype}")
        if key == 'x':
            features = shape[1]
            if features == 68: print("    -> Matches V4 (4 pieces, e.g. KPvKP)")
            elif features == 95: print("    -> Matches V4 (5 pieces, e.g. KRPvKP)")
            elif features == 65: print("    -> Matches V1/2 (4 pieces)")
            elif features == 91: print("    -> Matches V1/2 (5 pieces)")

check('data/KPvKP_canonical.npz')
check('data/KRPvKP_canonical.npz')
