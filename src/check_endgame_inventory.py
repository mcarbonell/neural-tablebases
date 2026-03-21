import os
import re

def check_inventory():
    syzygy_path = "syzygy"
    data_path = "data/v5"
    
    # 1. Get all Syzygy files (3 and 4 pieces)
    # Piece types: K, Q, R, B, N, P
    # Config format in filename: KQRvK.rtbw...
    pattern_34 = re.compile(r'^([KQRBNP]{1,5}v[KQRBNP]{1,5})\.rtbw$', re.IGNORECASE)
    
    available_syzygy = set()
    for f in os.listdir(syzygy_path):
        m = pattern_34.match(f)
        if m:
            available_syzygy.add(m.group(1).upper())
            
    # 2. Get existing datasets in data/v5
    existing_v5 = set()
    if os.path.exists(data_path):
        for f in os.listdir(data_path):
            if f.endswith("_canonical.npz"):
                name = f.replace("_canonical.npz", "").upper()
                existing_v5.add(name)
                
    # 3. Compare
    missing = sorted(list(available_syzygy - existing_v5))
    done = sorted(list(available_syzygy & existing_v5))
    
    # 4. Filter 3 vs 4 pieces
    missing_3 = []
    missing_4 = []
    for x in missing:
        if 'V' not in x: continue
        sides = x.split('V')
        total_pieces = len(sides[0]) + len(sides[1])
        if total_pieces == 3:
            missing_3.append(x)
        elif total_pieces == 4:
            missing_4.append(x)
    
    print("=" * 40)
    print("ENDGAME INVENTORY STATUS (3-4 Pieces)")
    print("=" * 40)
    print(f"Total Syzygy Configs found: {len(available_syzygy)}")
    print(f"Total Completed in V5: {len(done)}")
    print(f"Missing (To-Do): {len(missing)}")
    print("-" * 40)
    print(f"MISSING 3-PIECES ({len(missing_3)}):")
    print(", ".join(missing_3) if missing_3 else "None")
    print("-" * 40)
    print(f"MISSING 4-PIECES ({len(missing_4)}):")
    # Show in chunks to avoid overwhelming terminal
    for i in range(0, len(missing_4), 8):
        print(", ".join(missing_4[i:i+8]))
    
    print("=" * 40)
    print("Command suggestion for batch generation (first 5 missing 4-piece):")
    batch = missing_4[:5]
    for b in batch:
        print(f"python src/generate_datasets_parallel.py --config {b} --relative --version 5 --canonical --canonical-mode auto --data data/v5")

if __name__ == "__main__":
    check_inventory()
