import os
import re
import subprocess
import time
import sys
import argparse

def get_inventory(syzygy_path, data_path, max_pieces=5):
    # Regex to find configs (3 to max_pieces pieces)
    # Piece types: K, Q, R, B, N, P
    pattern = re.compile(r'^([KQRBNP]{1,5}v[KQRBNP]{1,5})\.rtbw$', re.IGNORECASE)
    
    available_syzygy = set()
    if not os.path.exists(syzygy_path):
        print(f"Error: Syzygy path {syzygy_path} not found.")
        return []
        
    for f in os.listdir(syzygy_path):
        m = pattern.match(f)
        if m:
            config = m.group(1).upper()
            # Count pieces (Kings are included in the string, e.g., KQvK has 3 chars before 'v' total? No, KQ and K)
            sides = config.split('V')
            total_pieces = len(sides[0]) + len(sides[1])
            if total_pieces <= max_pieces:
                available_syzygy.add(config)
            
    existing_v6 = set()
    if os.path.exists(data_path):
        for f in os.listdir(data_path):
            if f.endswith("_canonical.npz"):
                name = f.replace("_canonical.npz", "").upper()
                existing_v6.add(name)
                
    missing = sorted(list(available_syzygy - existing_v6))
    return missing

def run_generator(config, data_path, syzygy_path):
    print(f"\n" + "="*60)
    print(f"STARTING V6 GENERATION FOR: {config}")
    print("="*60)
    
    cmd = [
        sys.executable, "src/generate_datasets_parallel.py",
        "--syzygy", syzygy_path,
        "--data", data_path,
        "--config", config,
        "--relative",
        "--version", "6",
        "--canonical",
        "--canonical-mode", "auto"
    ]
    
    start_time = time.time()
    try:
        # Standard execution
        result = subprocess.run(cmd, check=True)
        duration = time.time() - start_time
        print(f"COMPLETED {config} in {duration:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR generating {config}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate ALL V6 datasets for the night.")
    parser.add_argument("--syzygy", type=str, default="syzygy")
    parser.add_argument("--data", type=str, default="data/v6")
    parser.add_argument("--max_pieces", type=int, default=5, help="Max pieces to include (default 5)")
    args = parser.parse_args()

    os.makedirs(args.data, exist_ok=True)
    
    missing = get_inventory(args.syzygy, args.data, args.max_pieces)
    
    print(f"Found {len(missing)} endgames to generate in {args.data}.")
    
    # Priority: 3-piece, then 4-piece, then 5-piece
    def piece_count(config):
        sides = config.split('V')
        return len(sides[0]) + len(sides[1])
        
    tasks = sorted(missing, key=lambda x: (piece_count(x), x))
    
    print(f"Total tasks: {len(tasks)}")
    
    for i, config in enumerate(tasks):
        print(f"\nTask {i+1}/{len(tasks)} | Endgame: {config} ({piece_count(config)} pieces)")
        success = run_generator(config, args.data, args.syzygy)
        if not success:
            print(f"Skipping {config} due to error.")
            
if __name__ == "__main__":
    main()
