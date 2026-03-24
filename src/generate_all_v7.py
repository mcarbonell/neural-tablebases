import os
import re
import subprocess
import time
import sys
import argparse

def get_inventory(syzygy_path, data_path, max_pieces=4):
    # Regex to find configs (3 to max_pieces pieces)
    pattern = re.compile(r'^([KQRBNP]{1,5}v[KQRBNP]{1,5})\.rtbw$', re.IGNORECASE)
    
    available_syzygy = set()
    if not os.path.exists(syzygy_path):
        print(f"Error: Syzygy path {syzygy_path} not found.")
        return []
        
    for f in os.listdir(syzygy_path):
        m = pattern.match(f)
        if m:
            config = m.group(1).upper()
            sides = config.split('V')
            total_pieces = len(sides[0]) + len(sides[1])
            if total_pieces <= max_pieces:
                available_syzygy.add(config)
            
    existing_v7 = set()
    if os.path.exists(data_path):
        for f in os.listdir(data_path):
            if f.endswith("_canonical.npz"):
                name = f.replace("_canonical.npz", "").upper()
                existing_v7.add(name)
                
    missing = sorted(list(available_syzygy - existing_v7))
    return missing

def run_generator(config, data_path, syzygy_path, workers=4):
    print(f"\n" + "="*60)
    print(f"STARTING V7 GENERATION FOR: {config}")
    print("="*60)
    
    cmd = [
        sys.executable, "src/generate_datasets_parallel.py",
        "--syzygy", syzygy_path,
        "--data", data_path,
        "--config", config,
        "--relative",
        "--version", "7",
        "--canonical",
        "--canonical-mode", "auto",
        "--workers", str(workers)
    ]
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True)
        duration = time.time() - start_time
        print(f"COMPLETED {config} in {duration:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR generating {config}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate ALL V7 datasets (Dynamic Mobility).")
    parser.add_argument("--syzygy", type=str, default="syzygy")
    parser.add_argument("--data", type=str, default="data/v7")
    parser.add_argument("--max_pieces", type=int, default=4, help="Max pieces to include (default 4)")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers for the parallel generator")
    args = parser.parse_args()

    os.makedirs(args.data, exist_ok=True)
    
    missing = get_inventory(args.syzygy, args.data, args.max_pieces)
    
    print(f"Found {len(missing)} endgames to generate in {args.data}.")
    
    def piece_count(config):
        sides = config.split('V')
        return len(sides[0]) + len(sides[1])
        
    tasks = sorted(missing, key=lambda x: (piece_count(x), x))
    
    print(f"Total tasks: {len(tasks)}")
    
    for i, config in enumerate(tasks):
        print(f"\nTask {i+1}/{len(tasks)} | Endgame: {config} ({piece_count(config)} pieces)")
        success = run_generator(config, args.data, args.syzygy, args.workers)
        if not success:
            print(f"Skipping {config} due to error.")
            
if __name__ == "__main__":
    main()
