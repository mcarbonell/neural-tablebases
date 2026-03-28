# Master script to generate ALL 3 and 4 piece endgames found in syzygy/
import os
import re
import subprocess
import time
import sys

def get_inventory():
    syzygy_path = "syzygy"
    data_path = "data/v5"
    
    # regex to find configs (3-4 pieces)
    pattern = re.compile(r'^([KQRBNP]{1,5}v[KQRBNP]{1,5})\.rtbw$', re.IGNORECASE)
    
    available_syzygy = set()
    for f in os.listdir(syzygy_path):
        m = pattern.match(f)
        if m:
            available_syzygy.add(m.group(1).upper())
            
    existing_v5 = set()
    if os.path.exists(data_path):
        for f in os.listdir(data_path):
            if f.endswith("_canonical.npz"):
                name = f.replace("_canonical.npz", "").upper()
                existing_v5.add(name)
                
    missing = sorted(list(available_syzygy - existing_v5))
    return missing

def run_generator(config):
    print(f"\n" + "="*60)
    print(f"STARTING GENERATION FOR: {config}")
    print("="*60)
    
    cmd = [
        sys.executable, "src/generate_datasets_parallel.py",
        "--config", config,
        "--relative",
        "--version", "5",
        "--canonical",
        "--canonical-mode", "auto",
        "--data", "data/v5"
    ]
    
    start_time = time.time()
    try:
        # We use a lower process priority if possible? No, let it run full speed.
        result = subprocess.run(cmd, check=True)
        duration = time.time() - start_time
        print(f"COMPLETED {config} in {duration:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR generating {config}: {e}")
        return False

def main():
    missing = get_inventory()
    
    print(f"Found {len(missing)} missing endgames in data/v5.")
    
    # Priority: 3-piece first
    missing_3 = [x for x in missing if len(x.split('V')[0]) + len(x.split('V')[1]) == 3]
    missing_4 = [x for x in missing if len(x.split('V')[0]) + len(x.split('V')[1]) == 4]
    
    tasks = missing_3 + missing_4
    
    print(f"Total tasks: {len(tasks)} ({len(missing_3)} 3-pieces, {len(missing_4)} 4-pieces)")
    
    for i, config in enumerate(tasks):
        print(f"\nTask {i+1}/{len(tasks)}")
        success = run_generator(config)
        if not success:
            print(f"Skipping {config} due to error.")
            
if __name__ == "__main__":
    main()
