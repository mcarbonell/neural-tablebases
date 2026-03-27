import os
import sys
import subprocess
import time

# Define the V8 Universe: 3-4 pieces + common 5-piece endgames
V8_UNIVERSE = [
    # 3-Piece
    "KQvK", "KRvK", "KBvK", "KNvK", "KPvK",
    
    # 4-Piece (vs King)
    "KQQvK", "KQRvK", "KQBvK", "KQNvK", "KQPvK",
    "KRRvK", "KRBvK", "KRNvK", "KRPvK",
    "KBBvK", "KBNvK", "KBPvK",
    "KNNvK", "KNPvK", "KPPvK",
    
    # 4-Piece (Multi-piece / Balanced)
    "KQvKQ", "KQvKR", "KQvKB", "KQvKN", "KQvKP",
    "KRvKR", "KRvKB", "KRvKN", "KRvKP",
    "KBvKB", "KBvKN", "KBvKP",
    "KNvKN", "KNvKP",
    "KPvKP",
    
    # 5-Piece
    "KRPvKP", "KBPvKP", "KNPvKP", "KPPvKP",
    "KBBvKB", "KBNvKB", "KBNvKP", "KBPvKB",
    "KRvKRP", "KQvKRP", "KRRvKR", "KQQvKQ",
    "KBBvKR", "KBNvKR"
]

def main():
    syzygy_path = "syzygy"
    output_base = "data/v8"
    shard_size = 4_000_000
    
    if not os.path.exists(syzygy_path):
        print(f"ERROR: Syzygy directory '{syzygy_path}' not found!")
        sys.exit(1)
        
    print("="*60, flush=True)
    print("V8 GNN UNIVERSAL DATASET GENERATOR - INDUSTRIAL SCALE", flush=True)
    print("="*60, flush=True)
    print(f"Targeting {len(V8_UNIVERSE)} endgame configurations.", flush=True)
    print(f"Destination: {output_base}", flush=True)
    print(f"Shard size: {shard_size:,} positions", flush=True)
    print("="*60, flush=True)
    
    os.makedirs(output_base, exist_ok=True)
    
    config_list = ",".join(V8_UNIVERSE)
    
    cmd = [
        sys.executable, "src/generate_gnn_dataset.py",
        "--syzygy", syzygy_path,
        "--output", output_base,
        "--configs", config_list,
        "--shard_size", "1000000",
        "--workers", "4",
        "--limit", "0",
        "--canonical"
    ]
    
    print("\n[+] Launching Universal Generator...", flush=True)
    sys.stdout.flush()
    
    try:
        # Simple blocking call that prints to stdout/stderr
        retcode = subprocess.call(cmd)
        
        if retcode == 0:
            print("\n" + "="*60, flush=True)
            print("UNIVERSAL V8 GENERATION COMPLETED SUCCESSFULLY", flush=True)
            print("="*60, flush=True)
        else:
            print(f"\nERROR: Generator exited with code {retcode}", flush=True)
            sys.exit(retcode)
            
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
