import os
import sys
import subprocess

# V8 Universe: ONLY 3 and 4 piece endgames (no 5-piece)
V8_3_4_PIECE = [
    # 3-Piece
    "KQvK", "KRvK", "KBvK", "KNvK", "KPvK",
    
    # 4-Piece (Asymmetric: Extra piece vs lone King)
    "KQQvK", "KQRvK", "KQBvK", "KQNvK", "KQPvK",
    "KRRvK", "KRBvK", "KRNvK", "KRPvK",
    "KBBvK", "KBNvK", "KBPvK",
    "KNNvK", "KNPvK", "KPPvK",
    
    # 4-Piece (Symmetric: Piece vs Piece)
    "KQvKQ", "KQvKR", "KQvKB", "KQvKN", "KQvKP",
    "KRvKR", "KRvKB", "KRvKN", "KRvKP",
    "KBvKB", "KBvKN", "KBvKP",
    "KNvKN", "KNvKP",
    "KPvKP",
]

def main():
    syzygy_path = "syzygy"
    output_base = "data/v8_3_4"
    
    if not os.path.exists(syzygy_path):
        print(f"ERROR: Syzygy directory '{syzygy_path}' not found!")
        sys.exit(1)
        
    print("=" * 60, flush=True)
    print("V8 GNN - 3+4 PIECE COMPLETE GENERATOR", flush=True)
    print("=" * 60, flush=True)
    print(f"Targeting {len(V8_3_4_PIECE)} endgame configurations.", flush=True)
    print(f"Destination: {output_base}", flush=True)
    print("=" * 60, flush=True)
    
    os.makedirs(output_base, exist_ok=True)
    
    config_list = ",".join(V8_3_4_PIECE)
    
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
    
    print("\n[+] Launching 3+4 Piece Generator...", flush=True)
    sys.stdout.flush()
    
    try:
        retcode = subprocess.call(cmd)
        
        if retcode == 0:
            print("\n" + "=" * 60, flush=True)
            print("3+4 PIECE GENERATION COMPLETED SUCCESSFULLY", flush=True)
            print("=" * 60, flush=True)
        else:
            print(f"\nERROR: Generator exited with code {retcode}", flush=True)
            sys.exit(retcode)
            
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
