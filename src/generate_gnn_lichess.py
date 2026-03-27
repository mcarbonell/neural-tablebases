import os
import sys
import json
import zstandard as zstd
import numpy as np
from tqdm import tqdm
import chess
import argparse
from concurrent.futures import ProcessPoolExecutor

# --- Lichess Mapping ---
# Stockfish Centipawns -> Win Probability (approx)
def cp_to_wdl(cp):
    """Normalized evaluation to 0..2 scale (Loss, Draw, Win)"""
    if cp is None: return 1 # Drawish if unknown
    # Normal win is ~150cp, Decisive is ~300cp
    if cp > 150: return 2
    if cp < -150: return 0
    return 1

def cp_to_normalized(cp):
    """Normalized to [-1, 1] using tanh"""
    if cp is None: return 0.0
    return float(np.tanh(cp / 400.0))

def process_batch_lichess(lines, engine_path):
    # Local engine instance per process
    from rust_engine import RustGnnEngine
    engine = RustGnnEngine(engine_path)
    
    p_ids_list = []
    node_tac_list = []
    edges_list = []
    edge_counts_list = []
    wdl_list = []
    eval_list = []
    
    for line in lines:
        try:
            data = json.loads(line)
            fen = data['fen']
            # Get deepest evaluation
            if not data.get('evals') or not data['evals']: continue
            top_eval = data['evals'][-1]
            pvs = top_eval.get('pvs', [])
            if not pvs: continue
            
            best_pv = pvs[0]
            cp = best_pv.get('cp')
            mate = best_pv.get('mate')
            
            # Convert mate to high CP
            if mate is not None:
                cp = 10000 if mate > 0 else -10000
                
            # Extract GNN Features
            p_ids, tactical, edges, count = engine.get_raw_features(fen)
            
            p_ids_list.append(p_ids)
            node_tac_list.append(tactical)
            edges_list.append(edges)
            edge_counts_list.append(count)
            wdl_list.append(cp_to_wdl(cp))
            eval_list.append(cp_to_normalized(cp))
            
        except Exception:
            continue
            
    return {
        'p_ids': p_ids_list,
        'node_tac': node_tac_list,
        'edges': edges_list,
        'edge_counts': edge_counts_list,
        'wdl': wdl_list,
        'eval': eval_list
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to lichess_db_eval.jsonl.zst")
    parser.add_argument("--output_dir", default="data/lichess_gnn")
    parser.add_argument("--shard_size", type=int, default=1000000)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    engine_path = "x88/target/release/x88_engine.dll" # Adjust for OS
    
    dctx = zstd.ZstdDecompressor()
    
    shard_idx = 0
    pos_in_shard = 0
    
    # Buffers for NPZ
    shard_p_ids = []
    shard_node_tac = []
    shard_edges = []
    shard_edge_counts = []
    shard_wdl = []
    shard_eval = []
    
    # Fixed Path
    engine_path = os.path.join("src", "rust_movegen", "target", "release", "rust_gnn_engine.dll")
    
    print(f"Starting Lichess GNN Industry: {args.input}", flush=True)
    
    with open(args.input, 'rb') as fh:
        stream = dctx.stream_reader(fh)
        import io
        text_stream = io.TextIOWrapper(stream, encoding='utf-8')
        
        batch_lines = []
        pbar = tqdm(unit="pos")
        
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for line in text_stream:
                batch_lines.append(line)
                
                if len(batch_lines) >= args.batch_size:
                    results = process_batch_lichess(batch_lines, engine_path)
                    
                    # Accumulate
                    shard_p_ids.extend(results['p_ids'])
                    shard_node_tac.extend(results['node_tac'])
                    shard_edges.extend(results['edges'])
                    shard_edge_counts.extend(results['edge_counts'])
                    shard_wdl.extend(results['wdl'])
                    shard_eval.extend(results['eval'])
                    
                    count = len(results['wdl'])
                    pos_in_shard += int(count)
                    pbar.update(count)
                    
                    if pos_in_shard >= args.shard_size:
                        shard_path = os.path.join(args.output_dir, f"lichess_v8_{shard_idx:03d}.npz")
                        np.savez_compressed(
                            shard_path,
                            p_ids=np.array(shard_p_ids, dtype=np.uint8),
                            node_tac=np.array(shard_node_tac, dtype=np.uint8),
                            edges=np.array(shard_edges, dtype=np.uint16),
                            edge_counts=np.array(shard_edge_counts, dtype=np.uint16),
                            wdl=np.array(shard_wdl, dtype=np.int8),
                            dtz=np.array(shard_eval, dtype=np.float32)
                        )
                        print(f"\nSaved Shard {shard_idx}: {pos_in_shard:,} positions", flush=True)
                        
                        shard_idx += 1
                        pos_in_shard = 0
                        shard_p_ids, shard_node_tac, shard_edges, shard_edge_counts, shard_wdl, shard_eval = [], [], [], [], [], []
                    
                    batch_lines = []
                    
if __name__ == "__main__":
    main()
