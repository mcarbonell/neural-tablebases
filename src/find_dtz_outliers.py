import numpy as np
import onnxruntime as ort
import argparse
import os

def find_dtz_outliers(npz_path, onnx_path, threshold=1.0, max_items=100):
    print(f"Loading data from {npz_path}...")
    data = np.load(npz_path)
    x = data['x']
    dtz_true = data['dtz']
    
    print(f"Loading ONNX model: {onnx_path}")
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    
    # Process in batches to avoid OOM or slow loop
    batch_size = 8192
    num_items = x.shape[0]
    
    dtz_preds = []
    print(f"Predicting DTZ for {num_items} positions...")
    
    for i in range(0, num_items, batch_size):
        end = min(i + batch_size, num_items)
        inp = x[i:end].astype(np.float32)
        out = sess.run(None, {input_name: inp})
        dtz_preds.append(out[1].flatten())
        if (i // batch_size) % 10 == 0:
            print(f"Progress: {i/num_items*100:.1f}%")
            
    dtz_preds = np.concatenate(dtz_preds)
    
    # Error
    diff = np.abs(dtz_preds - dtz_true)
    outlier_indices = np.where(diff > threshold)[0]
    
    print(f"\nFound {len(outlier_indices)} outliers with error > {threshold}")
    
    # We need FENs? The npz doesn't have FENs.
    # But we can use generate_datasets.py logic to reconstruct board if needed.
    # For now, let's just get the first N outliers' indices.
    
    return outlier_indices[:max_items], x, dtz_true, dtz_preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True)
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=2.0)
    args = parser.parse_args()
    
    indices, x, true, pred = find_dtz_outliers(args.npz, args.onnx, args.threshold)
    
    # Print first few
    for idx in indices[:10]:
        print(f"Index {idx}: True={true[idx]:.2f}, Pred={pred[idx]:.2f}, Diff={abs(true[idx]-pred[idx]):.2f}")
