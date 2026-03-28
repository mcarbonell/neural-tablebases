import onnxruntime as ort
import numpy as np
import argparse
import time
import os

def evaluate_model(onnx_path, npz_path, device_id=0, num_samples=100000):
    print(f"Loading data from {npz_path}...")
    data = np.load(npz_path)
    x = data['x']
    wdl_true_raw = data['wdl']
    dtz_true = data['dtz']
    
    # Shuffle and sample
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    indices = indices[:num_samples]
    
    x = x[indices]
    wdl_true_raw = wdl_true_raw[indices]
    dtz_true = dtz_true[indices]
    
    # Map WDL to indices (0: Loss, 1: Draw, 2: Win)
    wdl_true = np.zeros_like(wdl_true_raw)
    wdl_true[wdl_true_raw == -2] = 0
    wdl_true[wdl_true_raw == 0] = 1
    wdl_true[wdl_true_raw == 2] = 2
    
    print(f"Starting inference with ONNX model on Device {device_id}...")
    sess_options = ort.SessionOptions()
    providers = [
        ('DmlExecutionProvider', {'device_id': device_id}),
        'CPUExecutionProvider'
    ]
    session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
    
    input_name = session.get_inputs()[0].name
    
    # Run inference (in batches)
    batch_size = 8192
    wdl_preds = []
    dtz_preds = []
    
    start_time = time.time()
    for i in range(0, len(x), batch_size):
        batch_x = x[i:i+batch_size]
        outputs = session.run(None, {input_name: batch_x})
        wdl_logits, dtz_p = outputs
        wdl_preds.extend(np.argmax(wdl_logits, axis=1))
        dtz_preds.extend(dtz_p.flatten())
        if (i // batch_size) % 10 == 0:
            print(f"Processed {i}/{len(x)} positions...")
            
    total_time = time.time() - start_time
    wdl_preds = np.array(wdl_preds)
    dtz_preds = np.array(dtz_preds)
    
    # COMPUTE METRICS
    print("\n" + "="*40)
    print(" EVALUATION REPORT: Neural vs Syzygy")
    print("="*40)
    
    # 1. WDL ACCURACY
    acc = np.mean(wdl_preds == wdl_true)
    print(f"WDL Accuracy: {acc*100:.2f}%")
    
    # 2. CONFUSION MATRIX (Manual)
    labels = ['Loss', 'Draw', 'Win']
    cm = np.zeros((3, 3), dtype=int)
    for t in range(3):
        for p in range(3):
            cm[t, p] = np.sum((wdl_true == t) & (wdl_preds == p))
            
    print("\nConfusion Matrix:")
    print("True \\ Pred | Loss  | Draw  | Win   ")
    print("-" * 38)
    for i, row in enumerate(cm):
        print(f"{labels[i]:<10} | {row[0]:5} | {row[1]:5} | {row[2]:5}")
        
    # 3. FATAL ERRORS
    win_as_loss = np.sum((wdl_true == 2) & (wdl_preds == 0))
    loss_as_win = np.sum((wdl_true == 0) & (wdl_preds == 2))
    print(f"\nFATAL ERRORS (Win as Loss): {win_as_loss} ({win_as_loss/len(wdl_true)*100:.4f}%)")
    print(f"FATAL ERRORS (Loss as Win): {loss_as_win} ({loss_as_win/len(wdl_true)*100:.4f}%)")
    
    # 4. DTZ ERROR
    dtz_error = np.abs(dtz_preds - dtz_true)
    dtz_mae = np.mean(dtz_error)
    print(f"\nDTZ Mean Absolute Error: {dtz_mae:.2f} plies")
    
    print(f"\nThroughput: {len(x)/total_time:.0f} positions/sec")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="data/mlp_kpvkp_v4_canonical.onnx")
    parser.add_argument("--data", type=str, default="data/KPvKP_canonical.npz")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--samples", type=int, default=100000)
    args = parser.parse_args()
    
    evaluate_model(args.model, args.data, args.device, args.samples)
