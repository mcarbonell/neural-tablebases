import onnxruntime as ort
import numpy as np
import time
import argparse

def test_inference(onnx_path, device_id=0, num_batches=100, batch_size=16384):
    print(f"Testing inference on DML Device {device_id}...")
    
    sess_options = ort.SessionOptions()
    # Explicitly use DirectML
    providers = [
        ('DmlExecutionProvider', {
            'device_id': device_id,
        }),
        'CPUExecutionProvider',
    ]
    
    try:
        session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
    except Exception as e:
        print(f"Error creating session for Device {device_id}: {e}")
        return
        
    print(f"Session created. Active provider: {session.get_providers()[0]}")
    
    # Get input details
    input_name = session.get_inputs()[0].name
    input_shape = list(session.get_inputs()[0].shape)
    if input_shape[0] == 'batch_size' or input_shape[0] is None:
        input_shape[0] = batch_size
    
    # Generate dummy data
    dummy_data = np.random.randn(*input_shape).astype(np.float32)
    
    print(f"Running {num_batches} batches of size {batch_size}...")
    print("WATCH YOUR TASK MANAGER (NPU vs GPU)!")
    
    # Warmup
    _ = session.run(None, {input_name: dummy_data})
    
    start_time = time.time()
    for i in range(num_batches):
        _ = session.run(None, {input_name: dummy_data})
        if (i+1) % 10 == 0:
            print(f"Batch {i+1}/{num_batches}...")
            
    end_time = time.time()
    avg_batch_time = (end_time - start_time) / num_batches
    total_positions = num_batches * batch_size
    pos_per_sec = total_positions / (end_time - start_time)
    
    print("-" * 30)
    print(f"Device: {device_id}")
    print(f"Total time: {end_time - start_time:.2f}s")
    print(f"Avg batch time: {avg_batch_time*1000:.2f}ms")
    print(f"Performance: {pos_per_sec:.2f} positions/sec")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="DML Device ID (0 or 1)")
    parser.add_argument("--batches", type=int, default=100)
    args = parser.parse_args()
    
    test_inference("data/mlp_kpvkp_v4_canonical.onnx", device_id=args.device, num_batches=args.batches)
