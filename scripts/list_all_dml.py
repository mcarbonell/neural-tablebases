import torch
import torch_directml
import onnxruntime as ort

print("Listing DirectML Devices (via torch_directml):")
try:
    for i in range(10):
        try:
            device = torch_directml.device(i)
            print(f"Device {i}: {device} (Accessible)")
        except:
            break
except:
    pass

print("\nONNX Runtime Providers:")
print(ort.get_available_providers())

# We can try to force CPU to see if DML was just defaulting to GPU
# Actually, the test script confirmed DmlExecutionProvider for Device 0.
