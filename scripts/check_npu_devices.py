import torch
import torch_directml
import onnx
import onnxruntime as ort

print(f"DirectML available: {torch_directml.is_available()}")
device = torch_directml.device()
print(f"Default DirectML device: {device}")

# We can try to see if there are multiple devices
# AMD Ryzen AI usually has the 780M as Device 0 and NPU as Device 1 in some contexts
# or both in one? No, they are separate PCI devices.

print("Check ONNX Runtime providers:")
print(ort.get_available_providers())
