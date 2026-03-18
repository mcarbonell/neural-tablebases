# Analysis: AMD Ryzen AI NPU for Neural Tablebase Inference

**Author:** Antigravity AI Assistant
**Date:** March 18, 2026
**Target Architecture:** AMD Ryzen 7 8845HS (Ryzen AI 2nd Gen)

## 1. Overview of Ryzen AI (NPU)

The Ryzen 7 8845HS features an integrated Neural Processing Unit (NPU) capable of up to **16 TOPS** for AI workloads. While often detected by Windows (appearing in Task Manager at 0% usage), it remains dormant unless specifically targeted via the **AMD Ryzen AI Software** stack.

### Key Hardware Specs
- **NPU PERFORMANCE:** up to 16 TOPS
- **Total AI TOPS (CPU+GPU+NPU):** up to 38 TOPS
- **Instruction Support:** Optimized for INT8 and FP16 operations (minimal support for FP32/training).

---

## 2. Driver Status & Software Compatibility (Early 2026)

The software ecosystem for AMD NPUs has stabilized significantly since late 2024/early 2025, but it remains stricter than GPU-based compute (DirectML/CUDA).

### Windows 11 Compatibility
- **Primary API:** **ONNX Runtime** with **Vitis AI Execution Provider (EP)**.
- **Driver Model:** Requires the Latest **IPU (Intelligent Processing Unit)** driver from AMD's official support.
- **Current Status:** Stable for standard CNNs and small MLPs (like our tablebases). Hybrid modes (G+NPU) are becoming common for larger models but are overkill for 3-5 piece tablebases.

### Known Issues
- **Initialization Delay:** Models can take 1–2 minutes to "compile" the first time they are loaded on the NPU as the driver generates the microcoded executable for the XDNA architecture.
- **Quantization Required:** Running FP32 (standard PyTorch precision) on the NPU is either unsupported or extremely inefficient. For real performance, models must be quantized to **INT8** using AMD's `Olive` or `Quark` tools.
- **Driver Fragmentation:** While 8845HS is well-supported, some older "Ryzen AI" laptops from the 7000 series still face detection issues in modern ONNX builds.

---

## 3. Potential Application in Neural Tablebases

While the **GPU (Radeon 780M)** is currently our best tool for **training** (10x faster than CPU), the **NPU** is the ideal target for **inference/deployment**.

### Benefits
- **Zero Battery Drain:** The NPU is significantly more energy-efficient than the GPU for continuous evaluation.
- **Latency:** For single-position evaluation (like in a search tree), the NPU offers ultra-low latency once the model is warmed up.
- **CPU Offloading:** Frees up the CPU to focus entirely on move generation and search logic.

### Deployment Path (Proposed)
1. **Export:** Convert existing `.pth` (PyTorch) models to `.onnx`.
2. **Quantize:** Use `amd-quark` or `Vitis AI` to create an **INT8** ONNX model.
3. **Runtime:** Integrate `onnxruntime-directml` or `onnxruntime-vitisai` into the final engine.

---

## 4. Final Verdict

At this stage of the project, we should continue using **Radeon 780M (DirectML)** for all development and training tasks. The NPU should be treated as a **deployment-time optimization** target if the goal is to integrate these tablebases into a low-power handheld device or laptop-based chess engine.

---
**Status:** Propositional
**Next Steps:** None currently (GPU training prioritized).
