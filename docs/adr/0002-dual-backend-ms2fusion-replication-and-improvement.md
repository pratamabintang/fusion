# 2. Dual-Backend MS2Fusion Replication and Modular Improvement Architecture

Date: 2026-08-29

## Status

Accepted

## Context

We are implementing the multispectral state-space feature fusion network (MS2Fusion) from Shen et al. (2026), targeting pedestrian detection on the LLVIP dataset. Development and initial TDD verification occur on a CPU-only environment, while production training and benchmarking will target a dedicated NVIDIA RTX 5090 machine. Furthermore, we intend to benchmark exact paper replication metrics before exploring architectural improvements in feature fusion for object detection.

## Decision

1. **Dual-Backend Selective Scan Engine**:
   - Provide a pure PyTorch vectorized fallback operator (`selective_scan_ref`) that executes without requiring compiled CUDA C++ kernels, enabling fast unit testing and CPU execution.
   - Provide an automatic hardware seam that dynamically loads hardware-accelerated kernels (`selective_scan_cuda_core`) when CUDA-capable GPUs (such as the RTX 5090) are detected.
2. **Two-Phase Delivery Lifecycle**:
   - **Phase 1 (Replication Baseline)**: Implement the exact MS2Fusion modules (CP-SSM with $C$-parameter cross-exchange, SP-SSM with shared parameter constraints, FF-SSM with bidirectional sequence scanning and cross-excitation gating, depthwise 2D convolutions) integrated into a two-stream YOLOv5 (CSPDarkNet53) architecture.
   - **Phase 2 (Architectural Improvement)**: Design, integrate, and benchmark object detection improvements (e.g. cross-stage multi-scale state space propagation, modern backbone/head integration, illumination-aware parameter gating).
3. **Dataset & Test Fixtures**:
   - Build a standard PyTorch LLVIP dataset loader supporting paired RGB and Thermal modalities with single-class pedestrian annotations.
   - Provide synthetic paired-image mock fixtures to enable fast, deterministic TDD test cycles on any development system.

## Consequences

- Full test suite and forward/backward passes can be verified locally on CPU before deployment to RTX 5090.
- Clear separation between baseline replication verification and new architectural enhancements avoids confounding ablation results.
