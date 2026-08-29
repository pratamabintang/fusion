# 4. CLI Experiment Runner Suite and NVIDIA RTX 5090 Blackwell CUDA Packaging

Date: 2026-08-29

## Status

Accepted

## Context

With both the Phase 1 replication baseline and Phase 2 modular innovations (MS-SSM, IC-SSM, Combined) implemented and verified in the Python package, researchers and engineers require:
1. **High-Level CLI Tools**: A unified command-line interface to train models, evaluate checkpoints, run visual side-by-side inference demos, and conduct automated multi-model ablation benchmarks.
2. **Independent Model Execution**: Clean CLI switches allowing the pure baseline (`ms2fusion`) and each innovative variant (`ms_ssm`, `ic_ssm`, `combined`) to run, train, and evaluate in complete isolation.
3. **NVIDIA Blackwell (RTX 5090) CUDA Packaging**: Production packaging of the selective scan CUDA/C++ kernel targeting compute capability `sm_100` / `sm_120` via `setup.py` while preserving automatic fallback to the pure PyTorch reference operator on CPU development machines.

## Decision

1. **CLI Experiment Suite (`scripts/`)**:
   - `scripts/train.py`: Multi-epoch training CLI with YAML config support, CLI parameter overrides, mixed precision (BF16), and checkpoint saving to `runs/train/<exp_name>/weights/`.
   - `scripts/eval.py`: Validation evaluator measuring mAP@0.5 and mAP@0.5:0.95 from saved weights or on-the-fly models.
   - `scripts/demo.py`: Side-by-side visualization tool rendering detections on visible ($F_V$) and thermal ($F_T$) modalities.
   - `scripts/benchmark.py`: Automated comparative ablation runner generating standardized Markdown comparison reports comparing accuracy, parameters, model size, and inference latency across all four fusion architectures.
2. **YAML Configuration Repository (`configs/`)**:
   - Store reproducible configs: `configs/ms2fusion_baseline.yaml`, `configs/ms_ssm.yaml`, `configs/ic_ssm.yaml`, `configs/combined.yaml`, and `configs/default.yaml`.
3. **Blackwell CUDA C++ Extension (`src/fusion/csrc/` & `setup.py`)**:
   - Colocate C++/CUDA source files under `src/fusion/csrc/` (`selective_scan.cpp`, `selective_scan_cuda.cu`).
   - Configure `setup.py` using `torch.utils.cpp_extension.CUDAExtension` with compute capabilities `sm_100` (Blackwell datacenter) and `sm_120` (Blackwell GeForce RTX 5090), with graceful CPU-only fallback when CUDA is absent.

## Consequences

- Full standalone capability: Any model variant can be trained or benchmarked with a single CLI command (e.g., `python scripts/train.py --fusion-type ms_ssm`).
- Automatic ablation tables provide publication-ready Markdown reports.
- Seamless deployment on both CPU test environments and next-gen RTX 5090 Blackwell hardware.
