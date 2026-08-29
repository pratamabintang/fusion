# Spec: CLI Experiment Runner Suite and NVIDIA RTX 5090 Blackwell CUDA Packaging

Status: ready-for-agent

## Problem Statement

With the core detection architecture and Phase 2 modular state-space innovations (MS-SSM, IC-SSM) fully implemented and unit-tested in `src/fusion/`, users and researchers require:
1. High-level CLI tools to execute training runs, compute validation metrics, generate visual side-by-side inference outputs, and run multi-architecture ablation benchmarks without writing boilerplate Python code.
2. Independent, switchable execution of the pure baseline (`ms2fusion`) and each innovative variant (`ms_ssm`, `ic_ssm`, `combined`) via both YAML configuration files and CLI flags (`--fusion-type`).
3. Production C++/CUDA packaging in `src/fusion/csrc/` and `setup.py` ready for deployment on next-generation NVIDIA RTX 5090 Blackwell GPUs (`sm_100` / `sm_120`), with robust fallback for CPU development environments.

## Solution

1. **CLI Script Suite (`scripts/`)**:
   - `scripts/train.py`: Unified training entrypoint supporting YAML configs, CLI argument overrides, BF16/FP16 mixed precision, CosineAnnealingLR scheduling, and checkpoint saving to `runs/train/<name>/weights/{best.pt, last.pt}`.
   - `scripts/eval.py`: Standalone evaluation entrypoint computing mAP@0.5 and mAP@0.5:0.95 against the LLVIP test/val split.
   - `scripts/demo.py`: Multispectral inference visualizer generating side-by-side visible + thermal images with annotated detections and confidence scores under `runs/demo/<name>/`.
   - `scripts/benchmark.py`: Automated ablation benchmark runner training or evaluating all 4 fusion variants (`ms2fusion`, `ms_ssm`, `ic_ssm`, `combined`), measuring accuracy, parameter count, model size, and inference latency, and writing a Markdown comparison table to `runs/benchmark/ablation_report.md`.
2. **Reproducible Experiment Configs (`configs/`)**:
   - `configs/default.yaml`: Base training and dataset parameters.
   - `configs/ms2fusion_baseline.yaml`: Pure baseline paper replication settings.
   - `configs/ms_ssm.yaml`: Multi-Scale State-Space Memory Bridge settings.
   - `configs/ic_ssm.yaml`: Illumination & Contrast Adaptive State Modulation settings.
   - `configs/combined.yaml`: Combined memory bridge and illumination modulation settings.
3. **Blackwell CUDA Extension (`src/fusion/csrc/` & `setup.py`)**:
   - C++/CUDA source files: `src/fusion/csrc/selective_scan.cpp`, `src/fusion/csrc/selective_scan_cuda.cu`, and header `src/fusion/csrc/selective_scan.h`.
   - Root `setup.py` building `fusion.selective_scan_cuda_core` with Blackwell architecture compute capability support (`sm_100`, `sm_120`), with graceful CPU fallback.

## User Stories

1. As a researcher, I want to run `python scripts/train.py --config configs/ms2fusion_baseline.yaml --epochs 10` to train the pure baseline model and save weights and training logs.
2. As a researcher exploring innovations, I want to run `python scripts/train.py --fusion-type ms_ssm` or `python scripts/train.py --fusion-type combined` to train innovative variants independently.
3. As an engineer evaluating detection performance, I want to run `python scripts/eval.py --weights runs/train/baseline/weights/best.pt` to compute official mAP@0.5 and mAP@0.5:0.95 metrics on LLVIP.
4. As a practitioner inspecting failure modes, I want to run `python scripts/demo.py --weights weights/best.pt --num-samples 5` to produce annotated side-by-side visible and thermal visual outputs in `runs/demo/`.
5. As a researcher writing a paper, I want to run `python scripts/benchmark.py --data-dir LLVIP/ --quick` to automatically generate a side-by-side Markdown ablation table comparing all 4 fusion architectures on accuracy, parameters, and inference latency.
6. As a DevOps engineer deploying to an NVIDIA RTX 5090 Blackwell machine, I want `pip install -e .` to compile native Blackwell CUDA kernels (`sm_100`, `sm_120`) while allowing CPU developers to run seamlessly via PyTorch fallback.

## Implementation & Testing Decisions

1. **CLI Robustness**: All scripts use `argparse` with structured defaults, support `--help`, validate dataset paths, and gracefully handle missing weights or non-existent files.
2. **Config Loader**: `src/fusion/utils/config.py` provides `load_config(path, overrides)` merging YAML configurations with CLI parameter overrides.
3. **Visualization Utilities**: `src/fusion/utils/visualization.py` provides clean bounding box drawing with OpenCV/PIL and side-by-side canvas stitching.
4. **Automated PyTest Suite**: Unit and integration tests in `tests/test_scripts.py` verifying CLI argument parsing, configuration loading, demo generation, and benchmark report synthesis.
