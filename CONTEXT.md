# CONTEXT: Fusion

This document establishes the ubiquitous language, architectural principles, and core definitions for the Fusion project. All AI agents and human contributors should read and strictly adhere to the terms and constraints documented here.

---

## 1. Domain Vocabulary & Glossary

Use these precise terms consistently across code, tests, documentation, and issues:

- **Fusion**: The overarching project codebase and system.
- **Visible Modality ($F_V$)**: Optical RGB imagery capturing high-resolution color and texture details.
- **Thermal Modality ($F_T$)**: Infrared / thermal imagery capturing heat radiation and thermal signatures.
- **LLVIP Dataset**: A paired visible-infrared dataset for low-light vision and pedestrian detection (15,488 pairs).
- **CP-SSM (Cross-Parametric State Space Model)**: A state-space branch capturing cross-modal complementary features by exchanging output projection matrices ($C_V$ and $C_T$) between modality state spaces.
- **SP-SSM (Shared-Parametric State Space Model)**: A state-space branch learning modality-invariant shared representations by deriving shared parameters ($\Delta_s, B_s, C_s$) from joint feature embeddings.
- **FF-SSM (Feature Fusion State Space Model)**: A bidirectional state-space fusion block that processes forward ($[F_1, F_2]$) and reverse ($[F_2, F_1]$) sequences with channel excitation to mitigate feature forgetting and adaptively combine multi-modal features.
- **MS2Fusion Block**: The hierarchical fusion module combining CP-SSM, SP-SSM, and FF-SSM at multi-scale feature stages ($P_3, P_4, P_5$).
- **Two-Stream Detector Backbone**: Parallel feature extraction networks for RGB and Thermal streams that feed into multi-level fusion blocks prior to the neck and detection head.
- **Selective Scan Operator**: The continuous-to-discrete state-space recurrent/convolutional scan operator (with CUDA-accelerated kernel and pure PyTorch CPU/fallback implementation).
- **Baseline Model (Pure MS2Fusion)**: Exact paper replication using standard MS2Fusion blocks at $P_3, P_4, P_5$ without state memory propagation or illumination modulation (`fusion_type='ms2fusion'`).
- **MS-SSM (Multi-Scale State-Space Memory Bridge)**: Modular extension propagating hidden state memory vectors across feature pyramid levels ($P_3 \rightarrow P_4 \rightarrow P_5$) to preserve fine boundary context (`fusion_type='ms_ssm'`).
- **IC-SSM (Illumination & Contrast Adaptive State Modulation)**: Modular extension dynamically weighting state transitions ($\Delta, A$) and channel gating based on environmental illumination and contrast metrics (`fusion_type='ic_ssm'`).
- **Combined Model**: Integration of both the Multi-Scale Memory Bridge and Illumination-Adaptive Modulation (`fusion_type='combined'`).
- **Experiment Runner**: Unified CLI suite (`scripts/train.py`, `scripts/benchmark.py`, `scripts/demo.py`, `scripts/eval.py`) supporting independent execution, ablation benchmarking, and visualization across all model variants.
- **Seam**: A clean boundary between components where tests or alternate implementations can be attached without reaching into private internals.
- **Deep Module**: A module that exposes a simple interface hiding substantial logic, validation, and domain rules behind it.
- **Tracer Bullet**: An end-to-end slice through all architectural layers that demonstrates viability before fleshing out full functionality.
- **Decision Ticket**: A unit of work designed to resolve architectural ambiguity, answer a question, or produce a specification before code implementation.

---

## 2. Engineering Principles

1. **Disciplined Workflow**: Idea/Task → `/grill-with-docs` → `/to-spec` → `/to-tickets` → `/implement` (with `/tdd` & `/code-review`).
2. **Test-Driven Development**: Work in strict Red-Green-Refactor cycles. Write the failing test asserting desired behavior, verify it fails, make it pass with minimal code, then refactor cleanly.
3. **Keep Context Clean**: Separate architectural design and specification phases from individual ticket implementations.
4. **Preserve Documentation Integrity**: Always update `CONTEXT.md` and record significant decisions as Architecture Decision Records in `docs/adr/`.
