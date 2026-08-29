# Spec: Multispectral State-Space Feature Fusion (MS2Fusion) Replication and Architectural Improvement on LLVIP

Status: ready-for-agent

## Problem Statement

Multispectral object detection combines optical (visible/RGB) and infrared (thermal) spectral bands to achieve reliable perception in adverse environmental conditions such as low illumination, fog, and nighttime scenes. However, existing approaches often struggle with the trade-off between capturing long-range cross-modal context and maintaining computational efficiency. While CNNs are computationally lightweight, their local receptive fields limit cross-modal dependency modeling; conversely, Transformers offer global attention but incur quadratic complexity with long sequence lengths. 

The MS2Fusion paper proposes a State Space Model (SSM) based framework utilizing dual-path parametric interaction (CP-SSM for complementary features, SP-SSM for shared representations, and bidirectional FF-SSM for fusion). However, the paper omitted several critical architectural details (such as channel squeeze-and-excitation gating and local depthwise convolution pre-filtering), and Mamba implementations often depend tightly on specific CUDA environments. A researcher developing on a local CPU machine while preparing to deploy to a next-generation NVIDIA RTX 5090 Blackwell machine needs a verified, dual-backend implementation that faithfully replicates the baseline results on the LLVIP pedestrian dataset and provides modular seams to build and benchmark architectural improvements.

## Solution

Build a clean, modern, modular multispectral object detection system in `src/fusion/` featuring:
1. **Dual-Backend Selective Scan Operator**: Seamless automatic execution between a pure PyTorch vectorized CPU reference operator (for local test-driven development) and an optimized CUDA kernel (for high-throughput GPU training on the RTX 5090).
2. **Phase 1 Faithful Baseline Replication**: Exact implementation of the paper's MS2Fusion modules (CP-SSM with projection matrix exchange, SP-SSM with parameter sharing constraints, and FF-SSM with bidirectional sequence scanning and cross-excitation gating) integrated into a two-stream YOLOv5 CSPDarkNet53 detector.
3. **Phase 2 Modular Architectural Enhancements**: Advanced fusion variants including Cross-Scale State Space Memory Bridge (hierarchical state propagation across $P_3 \rightarrow P_4 \rightarrow P_5$) and Illumination & Contrast Adaptive State Modulation.
4. **LLVIP Data Pipeline**: A robust dataset loader and Pascal VOC XML annotation parser supporting both local mock fixtures for fast TDD testing and the standard LLVIP dataset.
5. **Precision & Engineering Rigor**: Mixed precision (BF16/FP16 with FP32 state accumulation) designed for RTX 5090 Blackwell architecture, validated by automated PyTest unit and integration tests.

## User Stories

1. As a computer vision researcher, I want a pure PyTorch CPU implementation of the selective scan operator, so that I can run unit tests, verify backward gradients, and develop models locally without requiring a GPU or custom CUDA compilation.
2. As a researcher deploying to an RTX 5090 machine, I want the selective scan operator to automatically detect CUDA availability and dispatch to high-performance CUDA kernels, so that training and inference achieve maximum hardware utilization.
3. As a developer, I want CP-SSM to exchange the hidden state projection matrix ($C_V$ with $C_T$) across visible and thermal branches, so that cross-modal complementary features are learned adaptively in state space.
4. As a developer, I want SP-SSM to derive shared parameter projections ($\Delta_s, B_s, C_s$) from joint feature embeddings, so that modality-invariant shared structural representations are enforced.
5. As a developer, I want FF-SSM to execute bidirectional forward ($[F_1, F_2]$) and reverse ($[F_2, F_1]$) sequence processing with cross-modal channel squeeze-and-excitation gating, so that feature forgetting is mitigated and multi-modal representations are merged effectively.
6. As a researcher, I want the full MS2Fusion block to incorporate local 3×3 depthwise convolutions with SiLU activations before state-space sequence unfolding, so that local spatial inductive biases are preserved.
7. As an engineer, I want the two-stream CSPDarkNet53 backbone to independently extract multi-scale features ($P_3, P_4, P_5$) for visible and thermal inputs, so that modality-specific feature representations are preserved before fusion.
8. As a developer, I want the fused multi-scale features to flow into a Path Aggregation Network (PANet) neck and YOLO detection head, so that bounding box regression and object classification are computed end-to-end.
9. As a researcher evaluating on LLVIP, I want a dataset loader that parses paired RGB and thermal images with their corresponding Pascal VOC XML annotations into standardized target bounding boxes.
10. As a developer practicing strict TDD, I want synthetic dataset fixtures, so that integration tests can run instantly and deterministically in automated CI/test suites.
11. As a researcher, I want evaluation routines that compute standard mean Average Precision metrics (mAP@0.5 and mAP@0.5:0.95) on the LLVIP validation split, so that performance can be directly compared against the published paper results (97.5% mAP@0.5).
12. As a researcher, I want mixed-precision training support (torch.bfloat16 / torch.float16 with float32 state accumulation), so that training on the RTX 5090 Blackwell architecture achieves high throughput without numerical divergence.
13. As a researcher pursuing novel improvements, I want a modular fusion seam that allows swapping the baseline MS2Fusion block for a Cross-Scale State Space Memory Bridge (MS-SSM), so that cross-level hierarchical context can be propagated between shallow and deep features.
14. As a researcher exploring environmental adaptability, I want an Illumination & Contrast Adaptive State Modulation module (IC-SSM), so that state transition parameters dynamically weight thermal features in low-light and visual features in well-lit conditions.
15. As a developer, I want comprehensive PyTest suites covering operator correctness, module output shapes, loss functions, and end-to-end detector steps, so that regressions are caught immediately.

## Implementation Decisions

### 1. Seams & Architecture Boundaries
- **Primary Model Seam (`MS2FusionDetector`)**:
  Exposes `forward(visible_images, thermal_images, targets=None)` returning either predictions `(boxes, scores, class_ids)` during inference or a dictionary of computed losses `{'loss_box', 'loss_obj', 'loss_cls', 'total_loss'}` during training.
- **Fusion Module Seam (`FusionBlockProtocol`)**:
  All fusion blocks (Original MS2Fusion, Baseline Add, Phase 2 Multi-Scale Memory Bridge, Illumination-Adaptive Gating) implement a unified contract:
  `forward(feat_v: Tensor, feat_t: Tensor) -> Tensor` (or multi-scale dictionary for cross-scale variants).
- **Operator Seam (`SelectiveScanOperator`)**:
  Dispatches to `selective_scan_cuda` if available and on a CUDA device; otherwise falls back to vectorized `selective_scan_ref` in pure PyTorch.
- **Dataset Seam (`LLVIPDatasetProtocol`)**:
  Standard PyTorch `Dataset` producing `(visible_tensor, thermal_tensor, target_boxes, image_meta)`.

### 2. Module Implementations & Interfaces
- **State-Space Operator**: Implements discrete recurrent selective scan $h_k = \bar{A} h_{k-1} + \bar{B} x_k, y_k = \bar{C} h_k + D x_k$ with $\bar{A} = \exp(\Delta A), \bar{B} = (\exp(\Delta A) - I)/\Delta A \cdot \Delta B, \bar{C} = C$.
- **CP-SSM**: Projects input channel dimension $d$ to inner dimension $2d$, applies $3\times3$ depthwise Conv2d + SiLU, flattens spatial tokens along row dimensions into sequence length $L=H \times W$, derives $(\Delta_V, B_V, C_V)$ and $(\Delta_T, B_T, C_T)$, executes selective scan with $C_T$ for the visible branch and $C_V$ for the thermal branch, and folds tokens back to $(B, d, H, W)$.
- **SP-SSM**: Derives shared parameters $(\Delta_s, B_s, C_s)$ from linearly projected $(F_V \oplus F_T)$, executing selective scan on $F_V$ and $F_T$ with shared state transitions and identity residual additions.
- **FF-SSM**: Takes pairs $[F_1, F_2]$ and $[F_2, F_1]$, applies bidirectional selective scanning, computes adaptive channel weights via global average pooling and 2-layer MLP excitation, scales the outputs, concatenates, and projects back to dimension $d$.
- **MS2Fusion Block**: Combines CP-SSM and SP-SSM in parallel, feeds their outputs to single-modality FF-SSM blocks, and fuses the two streams via a final cross-modal FF-SSM block.
- **Dual-Stream Backbone & Head**: Two parallel CSPDarkNet53 stream backbones generating $P_3, P_4, P_5$ feature maps, three MS2Fusion blocks at respective scales, a YOLO PANet neck, and anchor-based detection heads with IoU/GIoU, objectness, and classification loss functions.

### 3. Training & Evaluation Engine
- Mixed precision engine supporting `torch.amp.autocast('cuda', dtype=torch.bfloat16)` or CPU float32 execution.
- LLVIP-specific anchor definitions and single-class (`nc=1`, person) loss weighting.
- Full validation pipeline computing precision-recall curves, AP@50, and AP@50:95.

## Testing Decisions

1. **Unit Testing Behavior (not internals)**:
   - Verify mathematical invariants: CP-SSM output shape equals input shape for arbitrary $(B, C, H, W)$.
   - Verify SP-SSM parameter sharing: gradients propagate to shared linear layers from both modality branches.
   - Verify operator equivalence: verify that `selective_scan_ref` outputs match expected numerical recurrent scan formulations.
   - Verify gradient flow: backward pass on all modules produces non-zero, finite gradients for all trainable parameters without NaN or Inf.
2. **Integration Testing**:
   - Verify end-to-end `MS2FusionDetector` forward and backward step on CPU with a synthetic mini-batch.
   - Verify LLVIP data loading: load sample paired image XML annotations, check coordinate scaling, bounding box normalization, and DataLoader batching.
   - Verify single epoch training step: run an end-to-end train/val cycle on the sample LLVIP dataset.

## Out of Scope

- Multi-camera 3D object detection / LiDAR sensor fusion.
- Video-level temporal tracking across frames.
- Semantic segmentation and salient object detection heads (focused strictly on object detection as requested).
- Direct fine-tuning on unrelated non-pedestrian datasets (e.g. VEDAI aerial or MFNet segmentation).

## Further Notes

- Once Phase 1 replication is fully tested and verified against the acceptance gates, the Phase 2 architectural improvement modules (Cross-Scale State Space Memory Bridge and Illumination Adaptive Gating) will be implemented as modular extensions and benchmarked against the Phase 1 baseline.
