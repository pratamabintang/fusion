# 02: MS2Fusion Parametric Interaction Modules (CP-SSM, SP-SSM, FF-SSM)

**What to build:** The core MS2Fusion parametric interaction modules (CP-SSM for complementary features, SP-SSM for shared features, and bidirectional FF-SSM for feature fusion) including local 3×3 depthwise convolution pre-filtering and cross-modal channel squeeze-and-excitation gating.

**Blocked by:** 01 (Dual-Backend Selective Scan Operator & Mathematical Verification)

**Status:** ready-for-agent

- [ ] Implement `CPSSM` with projection matrix exchange ($C_V \leftrightarrow C_T$) and 3×3 depthwise Conv2d pre-filtering.
- [ ] Implement `SPSSM` with parameter sharing constraints ($\Delta_s, B_s, C_s$) generated from linearly projected joint feature embeddings ($F_V \oplus F_T$).
- [ ] Implement `FFSSM` with bidirectional sequence scanning ($[F_1, F_2]$ and $[F_2, F_1]$), cross-modal channel squeeze-and-excitation gating, and output projection.
- [ ] Implement the hierarchical `MS2FusionBlock` integrating CP-SSM, SP-SSM, and FF-SSM.
- [ ] Unit tests verify output shape matching $(B, C, H, W)$ for arbitrary batch sizes and spatial dimensions.
- [ ] Unit tests verify gradient backpropagation flows through all parameter pathways without numerical instability.
