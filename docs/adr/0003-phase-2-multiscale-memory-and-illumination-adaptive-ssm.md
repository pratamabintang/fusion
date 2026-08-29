# 3. Phase 2 Innovations: Multi-Scale State-Space Memory Bridge and Illumination-Adaptive Modulation

Date: 2026-08-29

## Status

Accepted

## Context

While Phase 1 faithfully replicates the baseline MS2Fusion architecture (Shen et al., 2026) across independent pyramid scales ($P_3, P_4, P_5$), multispectral pedestrian detection on LLVIP presents two domain challenges:
1. **Isolated Multi-Scale States**: Standard feature pyramid fusion treats pyramid levels in isolation, discarding fine-grained spatial boundaries and edge details captured in shallow state-space representations ($P_3$) before reaching deeper semantic layers ($P_5$).
2. **Static Cross-Modal Weighting**: Environmental illumination varies significantly across nighttime and daytime captures. Standard SSM parameterizations fix the transition and gating dynamics regardless of whether the optical modality is degraded by darkness or thermal contrast is reduced.

## Decision

1. **Multi-Scale State-Space Memory Bridge (MS-SSM)**:
   - Implement `MSSSMBlock` and `MultiScaleMemoryFusion` to propagate recurrent state-space hidden memory vectors across pyramid levels ($h_k^{(P3)} \rightarrow h_k^{(P4)} \rightarrow h_k^{(P5)}$).
   - Use cross-scale state projection (`memory_proj`) to adaptively condition higher-level state-space scans on fine-grained spatial context from lower-level scans.
2. **Illumination & Contrast Adaptive State Modulation (IC-SSM)**:
   - Implement `AdaptiveGatingModule` to extract global luminance, variance, and local contrast metrics from visible ($F_V$) and thermal ($F_T$) modalities.
   - Modulate continuous-to-discrete step sizes ($\Delta_V \leftarrow \Delta_V \cdot (0.5 + \alpha)$, $\Delta_T \leftarrow \Delta_T \cdot (1.5 - \alpha)$), transition matrices ($A_V, A_T$), and squeeze-and-excitation channel gating weights based on illumination reliability $\alpha \in [0, 1]$.
3. **Modular Detector Seam**:
   - Equip `MS2FusionDetector` with a configurable `fusion_type` dispatcher (`'ms2fusion'`, `'ms_ssm'`, `'ic_ssm'`, `'combined'`), allowing clean ablation experiments and zero-cost switching between baseline and improved modes.

## Consequences

- Direct propagation of hidden state memory improves detection of small, distant pedestrians at higher pyramid levels without added computational complexity.
- Adaptive parameter modulation dynamically prioritizes thermal signatures in dark scenes and optical textures in well-lit scenes.
- Modular `fusion_type` dispatcher ensures that paper replication baseline remains intact for formal ablation comparisons.
