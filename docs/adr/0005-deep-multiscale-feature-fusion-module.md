# 5. Deep Multi-Scale Feature Fusion Module

Date: 2026-08-31

## Status

Accepted

## Context

Prior to this decision, `MS2FusionDetector` directly orchestrated four fusion modes (`ms2fusion`, `ms_ssm`, `ic_ssm`, `combined`) inside its own forward pass. Mode-specific block initialization, inter-scale memory recurrence chaining ($h_k^{(P3)} \rightarrow h_k^{(P4)} \rightarrow h_k^{(P5)}$), and illumination parameter routing leaked across the seam into the detector module.

## Decision

1. **Deep Module Seam**:
   - Establish `MultiScaleFusion` in `src/fusion/models/fusion.py` as a deep module positioned at the seam between the two-stream backbone and the PANet neck.
   - The module interface takes paired pyramid features `(feats_v, feats_t)` where each is a 3-tuple `(P3, P4, P5)` and returns fused multi-scale features `(f_p3, f_p4, f_p5)`.
2. **Polymorphic Internal Adapters**:
   - Encapsulate each fusion formulation behind internal adapter classes (`BaselineMS2FusionAdapter`, `MultiScaleMemoryAdapter`, `IlluminationAdaptiveAdapter`, `CombinedAdapter`) satisfying a uniform multi-scale execution contract.
3. **State Dict Backward Compatibility**:
   - Equip `MS2FusionDetector` with an automatic state dict translation layer to load legacy checkpoints seamlessly.

## Consequences

- `MS2FusionDetector` forward pass collapses into a linear pipeline (`backbone` $\rightarrow$ `fusion` $\rightarrow$ `neck` $\rightarrow$ `head`).
- Multi-scale memory chaining and illumination gating are verified and tested directly at the `MultiScaleFusion` interface.
- Exact paper replication (`ms2fusion`) and ablation isolation for modular innovations (`ms_ssm`, `ic_ssm`, `combined`) are preserved.
