# 07: Phase 2 Innovation 2 — Illumination & Contrast Adaptive State Modulation (IC-SSM)

**What to build:** An environment-adaptive state space fusion block that calculates global illumination and contrast metrics from input modalities to dynamically weight $\Delta$ and $A$ state transitions, prioritizing thermal cues in dark scenes and optical cues in bright scenes, benchmarked against the Phase 1 baseline and MS-SSM.

**Blocked by:** 05 (Phase 1 Replication Baseline Trainer, Evaluator & Acceptance Gate Verification)

**Status:** ready-for-agent

- [ ] Implement `AdaptiveGatingModule` calculating illumination/contrast scores from $F_V$ and $F_T$.
- [ ] Implement `ICSSMBlock` that modulates state space transition matrices ($\Delta, A$) using adaptive gating scores.
- [ ] Unit tests verify dynamic parameter modulation ranges and gradient flow under varied illumination conditions.
- [ ] Benchmark IC-SSM against the Phase 1 baseline and MS-SSM on LLVIP.
- [ ] Document final ablation study comparing Original MS2Fusion vs MS-SSM vs IC-SSM vs Combined.
