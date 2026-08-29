# 01: Dual-Backend Selective Scan Operator & Mathematical Verification

**What to build:** A hardware-agnostic continuous-to-discrete selective state-space scan operator that runs seamlessly on both CPU (via a pure PyTorch vectorized reference implementation) and GPU (via CUDA C++ kernel dispatch for the RTX 5090), with complete forward and backward gradient verification.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] Implement `selective_scan_ref` in pure PyTorch supporting inputs $u$, $\Delta$, $A$, $B$, $C$, $D$, $\text{delta\_bias}$, and $\text{delta\_softplus}$.
- [ ] Implement automatic backend dispatcher (`selective_scan_fn`) that routes to CUDA C++ kernels when available and falls back to `selective_scan_ref` on CPU.
- [ ] Unit tests verify numerical correctness of state recurrence $h_k = \bar{A} h_{k-1} + \bar{B} x_k, y_k = \bar{C} h_k + D x_k$.
- [ ] Unit tests verify analytical and autograd backward gradients flow to all inputs and parameters without NaN/Inf.
- [ ] Operator supports float32 and bfloat16/float16 input dtypes with internal float32 state accumulation.
