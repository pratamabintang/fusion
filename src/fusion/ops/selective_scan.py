"""Selective Scan Operator — dual-backend state-space recurrence.

Implements the continuous-to-discrete selective state-space scan:
    h_k = Ā·h_{k-1} + B̄·x_k
    y_k = C·h_k + D·x_k

where Ā = exp(Δ·A), B̄ = Δ·B (simplified ZOH discretization).

CPU path uses a pure PyTorch vectorized reference (``selective_scan_ref``).
GPU path dispatches to the ``selective_scan_cuda_core`` C++ extension when
available, otherwise falls back to the reference implementation.
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expand_groups(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Expand a grouped 4-D tensor ``(B, G, N, L)`` to ``(B, D, N, L)``.

    Each group is repeated ``H = dim // G`` times along axis 1 so that the
    result has ``D`` channels matching the model dimension.
    """
    batch, G, N, L = tensor.shape
    H = dim // G
    return (
        tensor
        .unsqueeze(2)              # (B, G, 1, N, L)
        .expand(-1, -1, H, -1, -1)  # (B, G, H, N, L)
        .reshape(batch, dim, N, L)   # (B, D, N, L)
    )


# ---------------------------------------------------------------------------
# Pure-PyTorch reference operator
# ---------------------------------------------------------------------------

def selective_scan_ref(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = False,
    z: torch.Tensor | None = None,
) -> torch.Tensor:
    """Vectorized CPU reference for the selective scan operator.

    Parameters
    ----------
    u : Tensor (B, D, L)       — input sequence
    delta : Tensor (B, D, L)   — step-size / time-scale parameter
    A : Tensor (D, N)          — state transition matrix (log-space)
    B : Tensor (B, N, L) or (B, G, N, L) — input projection
    C : Tensor (B, N, L) or (B, G, N, L) — output projection
    D : Tensor (D,), optional  — skip-connection coefficient
    delta_bias : Tensor (D,), optional — additive bias on delta
    delta_softplus : bool      — apply softplus to delta after bias
    z : Tensor (B, D, L), optional — skip-connection/gate

    Returns
    -------
    Tensor (B, D, L) in the original input dtype.
    """
    dtype_in = u.dtype
    # Use at least float32 for accumulation; preserve float64 for gradcheck.
    compute_dtype = torch.float64 if u.dtype == torch.float64 else torch.float32
    u = u.to(compute_dtype)
    delta = delta.to(compute_dtype)

    if delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(-1).to(compute_dtype)

    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    A = A.to(compute_dtype)
    B = B.to(compute_dtype)
    C = C.to(compute_dtype)

    x = u.new_zeros((batch, dim, dstate))
    ys = []

    # Ā = exp(Δ·A)
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))

    # B̄·u  (grouped or ungrouped)
    if B.dim() == 3:
        deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
    else:
        B_expanded = _expand_groups(B, dim)
        deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B_expanded, u)

    # Expand grouped C once before the loop
    if C.dim() == 4:
        C_expanded = _expand_groups(C, dim)
    else:
        C_expanded = C

    # Recurrence
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if C.dim() == 3:
            y = torch.einsum('bdn,bn->bd', x, C_expanded[:, :, i])
        else:
            y = torch.einsum('bdn,bdn->bd', x, C_expanded[:, :, :, i])
        ys.append(y)

    y = torch.stack(ys, dim=2)  # (B, D, L)

    if D is not None:
        y = y + u * D.unsqueeze(1)
        
    if z is not None:
        y = y * F.silu(z.to(compute_dtype))

    return y.to(dtype=dtype_in)


# ---------------------------------------------------------------------------
# CUDA kernel dispatcher (torch.autograd.Function)
# ---------------------------------------------------------------------------

class _SelectiveScanCUDA(torch.autograd.Function):
    """Autograd wrapper around the ``selective_scan_cuda_core`` C++ extension.

    This class is only instantiated when the CUDA kernel is both importable
    *and* the input tensors reside on a CUDA device.  Calling it on CPU or
    without the compiled extension is a programming error and raises
    ``RuntimeError``.
    """

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, z=None):
        try:
            import selective_scan_cuda_core as cuda_ext
        except ImportError:
            try:
                import fusion.selective_scan_cuda_core as cuda_ext
            except ImportError as exc:
                raise RuntimeError(
                    "selective_scan_cuda_core is not installed. "
                    "Use selective_scan_ref for CPU execution."
                ) from exc

        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if D is not None:
            D = D.contiguous().float()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if delta_bias is not None:
            delta_bias = delta_bias.contiguous().float()

        if B.dim() == 3:
            B = B.unsqueeze(1)  # (B, 1, N, L)
        if C.dim() == 3:
            C = C.unsqueeze(1)  # (B, 1, N, L)

        out, x, *rest = cuda_ext.selective_scan_fwd_cuda(
            u, delta, A, B, C, D, z, delta_bias, delta_softplus
        )
        ctx.delta_softplus = delta_softplus
        ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x)
        return out

    @staticmethod
    def backward(ctx, dout):
        try:
            import selective_scan_cuda_core as cuda_ext
        except ImportError:
            try:
                import fusion.selective_scan_cuda_core as cuda_ext
            except ImportError as exc:
                raise RuntimeError(
                    "selective_scan_cuda_core is not installed."
                ) from exc

        u, delta, A, B, C, D, z, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, dz, ddelta_bias, *_ = cuda_ext.selective_scan_bwd_cuda(
            u, delta, A, B, C, D, z, delta_bias, dout, x, None,
            ctx.delta_softplus
        )
        return du, ddelta, dA, dB, dC, dD, ddelta_bias, None, dz


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def selective_scan_fn(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = False,
    z: torch.Tensor | None = None,
) -> torch.Tensor:
    """Hardware-aware dispatcher for the selective scan operator.

    Routes to the CUDA C++ kernel when ``selective_scan_cuda_core`` is
    importable **and** the input ``u`` resides on a CUDA device.  Otherwise
    falls back to :func:`selective_scan_ref`.
    """
    use_cuda = False
    if u.is_cuda:
        try:
            import selective_scan_cuda_core  # noqa: F401
            use_cuda = True
        except ImportError:
            try:
                import fusion.selective_scan_cuda_core  # noqa: F401
                use_cuda = True
            except ImportError:
                pass

    if use_cuda:
        return _SelectiveScanCUDA.apply(
            u, delta, A, B, C, D, delta_bias, delta_softplus, z,
        )
    return selective_scan_ref(u, delta, A, B, C, D, delta_bias, delta_softplus, z)
