import pytest
import torch
import torch.nn.functional as F
from fusion.ops.selective_scan import selective_scan_ref, selective_scan_fn

def test_selective_scan_shape():
    B, D_dim, L = 2, 4, 16
    N = 8
    
    u = torch.randn(B, D_dim, L)
    delta = torch.randn(B, D_dim, L)
    A = torch.randn(D_dim, N)
    B_tensor = torch.randn(B, N, L)
    C = torch.randn(B, N, L)
    
    out = selective_scan_ref(u, delta, A, B_tensor, C)
    
    assert out.shape == (B, D_dim, L)

def test_selective_scan_numeric():
    B, D_dim, L = 2, 4, 16
    N = 8
    
    u = torch.randn(B, D_dim, L)
    delta = torch.randn(B, D_dim, L)
    A = torch.randn(D_dim, N)
    B_tensor = torch.randn(B, N, L)
    C = torch.randn(B, N, L)
    
    out = selective_scan_ref(u, delta, A, B_tensor, C)
    
    # Naive loop
    x = torch.zeros(B, D_dim, N)
    ys = []
    for i in range(L):
        delta_i = delta[:, :, i] # B, D
        u_i = u[:, :, i] # B, D
        B_i = B_tensor[:, :, i] # B, N
        C_i = C[:, :, i] # B, N
        
        deltaA = torch.exp(torch.einsum('bd,dn->bdn', delta_i, A))
        deltaB_u = torch.einsum('bd,bn,bd->bdn', delta_i, B_i, u_i)
        
        x = deltaA * x + deltaB_u
        y = torch.einsum('bdn,bn->bd', x, C_i)
        ys.append(y)
        
    y_naive = torch.stack(ys, dim=2)
    assert torch.allclose(out, y_naive, atol=1e-5)

def test_selective_scan_delta_bias():
    B, D_dim, L, N = 2, 4, 16, 8
    u = torch.randn(B, D_dim, L)
    delta = torch.randn(B, D_dim, L)
    A = torch.randn(D_dim, N)
    B_tensor = torch.randn(B, N, L)
    C = torch.randn(B, N, L)
    delta_bias = torch.randn(D_dim)
    
    out1 = selective_scan_ref(u, delta, A, B_tensor, C, delta_bias=delta_bias)
    
    delta_plus_bias = delta + delta_bias.unsqueeze(-1)
    out2 = selective_scan_ref(u, delta_plus_bias, A, B_tensor, C)
    
    assert torch.allclose(out1, out2, atol=1e-5)

def test_selective_scan_delta_softplus():
    B, D_dim, L, N = 2, 4, 16, 8
    u = torch.randn(B, D_dim, L)
    delta = torch.randn(B, D_dim, L)
    A = torch.randn(D_dim, N)
    B_tensor = torch.randn(B, N, L)
    C = torch.randn(B, N, L)
    
    out1 = selective_scan_ref(u, delta, A, B_tensor, C, delta_softplus=True)
    
    delta_sp = F.softplus(delta)
    out2 = selective_scan_ref(u, delta_sp, A, B_tensor, C)
    
    assert torch.allclose(out1, out2, atol=1e-5)

def test_selective_scan_D():
    B, D_dim, L, N = 2, 4, 16, 8
    u = torch.randn(B, D_dim, L)
    delta = torch.randn(B, D_dim, L)
    A = torch.randn(D_dim, N)
    B_tensor = torch.randn(B, N, L)
    C = torch.randn(B, N, L)
    D = torch.randn(D_dim)
    
    out1 = selective_scan_ref(u, delta, A, B_tensor, C, D=D)
    out2 = selective_scan_ref(u, delta, A, B_tensor, C)
    
    assert torch.allclose(out1, out2 + u * D.unsqueeze(-1), atol=1e-5)

def test_selective_scan_backward():
    B, D_dim, L, N = 2, 4, 16, 8
    u = torch.randn(B, D_dim, L, requires_grad=True)
    delta = torch.randn(B, D_dim, L, requires_grad=True)
    A = torch.randn(D_dim, N, requires_grad=True)
    B_tensor = torch.randn(B, N, L, requires_grad=True)
    C = torch.randn(B, N, L, requires_grad=True)
    D = torch.randn(D_dim, requires_grad=True)
    delta_bias = torch.randn(D_dim, requires_grad=True)
    
    out = selective_scan_fn(u, delta, A, B_tensor, C, D, delta_bias, delta_softplus=True)
    
    loss = out.sum()
    loss.backward()
    
    for tensor in [u, delta, A, B_tensor, C, D, delta_bias]:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()

def test_selective_scan_bfloat16():
    B, D_dim, L, N = 2, 4, 16, 8
    u = torch.randn(B, D_dim, L, dtype=torch.bfloat16)
    delta = torch.randn(B, D_dim, L, dtype=torch.bfloat16)
    A = torch.randn(D_dim, N) # A is usually float32
    B_tensor = torch.randn(B, N, L, dtype=torch.bfloat16)
    C = torch.randn(B, N, L, dtype=torch.bfloat16)
    D = torch.randn(D_dim)
    delta_bias = torch.randn(D_dim)
    
    out = selective_scan_fn(u, delta, A, B_tensor, C, D, delta_bias, delta_softplus=True)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()

def test_selective_scan_dispatcher_fallback():
    B, D_dim, L, N = 2, 4, 16, 8
    u = torch.randn(B, D_dim, L)
    delta = torch.randn(B, D_dim, L)
    A = torch.randn(D_dim, N)
    B_tensor = torch.randn(B, N, L)
    C = torch.randn(B, N, L)
    
    out_fn = selective_scan_fn(u, delta, A, B_tensor, C)
    out_ref = selective_scan_ref(u, delta, A, B_tensor, C)
    
    assert torch.allclose(out_fn, out_ref, atol=1e-5)

def test_selective_scan_float16():
    B, D_dim, L, N = 2, 4, 8, 4
    # Use small-magnitude inputs to stay within float16 dynamic range (max ~65504)
    u = (torch.randn(B, D_dim, L) * 0.1).to(torch.float16)
    delta = (torch.randn(B, D_dim, L) * 0.1).to(torch.float16)
    A = -torch.rand(D_dim, N)  # negative A keeps state bounded
    B_tensor = (torch.randn(B, N, L) * 0.1).to(torch.float16)
    C = (torch.randn(B, N, L) * 0.1).to(torch.float16)
    D = torch.randn(D_dim) * 0.1
    delta_bias = torch.randn(D_dim) * 0.1

    out = selective_scan_fn(u, delta, A, B_tensor, C, D, delta_bias, delta_softplus=True)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()

def test_selective_scan_gradcheck():
    """Verify analytical gradients via torch.autograd.gradcheck."""
    B, D_dim, L, N = 1, 2, 4, 2
    u = torch.randn(B, D_dim, L, dtype=torch.float64, requires_grad=True)
    delta = torch.randn(B, D_dim, L, dtype=torch.float64, requires_grad=True)
    A = torch.randn(D_dim, N, dtype=torch.float64, requires_grad=True)
    B_tensor = torch.randn(B, N, L, dtype=torch.float64, requires_grad=True)
    C = torch.randn(B, N, L, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        selective_scan_ref,
        (u, delta, A, B_tensor, C),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )
