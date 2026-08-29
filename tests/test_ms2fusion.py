import torch
import pytest
from fusion.models.ms2fusion import CPSSM

def test_cpssm_shape_and_gradient():
    torch.manual_seed(42)
    B, C, H, W = 2, 32, 16, 16
    feat_v = torch.randn(B, C, H, W, requires_grad=True)
    feat_t = torch.randn(B, C, H, W, requires_grad=True)
    
    model = CPSSM(d_model=C)
    
    out_v, out_t = model(feat_v, feat_t)
    
    assert out_v.shape == (B, C, H, W)
    assert out_t.shape == (B, C, H, W)
    
    loss = out_v.sum() + out_t.sum()
    loss.backward()
    
    assert feat_v.grad is not None
    assert feat_t.grad is not None
    
    for param in model.parameters():
        assert param.grad is not None

def test_cpssm_cross_exchange():
    torch.manual_seed(42)
    B, C, H, W = 1, 32, 8, 8
    feat_v = torch.randn(B, C, H, W)
    feat_t = torch.randn(B, C, H, W)
    
    model = CPSSM(d_model=C)
    
    feat_v.requires_grad = True
    feat_t.requires_grad = True
    
    out_v, out_t = model(feat_v, feat_t)
    
    loss_t = out_t.sum()
    loss_t.backward()
    
    assert feat_v.grad is not None
    assert torch.abs(feat_v.grad).sum() > 0, "feat_v should have gradients from out_t due to cross exchange"
    
    feat_v.grad.zero_()
    feat_t.grad.zero_()
    
    out_v, out_t = model(feat_v, feat_t)
    loss_v = out_v.sum()
    loss_v.backward()
    
    assert feat_t.grad is not None
    assert torch.abs(feat_t.grad).sum() > 0, "feat_t should have gradients from out_v due to cross exchange"

from fusion.models.ms2fusion import SPSSM

def test_spssm_shape_and_gradient():
    torch.manual_seed(42)
    B, C, H, W = 2, 32, 16, 16
    feat_v = torch.randn(B, C, H, W, requires_grad=True)
    feat_t = torch.randn(B, C, H, W, requires_grad=True)
    
    model = SPSSM(d_model=C)
    
    out_v, out_t = model(feat_v, feat_t)
    
    assert out_v.shape == (B, C, H, W)
    assert out_t.shape == (B, C, H, W)
    
    loss = out_v.sum() + out_t.sum()
    loss.backward()
    
    assert feat_v.grad is not None
    assert feat_t.grad is not None
    
    for param in model.parameters():
        assert param.grad is not None

from fusion.models.ms2fusion import FFSSM

def test_ffssm_shape_and_gradient():
    torch.manual_seed(42)
    B, C, H, W = 2, 32, 16, 16
    f1 = torch.randn(B, C, H, W, requires_grad=True)
    f2 = torch.randn(B, C, H, W, requires_grad=True)
    
    model = FFSSM(d_model=C)
    
    out = model(f1, f2)
    
    assert out.shape == (B, C, H, W)
    
    loss = out.sum()
    loss.backward()
    
    assert f1.grad is not None
    assert f2.grad is not None
    
    for param in model.parameters():
        assert param.grad is not None

from fusion.models.ms2fusion import MS2FusionBlock

def test_ms2fusion_block_shape_and_gradient():
    torch.manual_seed(42)
    B, C, H, W = 2, 32, 16, 16
    feat_v = torch.randn(B, C, H, W, requires_grad=True)
    feat_t = torch.randn(B, C, H, W, requires_grad=True)
    
    model = MS2FusionBlock(d_model=C)
    
    out = model(feat_v, feat_t)
    
    assert out.shape == (B, C, H, W)
    
    loss = out.sum()
    loss.backward()
    
    assert feat_v.grad is not None
    assert feat_t.grad is not None
    
    for param in model.parameters():
        assert param.grad is not None

def test_ms2fusion_bfloat16():
    torch.manual_seed(42)
    B, C, H, W = 1, 32, 8, 8
    feat_v = torch.randn(B, C, H, W, dtype=torch.bfloat16)
    feat_t = torch.randn(B, C, H, W, dtype=torch.bfloat16)
    
    model = MS2FusionBlock(d_model=C).bfloat16()
    
    out = model(feat_v, feat_t)
    
    assert out.shape == (B, C, H, W)
    assert out.dtype == torch.bfloat16
