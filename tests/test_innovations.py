import torch
import pytest
from fusion.models.extensions import MSSSMBlock, MultiScaleMemoryFusion, AdaptiveGatingModule, ICSSMBlock
from fusion.models.detector import MS2FusionDetector

def test_msssm_state_propagation():
    block = MSSSMBlock(d_model_in=64, d_model_out=64, d_state=4, ssm_ratio=2.0)
    feat_v = torch.randn(2, 64, 32, 32)
    feat_t = torch.randn(2, 64, 32, 32)
    
    # Forward without memory
    fused, out_mem = block(feat_v, feat_t, None)
    assert fused.shape == (2, 64, 32, 32)
    
    d_inner = int(64 * 2.0)
    assert out_mem.shape == (2, d_inner, 4)
    
    # Forward with memory
    memory = torch.randn(2, d_inner, 4)
    fused2, out_mem2 = block(feat_v, feat_t, memory)
    assert fused2.shape == (2, 64, 32, 32)
    assert out_mem2.shape == (2, d_inner, 4)

def test_adaptive_gating_illumination():
    gating = AdaptiveGatingModule(in_channels=64)
    # create fake bright and dark inputs
    feat_v = torch.randn(2, 64, 32, 32)
    feat_t = torch.randn(2, 64, 32, 32)
    alpha = gating(feat_v, feat_t)
    assert alpha.shape == (2, 1, 1, 1)
    assert (alpha >= 0).all() and (alpha <= 1).all()

def test_ic_ssm_modulation():
    block = ICSSMBlock(d_model_in=64, d_model_out=64, d_state=4, ssm_ratio=2.0)
    feat_v = torch.randn(2, 64, 32, 32, requires_grad=True)
    feat_t = torch.randn(2, 64, 32, 32, requires_grad=True)
    
    fused, out_mem = block(feat_v, feat_t)
    assert fused.shape == (2, 64, 32, 32)
    
    loss = fused.sum()
    loss.backward()
    
    assert feat_v.grad is not None
    assert feat_t.grad is not None

def test_detector_fusion_modes():
    modes = ['ms2fusion', 'ms_ssm', 'ic_ssm', 'combined']
    feat_v = torch.randn(2, 3, 256, 256)
    feat_t = torch.randn(2, 3, 256, 256)
    for mode in modes:
        detector = MS2FusionDetector(fusion_type=mode, base_channels=16)
        preds = detector(feat_v, feat_t)
        assert len(preds) > 0

def test_comparative_ablation_training_step():
    modes = ['ms2fusion', 'ms_ssm', 'ic_ssm', 'combined']
    feat_v = torch.randn(2, 3, 256, 256)
    feat_t = torch.randn(2, 3, 256, 256)
    targets = torch.tensor([[0, 1, 0.5, 0.5, 0.1, 0.1], [1, 1, 0.5, 0.5, 0.1, 0.1]])
    
    for mode in modes:
        detector = MS2FusionDetector(fusion_type=mode, base_channels=16)
        detector.train()
        loss, _ = detector(feat_v, feat_t, targets)
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()
