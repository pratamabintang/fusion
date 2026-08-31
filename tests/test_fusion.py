import pytest
import torch

from fusion.models.fusion import MultiScaleFusion
from fusion.models.detector import MS2FusionDetector


@pytest.mark.parametrize("fusion_type", ["ms2fusion", "ms_ssm", "ic_ssm", "combined"])
def test_multiscale_fusion_forward_and_backward(fusion_type):
    B = 2
    c_p3, c_p4, c_p5 = 64, 128, 256
    H, W = 32, 32

    fusion = MultiScaleFusion(
        in_channels=[c_p3, c_p4, c_p5],
        ssm_ratio=2.0,
        d_state=4,
        fusion_type=fusion_type,
    )

    p3_v = torch.randn(B, c_p3, H, W, requires_grad=True)
    p4_v = torch.randn(B, c_p4, H // 2, W // 2, requires_grad=True)
    p5_v = torch.randn(B, c_p5, H // 4, W // 4, requires_grad=True)

    p3_t = torch.randn(B, c_p3, H, W, requires_grad=True)
    p4_t = torch.randn(B, c_p4, H // 2, W // 2, requires_grad=True)
    p5_t = torch.randn(B, c_p5, H // 4, W // 4, requires_grad=True)

    feats_v = (p3_v, p4_v, p5_v)
    feats_t = (p3_t, p4_t, p5_t)

    # 1. Forward pass via paired sequence interface
    f_p3, f_p4, f_p5 = fusion(feats_v, feats_t)

    assert f_p3.shape == (B, c_p3, H, W)
    assert f_p4.shape == (B, c_p4, H // 2, W // 2)
    assert f_p5.shape == (B, c_p5, H // 4, W // 4)

    # 2. Backward pass
    loss = f_p3.sum() + f_p4.sum() + f_p5.sum()
    loss.backward()

    for feat in [p3_v, p4_v, p5_v, p3_t, p4_t, p5_t]:
        assert feat.grad is not None
        assert torch.isfinite(feat.grad).all()

    # Parameters that are active in the forward graph must receive valid finite gradients
    active_params = [
        (name, p) for name, p in fusion.named_parameters()
        if not (
            ("fuse_p3.memory_proj" in name) or
            (fusion_type == "ic_ssm" and "memory_proj" in name)
        )
    ]
    for name, param in active_params:
        assert param.grad is not None, f"Active parameter {name} in {fusion_type} has no gradient"


def test_multiscale_fusion_invalid_type():
    with pytest.raises(ValueError, match="Unknown fusion_type"):
        MultiScaleFusion(in_channels=[64, 128, 256], fusion_type="invalid_mode")


def test_detector_with_deep_multiscale_fusion():
    for fusion_type in ["ms2fusion", "ms_ssm", "ic_ssm", "combined"]:
        detector = MS2FusionDetector(
            nc=1,
            base_channels=16,
            base_depth=1,
            fusion_type=fusion_type,
        )
        detector.eval()

        B, C, H, W = 2, 3, 128, 128
        feat_v = torch.randn(B, C, H, W)
        feat_t = torch.randn(B, C, H, W)

        with torch.no_grad():
            preds, raw = detector(feat_v, feat_t)

        assert isinstance(preds, torch.Tensor)
        assert len(raw) == 3


def test_detector_legacy_checkpoint_key_loading():
    # Construct a legacy-style state dict with fuse_p3, fuse_p4, fuse_p5 keys
    detector_old = MS2FusionDetector(nc=1, base_channels=16, base_depth=1, fusion_type="ms2fusion")
    old_state_dict = detector_old.state_dict()

    # Instantiate new detector
    detector_new = MS2FusionDetector(nc=1, base_channels=16, base_depth=1, fusion_type="ms2fusion")
    
    # Load old state dict without errors
    detector_new.load_state_dict(old_state_dict)

    detector_old.eval()
    detector_new.eval()

    B, C, H, W = 2, 3, 128, 128
    feat_v = torch.randn(B, C, H, W)
    feat_t = torch.randn(B, C, H, W)
    with torch.no_grad():
        preds_old, _ = detector_old(feat_v, feat_t)
        preds_new, _ = detector_new(feat_v, feat_t)

    assert torch.allclose(preds_old, preds_new, atol=1e-5)
