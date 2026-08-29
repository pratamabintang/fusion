import torch
import pytest

from fusion.models.backbone import DualStreamCSPDarkNet, CSPDarkNet

def test_backbone_shapes_and_gradient():
    base_channels = 32
    base_depth = 1
    
    backbone = DualStreamCSPDarkNet(in_channels=3, base_channels=base_channels, base_depth=base_depth)
    
    B, C, H, W = 2, 3, 256, 256
    feat_v = torch.randn(B, C, H, W, requires_grad=True)
    feat_t = torch.randn(B, C, H, W, requires_grad=True)
    
    out_v, out_t = backbone(feat_v, feat_t)
    
    assert len(out_v) == 3
    assert len(out_t) == 3
    
    p3_v, p4_v, p5_v = out_v
    p3_t, p4_t, p5_t = out_t
    
    # Check shapes
    assert p3_v.shape == (B, base_channels * 4, H // 8, W // 8)
    assert p4_v.shape == (B, base_channels * 8, H // 16, W // 16)
    assert p5_v.shape == (B, base_channels * 16, H // 32, W // 32)
    
    # Check gradients
    loss = p3_v.sum() + p4_v.sum() + p5_v.sum() + p3_t.sum() + p4_t.sum() + p5_t.sum()
    loss.backward()
    
    assert feat_v.grad is not None
    assert feat_t.grad is not None
    
    for param in backbone.parameters():
        assert param.grad is not None

from fusion.models.neck import PANet

def test_panet_shapes_and_gradient():
    in_channels = [256, 512, 1024]
    neck = PANet(in_channels=in_channels)
    
    B, H, W = 2, 32, 32
    # mock features corresponding to p3, p4, p5 from fusion blocks
    f_p3 = torch.randn(B, in_channels[0], H, W, requires_grad=True)
    f_p4 = torch.randn(B, in_channels[1], H // 2, W // 2, requires_grad=True)
    f_p5 = torch.randn(B, in_channels[2], H // 4, W // 4, requires_grad=True)
    
    n3, n4, n5 = neck([f_p3, f_p4, f_p5])
    
    assert n3.shape == (B, 256, H, W)
    assert n4.shape == (B, 512, H // 2, W // 2)
    assert n5.shape == (B, 1024, H // 4, W // 4)
    
    loss = n3.sum() + n4.sum() + n5.sum()
    loss.backward()
    
    assert f_p3.grad is not None
    assert f_p4.grad is not None
    assert f_p5.grad is not None

from fusion.models.head import Detect

def test_detect_head_forward():
    nc = 2
    ch = [256, 512, 1024]
    head = Detect(nc=nc, ch=ch)
    
    B, H, W = 2, 32, 32
    f_p3 = torch.randn(B, ch[0], H, W)
    f_p4 = torch.randn(B, ch[1], H // 2, W // 2)
    f_p5 = torch.randn(B, ch[2], H // 4, W // 4)
    
    # Train mode
    head.train()
    preds = head([f_p3, f_p4, f_p5])
    assert len(preds) == 3
    
    na = len(head.anchors[0])
    no = nc + 5
    assert preds[0].shape == (B, na, H, W, no)
    assert preds[1].shape == (B, na, H // 2, W // 2, no)
    assert preds[2].shape == (B, na, H // 4, W // 4, no)
    
    # Eval mode
    head.eval()
    with torch.no_grad():
        decoded, raw = head([f_p3, f_p4, f_p5])
    
    assert len(raw) == 3
    
    total_anchors = na * (H * W + (H // 2) * (W // 2) + (H // 4) * (W // 4))
    assert decoded.shape == (B, total_anchors, no)

from fusion.models.loss import bbox_iou, YOLOLoss

def test_bbox_iou_ciou():
    box1 = torch.tensor([[10.0, 10.0, 5.0, 5.0]])
    box2 = torch.tensor([[10.0, 10.0, 5.0, 5.0]])
    iou = bbox_iou(box1, box2, CIoU=True)
    assert torch.allclose(iou, torch.tensor([1.0]), atol=1e-4)

def test_yolo_loss_computation():
    from fusion.models.head import Detect
    head = Detect(nc=1, ch=[256, 512, 1024])
    loss_fn = YOLOLoss(head, nc=1)
    
    B, H, W = 2, 32, 32
    f_p3 = torch.randn(B, 256, H, W)
    f_p4 = torch.randn(B, 512, H // 2, W // 2)
    f_p5 = torch.randn(B, 1024, H // 4, W // 4)
    preds = head([f_p3, f_p4, f_p5])
    
    targets = torch.tensor([
        [0, 0, 0.5, 0.5, 0.1, 0.1],
        [1, 0, 0.2, 0.3, 0.05, 0.05]
    ])
    
    loss, loss_dict = loss_fn(preds, targets)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert 'loss_box' in loss_dict
    assert 'loss_obj' in loss_dict
    assert 'loss_cls' in loss_dict
    assert 'total_loss' in loss_dict

from fusion.models.detector import MS2FusionDetector

def test_ms2fusion_detector_train_step():
    detector = MS2FusionDetector(nc=1, base_channels=32, base_depth=1)
    detector.train()
    
    B, C, H, W = 2, 3, 128, 128
    feat_v = torch.randn(B, C, H, W)
    feat_t = torch.randn(B, C, H, W)
    
    targets = torch.tensor([
        [0, 0, 0.5, 0.5, 0.2, 0.2],
        [1, 0, 0.3, 0.3, 0.1, 0.1]
    ])
    
    total_loss, loss_dict = detector(feat_v, feat_t, targets=targets)
    
    assert isinstance(total_loss, torch.Tensor)
    total_loss.backward()
    
    for name, param in detector.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"

def test_ms2fusion_detector_eval_step():
    detector = MS2FusionDetector(nc=1, base_channels=32, base_depth=1)
    detector.eval()
    
    B, C, H, W = 2, 3, 128, 128
    feat_v = torch.randn(B, C, H, W)
    feat_t = torch.randn(B, C, H, W)
    
    with torch.no_grad():
        preds, raw = detector(feat_v, feat_t)
    
    assert isinstance(preds, torch.Tensor)
    assert preds.shape[0] == B
    assert preds.shape[2] == 1 + 5 # no = nc + 5
    assert len(raw) == 3
