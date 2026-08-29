import torch
import torch.nn as nn

from fusion.models.backbone import DualStreamCSPDarkNet
from fusion.models.neck import PANet
from fusion.models.head import Detect
from fusion.models.loss import YOLOLoss

from fusion.models.ms2fusion import MS2FusionBlock

from fusion.models.extensions import MultiScaleMemoryFusion, ICSSMBlock

class MS2FusionDetector(nn.Module):
    def __init__(self, nc=1, base_channels=32, base_depth=1, anchors=None, ssm_ratio=2.0, d_state=4, fusion_type='ms2fusion'):
        super().__init__()
        self.fusion_type = fusion_type
        self.backbone = DualStreamCSPDarkNet(in_channels=3, base_channels=base_channels, base_depth=base_depth)
        
        c_p3 = base_channels * 4
        c_p4 = base_channels * 8
        c_p5 = base_channels * 16
        
        if fusion_type == 'ms2fusion':
            self.fuse_p3 = MS2FusionBlock(c_p3, ssm_ratio=ssm_ratio, d_state=d_state)
            self.fuse_p4 = MS2FusionBlock(c_p4, ssm_ratio=ssm_ratio, d_state=d_state)
            self.fuse_p5 = MS2FusionBlock(c_p5, ssm_ratio=ssm_ratio, d_state=d_state)
        elif fusion_type == 'ms_ssm':
            self.fusion = MultiScaleMemoryFusion(c_p3, c_p4, c_p5, ssm_ratio=ssm_ratio, d_state=d_state)
            # Replace fuse_px with MultiScaleMemoryFusion instances
        elif fusion_type == 'combined':
            # Create a version of MultiScaleMemoryFusion that uses ICSSMBlock
            self.fuse_p3 = ICSSMBlock(c_p3, d_model_out=c_p3, d_state=d_state, ssm_ratio=ssm_ratio)
            self.fuse_p4 = ICSSMBlock(c_p4, d_model_out=c_p3, d_state=d_state, ssm_ratio=ssm_ratio)
            self.fuse_p5 = ICSSMBlock(c_p5, d_model_out=c_p4, d_state=d_state, ssm_ratio=ssm_ratio)
        elif fusion_type == 'ic_ssm':
            self.fuse_p3 = ICSSMBlock(c_p3, d_model_out=c_p3, d_state=d_state, ssm_ratio=ssm_ratio)
            self.fuse_p4 = ICSSMBlock(c_p4, d_model_out=c_p4, d_state=d_state, ssm_ratio=ssm_ratio)
            self.fuse_p5 = ICSSMBlock(c_p5, d_model_out=c_p5, d_state=d_state, ssm_ratio=ssm_ratio)
        
        self.neck = PANet(in_channels=[c_p3, c_p4, c_p5])
        self.head = Detect(nc=nc, anchors=anchors, ch=[c_p3, c_p4, c_p5])
        self.loss_fn = YOLOLoss(self.head, nc=nc)

    def forward(self, feat_v, feat_t, targets=None):
        out_v, out_t = self.backbone(feat_v, feat_t)
        
        p3_v, p4_v, p5_v = out_v
        p3_t, p4_t, p5_t = out_t
        
        if self.fusion_type == 'ms_ssm':
            f_p3, f_p4, f_p5 = self.fusion(p3_v, p3_t, p4_v, p4_t, p5_v, p5_t)
        elif self.fusion_type == 'combined':
            f_p3, h_p3 = self.fuse_p3(p3_v, p3_t, memory_state=None)
            f_p4, h_p4 = self.fuse_p4(p4_v, p4_t, memory_state=h_p3)
            f_p5, h_p5 = self.fuse_p5(p5_v, p5_t, memory_state=h_p4)
        elif self.fusion_type == 'ic_ssm':
            f_p3, _ = self.fuse_p3(p3_v, p3_t)
            f_p4, _ = self.fuse_p4(p4_v, p4_t)
            f_p5, _ = self.fuse_p5(p5_v, p5_t)
        else: # ms2fusion
            f_p3 = self.fuse_p3(p3_v, p3_t)
            f_p4 = self.fuse_p4(p4_v, p4_t)
            f_p5 = self.fuse_p5(p5_v, p5_t)
        
        n3, n4, n5 = self.neck([f_p3, f_p4, f_p5])
        
        preds = self.head([n3, n4, n5])
        
        if targets is not None:
            return self.loss_fn(preds, targets)
        
        return preds
