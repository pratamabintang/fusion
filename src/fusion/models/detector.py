import torch
import torch.nn as nn

from fusion.models.backbone import DualStreamCSPDarkNet
from fusion.models.fusion import MultiScaleFusion
from fusion.models.head import Detect
from fusion.models.loss import YOLOLoss
from fusion.models.neck import PANet


class MS2FusionDetector(nn.Module):
    def __init__(
        self,
        nc: int = 1,
        base_channels: int = 32,
        base_depth: int = 1,
        anchors=None,
        ssm_ratio: float = 2.0,
        d_state: int = 4,
        fusion_type: str = "ms2fusion",
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.backbone = DualStreamCSPDarkNet(in_channels=3, base_channels=base_channels, base_depth=base_depth)

        c_p3 = base_channels * 4
        c_p4 = base_channels * 8
        c_p5 = base_channels * 16

        self.fusion = MultiScaleFusion(
            in_channels=[c_p3, c_p4, c_p5],
            ssm_ratio=ssm_ratio,
            d_state=d_state,
            fusion_type=fusion_type,
        )

        self.neck = PANet(in_channels=[c_p3, c_p4, c_p5])
        self.head = Detect(nc=nc, anchors=anchors, ch=[c_p3, c_p4, c_p5])
        self.loss_fn = YOLOLoss(self.head, nc=nc)

    def forward(self, feat_v: torch.Tensor, feat_t: torch.Tensor, targets: torch.Tensor = None):
        out_v, out_t = self.backbone(feat_v, feat_t)
        f_p3, f_p4, f_p5 = self.fusion(out_v, out_t)
        n3, n4, n5 = self.neck([f_p3, f_p4, f_p5])
        preds = self.head([n3, n4, n5])

        if targets is not None:
            return self.loss_fn(preds, targets)

        return preds

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """Load state dict with automatic legacy key translation."""
        new_state_dict = {}
        for k, v in state_dict.items():
            # Translate legacy keys like fuse_p3.* -> fusion.adapter.fuse_p3.*
            if k.startswith("fuse_p3.") or k.startswith("fuse_p4.") or k.startswith("fuse_p5."):
                new_k = f"fusion.adapter.{k}"
                new_state_dict[new_k] = v
            # Translate older multi-scale fusion keys like fusion.fuse_p3.* -> fusion.adapter.fuse_p3.*
            elif k.startswith("fusion.fuse_p3.") or k.startswith("fusion.fuse_p4.") or k.startswith("fusion.fuse_p5."):
                new_k = k.replace("fusion.", "fusion.adapter.", 1)
                new_state_dict[new_k] = v
            else:
                new_state_dict[k] = v
        return super().load_state_dict(new_state_dict, strict=strict)
