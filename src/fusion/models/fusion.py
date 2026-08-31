"""Multi-Scale Feature Fusion Module.

Consolidates multi-scale state-space feature fusion across all variants:
- BaselineMS2FusionAdapter ('ms2fusion'): Exact paper replication across P3, P4, P5
- MultiScaleMemoryAdapter ('ms_ssm'): Multi-scale state-space memory bridge (P3 -> P4 -> P5)
- IlluminationAdaptiveAdapter ('ic_ssm'): Illumination and contrast adaptive state modulation
- CombinedAdapter ('combined'): Multi-scale memory bridge with illumination modulation
"""

from typing import Sequence, Tuple, Union
import torch
import torch.nn as nn

from fusion.models.ms2fusion import MS2FusionBlock
from fusion.models.extensions import MSSSMBlock, ICSSMBlock


class BaselineMS2FusionAdapter(nn.Module):
    """Adapter for baseline independent MS2Fusion blocks across scales."""

    def __init__(self, in_channels: Sequence[int], ssm_ratio: float = 2.0, d_state: int = 4, **kwargs):
        super().__init__()
        c_p3, c_p4, c_p5 = in_channels
        self.fuse_p3 = MS2FusionBlock(c_p3, ssm_ratio=ssm_ratio, d_state=d_state, **kwargs)
        self.fuse_p4 = MS2FusionBlock(c_p4, ssm_ratio=ssm_ratio, d_state=d_state, **kwargs)
        self.fuse_p5 = MS2FusionBlock(c_p5, ssm_ratio=ssm_ratio, d_state=d_state, **kwargs)

    def forward(
        self,
        p3_v: torch.Tensor,
        p3_t: torch.Tensor,
        p4_v: torch.Tensor,
        p4_t: torch.Tensor,
        p5_v: torch.Tensor,
        p5_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f_p3 = self.fuse_p3(p3_v, p3_t)
        f_p4 = self.fuse_p4(p4_v, p4_t)
        f_p5 = self.fuse_p5(p5_v, p5_t)
        return f_p3, f_p4, f_p5


class MultiScaleMemoryAdapter(nn.Module):
    """Adapter for Multi-Scale State-Space Memory Bridge (MS-SSM)."""

    def __init__(self, in_channels: Sequence[int], ssm_ratio: float = 2.0, d_state: int = 4, **kwargs):
        super().__init__()
        c_p3, c_p4, c_p5 = in_channels
        self.fuse_p3 = MSSSMBlock(c_p3, d_model_out=c_p3, d_state=d_state, ssm_ratio=ssm_ratio)
        self.fuse_p4 = MSSSMBlock(c_p4, d_model_out=c_p3, d_state=d_state, ssm_ratio=ssm_ratio)
        self.fuse_p5 = MSSSMBlock(c_p5, d_model_out=c_p4, d_state=d_state, ssm_ratio=ssm_ratio)

    def forward(
        self,
        p3_v: torch.Tensor,
        p3_t: torch.Tensor,
        p4_v: torch.Tensor,
        p4_t: torch.Tensor,
        p5_v: torch.Tensor,
        p5_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f_p3, h_p3 = self.fuse_p3(p3_v, p3_t, memory_state=None)
        f_p4, h_p4 = self.fuse_p4(p4_v, p4_t, memory_state=h_p3)
        f_p5, h_p5 = self.fuse_p5(p5_v, p5_t, memory_state=h_p4)
        return f_p3, f_p4, f_p5


class IlluminationAdaptiveAdapter(nn.Module):
    """Adapter for Illumination & Contrast Adaptive Modulation (IC-SSM)."""

    def __init__(self, in_channels: Sequence[int], ssm_ratio: float = 2.0, d_state: int = 4, **kwargs):
        super().__init__()
        c_p3, c_p4, c_p5 = in_channels
        self.fuse_p3 = ICSSMBlock(c_p3, d_model_out=c_p3, d_state=d_state, ssm_ratio=ssm_ratio)
        self.fuse_p4 = ICSSMBlock(c_p4, d_model_out=c_p4, d_state=d_state, ssm_ratio=ssm_ratio)
        self.fuse_p5 = ICSSMBlock(c_p5, d_model_out=c_p5, d_state=d_state, ssm_ratio=ssm_ratio)

    def forward(
        self,
        p3_v: torch.Tensor,
        p3_t: torch.Tensor,
        p4_v: torch.Tensor,
        p4_t: torch.Tensor,
        p5_v: torch.Tensor,
        p5_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f_p3, _ = self.fuse_p3(p3_v, p3_t)
        f_p4, _ = self.fuse_p4(p4_v, p4_t)
        f_p5, _ = self.fuse_p5(p5_v, p5_t)
        return f_p3, f_p4, f_p5


class CombinedAdapter(nn.Module):
    """Adapter for Combined Model (Multi-Scale Memory + Illumination Modulation)."""

    def __init__(self, in_channels: Sequence[int], ssm_ratio: float = 2.0, d_state: int = 4, **kwargs):
        super().__init__()
        c_p3, c_p4, c_p5 = in_channels
        self.fuse_p3 = ICSSMBlock(c_p3, d_model_out=c_p3, d_state=d_state, ssm_ratio=ssm_ratio)
        self.fuse_p4 = ICSSMBlock(c_p4, d_model_out=c_p3, d_state=d_state, ssm_ratio=ssm_ratio)
        self.fuse_p5 = ICSSMBlock(c_p5, d_model_out=c_p4, d_state=d_state, ssm_ratio=ssm_ratio)

    def forward(
        self,
        p3_v: torch.Tensor,
        p3_t: torch.Tensor,
        p4_v: torch.Tensor,
        p4_t: torch.Tensor,
        p5_v: torch.Tensor,
        p5_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f_p3, h_p3 = self.fuse_p3(p3_v, p3_t, memory_state=None)
        f_p4, h_p4 = self.fuse_p4(p4_v, p4_t, memory_state=h_p3)
        f_p5, h_p5 = self.fuse_p5(p5_v, p5_t, memory_state=h_p4)
        return f_p3, f_p4, f_p5


_FUSION_ADAPTER_REGISTRY = {
    "ms2fusion": BaselineMS2FusionAdapter,
    "ms_ssm": MultiScaleMemoryAdapter,
    "ic_ssm": IlluminationAdaptiveAdapter,
    "combined": CombinedAdapter,
}


class MultiScaleFusion(nn.Module):
    """Deep Multi-Scale Feature Fusion Module.

    Positions a unified fusion interface at the seam between the two-stream
    backbone and the PANet neck, encapsulating multi-scale routing, memory
    bridges, and illumination weighting behind a clean, testable contract.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        ssm_ratio: float = 2.0,
        d_state: int = 4,
        fusion_type: str = "ms2fusion",
        **kwargs,
    ):
        super().__init__()
        self.fusion_type = fusion_type.lower()
        if self.fusion_type not in _FUSION_ADAPTER_REGISTRY:
            raise ValueError(
                f"Unknown fusion_type '{fusion_type}'. Supported variants: {list(_FUSION_ADAPTER_REGISTRY.keys())}"
            )

        adapter_cls = _FUSION_ADAPTER_REGISTRY[self.fusion_type]
        self.adapter = adapter_cls(
            in_channels=in_channels,
            ssm_ratio=ssm_ratio,
            d_state=d_state,
            **kwargs,
        )

    def forward(
        self,
        feats_v: Union[Sequence[torch.Tensor], torch.Tensor],
        feats_t: Union[Sequence[torch.Tensor], torch.Tensor],
        p4_v: torch.Tensor = None,
        p4_t: torch.Tensor = None,
        p5_v: torch.Tensor = None,
        p5_t: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute multi-scale fusion across visible and thermal modalities.

        Accepts either:
        - Two 3-tuples/lists: `forward(feats_v, feats_t)` where `feats_v=(p3_v, p4_v, p5_v)`
        - Six individual tensors: `forward(p3_v, p3_t, p4_v, p4_t, p5_v, p5_t)`
        """
        if isinstance(feats_v, (list, tuple)):
            p3_v, p4_v, p5_v = feats_v
            p3_t, p4_t, p5_t = feats_t
        else:
            p3_v, p3_t = feats_v, feats_t
            if p4_v is None or p4_t is None or p5_v is None or p5_t is None:
                raise ValueError("When passing individual tensors, all 6 tensors (p3_v, p3_t, p4_v, p4_t, p5_v, p5_t) must be provided.")

        return self.adapter(p3_v, p3_t, p4_v, p4_t, p5_v, p5_t)
