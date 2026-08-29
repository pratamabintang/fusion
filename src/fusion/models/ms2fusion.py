"""MS2Fusion: Multispectral State-Space Feature Fusion Modules.

Implements the parametric interaction modules from Shen et al. (2026):
- CPSSM: Cross-Parametric State Space Model with projection matrix exchange (C_V <-> C_T)
- SPSSM: Shared-Parametric State Space Model with joint feature embedding parameter derivation
- FFSSM: Feature Fusion State Space Model with bidirectional scanning and cross-modal SE gating
- MS2FusionBlock: Hierarchical multi-scale fusion block integrating CP-SSM, SP-SSM, and FF-SSM
"""

import math
from typing import Tuple

import torch
import torch.nn as nn

from fusion.ops.selective_scan import selective_scan_fn


class CPSSM(nn.Module):
    """Cross-Parametric State Space Model (CP-SSM).

    Captures cross-modal complementary features between visible (F_V) and
    thermal (F_T) modalities by exchanging their hidden state output projection
    matrices (C_V <-> C_T) during selective scanning.

    Parameters
    ----------
    d_model : int
        Channel dimension of the input feature maps.
    d_state : int
        State-space hidden dimension (default: 4).
    ssm_ratio : float
        Channel expansion ratio for inner SSM dimension (default: 2.0).
    dt_rank : str or int
        Rank of the delta projection. Defaults to 'auto' (ceil(d_model / 16)).
    d_conv : int
        Kernel size of depthwise convolution pre-filtering (default: 3).
    conv_bias : bool
        Whether to include bias in depthwise conv (default: True).
    dropout : float
        Dropout probability (default: 0.0).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 4,
        ssm_ratio: float = 2.0,
        dt_rank: str = "auto",
        d_conv: int = 3,
        conv_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.ssm_ratio = ssm_ratio
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        
        self.in_proj_v = nn.Linear(d_model, self.d_inner)
        self.in_proj_t = nn.Linear(d_model, self.d_inner)
        
        self.conv_v = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv//2, groups=self.d_inner, bias=conv_bias)
        self.conv_t = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv//2, groups=self.d_inner, bias=conv_bias)
        self.act = nn.SiLU()
        
        self.x_proj_v = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.x_proj_t = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        
        self.dt_proj_v = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.dt_proj_t = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log_v = nn.Parameter(torch.log(A))
        self.A_log_t = nn.Parameter(torch.log(A))
        
        self.D_v = nn.Parameter(torch.ones(self.d_inner))
        self.D_t = nn.Parameter(torch.ones(self.d_inner))
        
        self.norm_v = nn.LayerNorm(self.d_inner)
        self.norm_t = nn.LayerNorm(self.d_inner)
        
        self.out_proj_v = nn.Linear(self.d_inner, d_model)
        self.out_proj_t = nn.Linear(self.d_inner, d_model)
        
        self.dropout_v = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.dropout_t = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, feat_v: torch.Tensor, feat_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = feat_v.shape
        L = H * W
        
        # Projections and convolutions
        # V branch
        x_v = feat_v.permute(0, 2, 3, 1).contiguous() # B, H, W, C
        x_v = self.in_proj_v(x_v).permute(0, 3, 1, 2).contiguous() # B, d_inner, H, W
        x_v = self.act(self.conv_v(x_v))
        x_v_flat = x_v.view(B, self.d_inner, L) # B, d_inner, L
        
        # T branch
        x_t = feat_t.permute(0, 2, 3, 1).contiguous() # B, H, W, C
        x_t = self.in_proj_t(x_t).permute(0, 3, 1, 2).contiguous() # B, d_inner, H, W
        x_t = self.act(self.conv_t(x_t))
        x_t_flat = x_t.view(B, self.d_inner, L) # B, d_inner, L
        
        # x_proj expects B, L, d_inner
        x_v_proj_in = x_v_flat.transpose(1, 2)
        x_t_proj_in = x_t_flat.transpose(1, 2)
        
        v_proj = self.x_proj_v(x_v_proj_in) # B, L, dt_rank + 2 * d_state
        t_proj = self.x_proj_t(x_t_proj_in)
        
        dt_v, B_v, C_v = torch.split(v_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_t, B_t, C_t = torch.split(t_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt_v = self.dt_proj_v(dt_v) # B, L, d_inner
        dt_t = self.dt_proj_t(dt_t)
        
        # selective scan wants dt, B, C shaped B, d_inner, L and B, d_state, L
        dt_v = dt_v.transpose(1, 2).contiguous() # B, d_inner, L
        dt_t = dt_t.transpose(1, 2).contiguous()
        B_v = B_v.transpose(1, 2).contiguous() # B, d_state, L
        B_t = B_t.transpose(1, 2).contiguous()
        C_v = C_v.transpose(1, 2).contiguous()
        C_t = C_t.transpose(1, 2).contiguous()
        
        A_v = -torch.exp(self.A_log_v)
        A_t = -torch.exp(self.A_log_t)
        
        # CROSS PROJECTION EXCHANGE
        # y_V uses C_T, y_T uses C_V
        y_v = selective_scan_fn(x_v_flat, dt_v, A_v, B_v, C_t, self.D_v, delta_bias=self.dt_proj_v.bias, delta_softplus=True)
        y_t = selective_scan_fn(x_t_flat, dt_t, A_t, B_t, C_v, self.D_t, delta_bias=self.dt_proj_t.bias, delta_softplus=True)
        
        # Output projections
        y_v = y_v.transpose(1, 2).contiguous() # B, L, d_inner
        y_t = y_t.transpose(1, 2).contiguous()
        
        y_v = self.norm_v(y_v)
        y_t = self.norm_t(y_t)
        
        out_v = self.out_proj_v(y_v) # B, L, d_model
        out_t = self.out_proj_t(y_t)
        
        out_v = self.dropout_v(out_v).transpose(1, 2).contiguous().view(B, self.d_model, H, W)
        out_t = self.dropout_t(out_t).transpose(1, 2).contiguous().view(B, self.d_model, H, W)
        
        feat_v_cross = feat_v + out_v
        feat_t_cross = feat_t + out_t
        
        return feat_v_cross, feat_t_cross

class SPSSM(nn.Module):
    """Shared-Parametric State Space Model (SP-SSM).

    Learns modality-invariant shared structural representations between visible
    (F_V) and thermal (F_T) modalities by deriving shared state-space transition
    and projection parameters (Delta_s, B_s, C_s) from joint feature embeddings.

    Parameters
    ----------
    d_model : int
        Channel dimension of the input feature maps.
    d_state : int
        State-space hidden dimension (default: 4).
    ssm_ratio : float
        Channel expansion ratio for inner SSM dimension (default: 2.0).
    dt_rank : str or int
        Rank of the delta projection. Defaults to 'auto' (ceil(d_model / 16)).
    d_conv : int
        Kernel size of depthwise convolution pre-filtering (default: 3).
    conv_bias : bool
        Whether to include bias in depthwise conv (default: True).
    dropout : float
        Dropout probability (default: 0.0).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 4,
        ssm_ratio: float = 2.0,
        dt_rank: str = "auto",
        d_conv: int = 3,
        conv_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.ssm_ratio = ssm_ratio
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        
        self.in_proj_share_v = nn.Linear(d_model, self.d_inner)
        self.in_proj_share_t = nn.Linear(d_model, self.d_inner)
        
        self.conv_v = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv//2, groups=self.d_inner, bias=conv_bias)
        self.conv_t = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv//2, groups=self.d_inner, bias=conv_bias)
        self.act = nn.SiLU()
        
        self.x_proj_share = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj_s = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log_s = nn.Parameter(torch.log(A))
        self.D_s = nn.Parameter(torch.ones(self.d_inner))
        
        self.norm_share = nn.LayerNorm(self.d_inner)
        self.out_proj_share = nn.Linear(self.d_inner, d_model)
        
        self.dropout_share = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, feat_v: torch.Tensor, feat_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = feat_v.shape
        L = H * W
        
        # V branch
        x_v = feat_v.permute(0, 2, 3, 1).contiguous()
        x_v = self.in_proj_share_v(x_v).permute(0, 3, 1, 2).contiguous()
        x_v = self.act(self.conv_v(x_v))
        x_share_v = x_v.view(B, self.d_inner, L)
        
        # T branch
        x_t = feat_t.permute(0, 2, 3, 1).contiguous()
        x_t = self.in_proj_share_t(x_t).permute(0, 3, 1, 2).contiguous()
        x_t = self.act(self.conv_t(x_t))
        x_share_t = x_t.view(B, self.d_inner, L)
        
        # Joint feature representation
        x_joint = x_share_v + x_share_t
        x_joint_proj_in = x_joint.transpose(1, 2)
        
        share_proj = self.x_proj_share(x_joint_proj_in) # B, L, dt_rank + 2 * d_state
        dt_s, B_s, C_s = torch.split(share_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt_s = self.dt_proj_s(dt_s)
        dt_s = dt_s.transpose(1, 2).contiguous() # B, d_inner, L
        B_s = B_s.transpose(1, 2).contiguous()
        C_s = C_s.transpose(1, 2).contiguous()
        
        A_s = -torch.exp(self.A_log_s)
        
        y_share_v = selective_scan_fn(x_share_v, dt_s, A_s, B_s, C_s, self.D_s, delta_bias=self.dt_proj_s.bias, delta_softplus=True)
        y_share_t = selective_scan_fn(x_share_t, dt_s, A_s, B_s, C_s, self.D_s, delta_bias=self.dt_proj_s.bias, delta_softplus=True)
        
        y_share_v = y_share_v.transpose(1, 2).contiguous()
        y_share_t = y_share_t.transpose(1, 2).contiguous()
        
        y_share_v = self.norm_share(y_share_v)
        y_share_t = self.norm_share(y_share_t)
        
        out_v = self.out_proj_share(y_share_v)
        out_t = self.out_proj_share(y_share_t)
        
        out_v = self.dropout_share(out_v).transpose(1, 2).contiguous().view(B, self.d_model, H, W)
        out_t = self.dropout_share(out_t).transpose(1, 2).contiguous().view(B, self.d_model, H, W)
        
        feat_v_share = feat_v + out_v
        feat_t_share = feat_t + out_t
        
        return feat_v_share, feat_t_share

class FFSSM(nn.Module):
    """Feature Fusion State Space Model (FF-SSM).

    Bidirectional state-space fusion block with cross-modal channel
    squeeze-and-excitation (SE) gating. Scans paired feature streams, dynamically
    scales their state representations using complementary channel attention, and
    merges them into a unified feature representation.

    Parameters
    ----------
    d_model : int
        Channel dimension of the input feature maps.
    d_state : int
        State-space hidden dimension (default: 4).
    ssm_ratio : float
        Channel expansion ratio for inner SSM dimension (default: 2.0).
    dt_rank : str or int
        Rank of the delta projection. Defaults to 'auto' (ceil(d_model / 16)).
    d_conv : int
        Kernel size of depthwise convolution pre-filtering (default: 3).
    conv_bias : bool
        Whether to include bias in depthwise conv (default: True).
    dropout : float
        Dropout probability (default: 0.0).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 4,
        ssm_ratio: float = 2.0,
        dt_rank: str = "auto",
        d_conv: int = 3,
        conv_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        
        self.in_proj_1 = nn.Linear(d_model, self.d_inner)
        self.in_proj_2 = nn.Linear(d_model, self.d_inner)
        
        self.conv_1 = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv//2, groups=self.d_inner, bias=conv_bias)
        self.conv_2 = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv//2, groups=self.d_inner, bias=conv_bias)
        self.act = nn.SiLU()
        
        self.x_proj_1 = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.x_proj_2 = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        
        self.dt_proj_1 = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.dt_proj_2 = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log_1 = nn.Parameter(torch.log(A))
        self.A_log_2 = nn.Parameter(torch.log(A))
        self.D_1 = nn.Parameter(torch.ones(self.d_inner))
        self.D_2 = nn.Parameter(torch.ones(self.d_inner))
        
        # Cross-Modal SE
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp_1 = nn.Sequential(
            nn.Conv2d(self.d_inner, self.d_inner // 2, 1),
            nn.ReLU(),
            nn.Conv2d(self.d_inner // 2, self.d_inner, 1),
            nn.Sigmoid()
        )
        self.mlp_2 = nn.Sequential(
            nn.Conv2d(self.d_inner, self.d_inner // 2, 1),
            nn.ReLU(),
            nn.Conv2d(self.d_inner // 2, self.d_inner, 1),
            nn.Sigmoid()
        )
        
        self.out_proj = nn.Linear(2 * self.d_inner, d_model)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        B, C, H, W = f1.shape
        L = H * W
        
        x1 = f1.permute(0, 2, 3, 1).contiguous()
        x1 = self.in_proj_1(x1).permute(0, 3, 1, 2).contiguous()
        x1 = self.act(self.conv_1(x1))
        
        x2 = f2.permute(0, 2, 3, 1).contiguous()
        x2 = self.in_proj_2(x2).permute(0, 3, 1, 2).contiguous()
        x2 = self.act(self.conv_2(x2))
        
        x1_flat = x1.view(B, self.d_inner, L)
        x2_flat = x2.view(B, self.d_inner, L)
        
        # Scan 1
        x1_proj_in = x1_flat.transpose(1, 2)
        proj_1 = self.x_proj_1(x1_proj_in)
        dt_1, B_1, C_1 = torch.split(proj_1, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_1 = self.dt_proj_1(dt_1).transpose(1, 2).contiguous()
        B_1 = B_1.transpose(1, 2).contiguous()
        C_1 = C_1.transpose(1, 2).contiguous()
        A_1 = -torch.exp(self.A_log_1)
        y1 = selective_scan_fn(x1_flat, dt_1, A_1, B_1, C_1, self.D_1, delta_bias=self.dt_proj_1.bias, delta_softplus=True)
        
        # Scan 2
        x2_proj_in = x2_flat.transpose(1, 2)
        proj_2 = self.x_proj_2(x2_proj_in)
        dt_2, B_2, C_2 = torch.split(proj_2, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_2 = self.dt_proj_2(dt_2).transpose(1, 2).contiguous()
        B_2 = B_2.transpose(1, 2).contiguous()
        C_2 = C_2.transpose(1, 2).contiguous()
        A_2 = -torch.exp(self.A_log_2)
        y2 = selective_scan_fn(x2_flat, dt_2, A_2, B_2, C_2, self.D_2, delta_bias=self.dt_proj_2.bias, delta_softplus=True)
        
        # Reshape y1, y2 back to H, W
        y1 = y1.view(B, self.d_inner, H, W)
        y2 = y2.view(B, self.d_inner, H, W)
        
        # SE gating
        e1 = self.mlp_1(self.pool(x1))
        e2 = self.mlp_2(self.pool(x2))
        
        y1 = y1 * e2
        y2 = y2 * e1
        
        # Concat and project
        y = torch.cat([y1, y2], dim=1)  # B, 2*d_inner, H, W
        y = y.permute(0, 2, 3, 1).contiguous()
        out = self.out_proj(y).permute(0, 3, 1, 2).contiguous()
        return f1 + f2 + out


class MS2FusionBlock(nn.Module):
    """Hierarchical Multi-Scale State-Space Feature Fusion Block.

    Integrates CP-SSM (complementary feature interaction), SP-SSM (shared
    feature extraction), and FF-SSM (intra-modality and inter-modality fusion)
    into a unified fusion block.

    Parameters
    ----------
    d_model : int
        Channel dimension of the input feature maps.
    d_state : int
        State-space hidden dimension (default: 4).
    ssm_ratio : float
        Channel expansion ratio for inner SSM dimension (default: 2.0).
    dt_rank : str or int
        Rank of the delta projection. Defaults to 'auto' (ceil(d_model / 16)).
    d_conv : int
        Kernel size of depthwise convolution pre-filtering (default: 3).
    conv_bias : bool
        Whether to include bias in depthwise conv (default: True).
    dropout : float
        Dropout probability (default: 0.0).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 4,
        ssm_ratio: float = 2.0,
        dt_rank: str = "auto",
        d_conv: int = 3,
        conv_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.cpssm = CPSSM(d_model, d_state, ssm_ratio, dt_rank, d_conv, conv_bias, dropout)
        self.spssm = SPSSM(d_model, d_state, ssm_ratio, dt_rank, d_conv, conv_bias, dropout)
        self.ff_v = FFSSM(d_model, d_state, ssm_ratio, dt_rank, d_conv, conv_bias, dropout)
        self.ff_t = FFSSM(d_model, d_state, ssm_ratio, dt_rank, d_conv, conv_bias, dropout)
        self.ff_cross = FFSSM(d_model, d_state, ssm_ratio, dt_rank, d_conv, conv_bias, dropout)
        
    def forward(self, feat_v: torch.Tensor, feat_t: torch.Tensor) -> torch.Tensor:
        feat_v_cross, feat_t_cross = self.cpssm(feat_v, feat_t)
        feat_v_share, feat_t_share = self.spssm(feat_v, feat_t)
        
        feat_v_fused = self.ff_v(feat_v_cross, feat_v_share)
        feat_t_fused = self.ff_t(feat_t_cross, feat_t_share)
        
        feat_fused = self.ff_cross(feat_v_fused, feat_t_fused)
        
        return feat_fused
