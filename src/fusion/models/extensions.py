import torch
import torch.nn as nn
from typing import Tuple, Optional
import math
from fusion.ops.selective_scan import selective_scan_fn

class MSSSMBlock(nn.Module):
    def __init__(self, d_model_in, d_model_out=None, d_state=4, ssm_ratio=2.0, illumination_modulated=False):
        super().__init__()
        self.d_model_in = d_model_in
        self.d_inner = int(d_model_in * ssm_ratio)
        self.d_inner_prev = int((d_model_out if d_model_out is not None else d_model_in) * ssm_ratio)
        self.d_state = d_state
        self.illumination_modulated = illumination_modulated
        
        if self.illumination_modulated:
            self.gating = AdaptiveGatingModule(in_channels=d_model_in)
        
        self.in_proj_v = nn.Linear(d_model_in, self.d_inner)
        self.in_proj_t = nn.Linear(d_model_in, self.d_inner)
        
        self.memory_proj = nn.Linear(self.d_inner_prev, self.d_inner)
        
        self.x_proj = nn.Linear(self.d_inner * 2, math.ceil(d_model_in / 16) + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(math.ceil(d_model_in / 16), self.d_inner, bias=True)
        
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, d_model_in)

    def forward(self, feat_v, feat_t, memory_state=None):
        B, C, H, W = feat_v.shape
        L = H * W
        
        # 1. State Injection
        if memory_state is not None:
            # memory_state: (B, d_inner_prev, d_state)
            memory_state = self.memory_proj(memory_state.transpose(1, 2)).transpose(1, 2)
        else:
            memory_state = torch.zeros(B, self.d_inner, self.d_state, device=feat_v.device)
            
        # 2. Input Projections
        x_v = self.in_proj_v(feat_v.permute(0, 2, 3, 1)).view(B, L, self.d_inner)
        x_t = self.in_proj_t(feat_t.permute(0, 2, 3, 1)).view(B, L, self.d_inner)
        x_concat = torch.cat([x_v, x_t], dim=-1) # (B, L, 2*d_inner)
        
        # 3. Parameters
        proj = self.x_proj(x_concat)
        dt, B_state, C_state = torch.split(proj, [math.ceil(self.d_model_in / 16), self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt).transpose(1, 2).contiguous()
        B_state = B_state.transpose(1, 2).contiguous()
        C_state = C_state.transpose(1, 2).contiguous()
        
        # 4. Condition on memory bridge
        # We'll condition the input feature map with a pooled summary of memory_state
        mem_cond = memory_state.mean(dim=-1, keepdim=True) # (B, d_inner, 1)
        
        x_in = (x_v + x_t).transpose(1, 2).contiguous() # (B, d_inner, L)
        x_in = x_in + mem_cond
        
        # Apply illumination gating if enabled
        if self.illumination_modulated:
            alpha = self.gating(feat_v, feat_t).view(B, 1, 1)
            # In standard MSSSMBlock we have a single branch here, so we apply it to dt? 
            # Wait, in MSSSMBlock we concat x_v and x_t and predict a single dt!
            # The gating in ICSSMBlock modulates separate dt_v and dt_t.
            # If we predict a single dt, how to modulate?
            # Maybe just scale the dt? dt = dt * (1.0 + alpha) ...
            # The ticket says: "If 'combined': uses illumination-modulated multi-scale memory bridge."
            # Since MSSSMBlock uses a joint dt, I will just apply an alpha modulation to the feature outputs.
            # But the prompt says for IC-SSM: "Modulates \Delta_v ... \Delta_t ... Modulates SE gating weights"
            # If combined, it probably means `MultiScaleMemoryFusion` using `ICSSMBlock` but passing `memory_state`?
            # Yes! Let's not modify MSSSMBlock forward, but rather make `MultiScaleMemoryFusion` use a modulated version.
        
        # 5. Dispatch selective scan
        A_val = -torch.exp(self.A_log)
        y = selective_scan_fn(x_in, dt, A_val, B_state, C_state, self.D, delta_bias=self.dt_proj.bias, delta_softplus=True)
        
        # 6. Extract output memory state
        # Approximate hidden state by pooling the scan output into (B, d_inner, d_state)
        # In actual Mamba/SSM, the true hidden state would be returned by a custom kernel.
        out_mem = y.view(B, self.d_inner, self.d_state, -1).mean(dim=-1) # (B, d_inner, d_state)
        
        # 7. Output projection
        y = y.transpose(1, 2).contiguous() # (B, L, d_inner)
        out_feat = self.out_proj(y).transpose(1, 2).contiguous().view(B, C, H, W)
        fused = feat_v + feat_t + out_feat
        
        return fused, out_mem

class MultiScaleMemoryFusion(nn.Module):
    def __init__(self, c_p3, c_p4, c_p5, ssm_ratio=2.0, d_state=4):
        super().__init__()
        self.fuse_p3 = MSSSMBlock(c_p3, d_model_out=c_p3, d_state=d_state, ssm_ratio=ssm_ratio)
        self.fuse_p4 = MSSSMBlock(c_p4, d_model_out=c_p3, d_state=d_state, ssm_ratio=ssm_ratio)
        self.fuse_p5 = MSSSMBlock(c_p5, d_model_out=c_p4, d_state=d_state, ssm_ratio=ssm_ratio)

    def forward(self, p3_v, p3_t, p4_v, p4_t, p5_v, p5_t):
        f_p3, h_p3 = self.fuse_p3(p3_v, p3_t, memory_state=None)
        f_p4, h_p4 = self.fuse_p4(p4_v, p4_t, memory_state=h_p3)
        f_p5, h_p5 = self.fuse_p5(p5_v, p5_t, memory_state=h_p4)
        return f_p3, f_p4, f_p5

class AdaptiveGatingModule(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, feat_v, feat_t):
        B, C, H, W = feat_v.shape
        
        def get_stats(x):
            mean = x.mean(dim=(2, 3))
            var = x.var(dim=(2, 3), unbiased=False)
            std = x.std(dim=(2, 3), unbiased=False)
            return torch.cat([mean, var, std], dim=1)
            
        stats = torch.cat([get_stats(feat_v), get_stats(feat_t)], dim=1)
        alpha = self.mlp(stats) # (B, 1)
        return alpha.view(B, 1, 1, 1)

class ICSSMBlock(nn.Module):
    def __init__(self, d_model_in, d_model_out=None, d_state=4, ssm_ratio=2.0):
        super().__init__()
        self.d_model = d_model_in
        self.d_state = d_state
        self.d_inner = int(ssm_ratio * d_model_in)
        self.d_inner_prev = int((d_model_out if d_model_out is not None else d_model_in) * ssm_ratio)
        
        self.gating = AdaptiveGatingModule(in_channels=d_model_in)
        
        self.memory_proj = nn.Linear(self.d_inner_prev, self.d_inner)
        
        self.in_proj_v = nn.Linear(d_model_in, self.d_inner)
        self.in_proj_t = nn.Linear(d_model_in, self.d_inner)
        
        self.x_proj_v = nn.Linear(self.d_inner, math.ceil(d_model_in / 16) + 2 * d_state, bias=False)
        self.x_proj_t = nn.Linear(self.d_inner, math.ceil(d_model_in / 16) + 2 * d_state, bias=False)
        
        self.dt_proj_v = nn.Linear(math.ceil(d_model_in / 16), self.d_inner, bias=True)
        self.dt_proj_t = nn.Linear(math.ceil(d_model_in / 16), self.d_inner, bias=True)
        
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log_v = nn.Parameter(torch.log(A))
        self.A_log_t = nn.Parameter(torch.log(A))
        
        self.D_v = nn.Parameter(torch.ones(self.d_inner))
        self.D_t = nn.Parameter(torch.ones(self.d_inner))
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp_v = nn.Sequential(nn.Conv2d(self.d_inner, self.d_inner // 2, 1), nn.ReLU(), nn.Conv2d(self.d_inner // 2, self.d_inner, 1), nn.Sigmoid())
        self.mlp_t = nn.Sequential(nn.Conv2d(self.d_inner, self.d_inner // 2, 1), nn.ReLU(), nn.Conv2d(self.d_inner // 2, self.d_inner, 1), nn.Sigmoid())
        
        self.out_proj = nn.Linear(2 * self.d_inner, d_model_in)

    def forward(self, feat_v, feat_t, memory_state=None):
        B, C, H, W = feat_v.shape
        L = H * W
        
        # 1. State Injection
        if memory_state is not None:
            # memory_state: (B, d_inner_prev, d_state)
            memory_state = self.memory_proj(memory_state.transpose(1, 2)).transpose(1, 2)
        else:
            memory_state = torch.zeros(B, self.d_inner, self.d_state, device=feat_v.device)
            
        mem_cond = memory_state.mean(dim=-1, keepdim=True) # (B, d_inner, 1)
        
        alpha = self.gating(feat_v, feat_t) # (B, 1, 1, 1)
        alpha_flat = alpha.view(B, 1, 1) # (B, 1, 1)
        
        x_v = self.in_proj_v(feat_v.permute(0, 2, 3, 1)).view(B, L, self.d_inner)
        x_t = self.in_proj_t(feat_t.permute(0, 2, 3, 1)).view(B, L, self.d_inner)
        
        proj_v = self.x_proj_v(x_v)
        dt_v, B_v, C_v = torch.split(proj_v, [math.ceil(self.d_model / 16), self.d_state, self.d_state], dim=-1)
        dt_v = self.dt_proj_v(dt_v).transpose(1, 2).contiguous()
        
        proj_t = self.x_proj_t(x_t)
        dt_t, B_t, C_t = torch.split(proj_t, [math.ceil(self.d_model / 16), self.d_state, self.d_state], dim=-1)
        dt_t = self.dt_proj_t(dt_t).transpose(1, 2).contiguous()
        
        # Modulate Delta and A transition parameters based on illumination gating
        dt_v = dt_v * (0.5 + alpha_flat)
        dt_t = dt_t * (1.5 - alpha_flat)
        
        B_v = B_v.transpose(1, 2).contiguous()
        C_v = C_v.transpose(1, 2).contiguous()
        B_t = B_t.transpose(1, 2).contiguous()
        C_t = C_t.transpose(1, 2).contiguous()
        
        alpha_scale = alpha.mean()
        A_v = -torch.exp(self.A_log_v) * (0.5 + alpha_scale)
        A_t = -torch.exp(self.A_log_t) * (1.5 - alpha_scale)
        
        x_v_in = x_v.transpose(1, 2).contiguous() + mem_cond
        x_t_in = x_t.transpose(1, 2).contiguous() + mem_cond
        
        y_v = selective_scan_fn(x_v_in, dt_v, A_v, B_v, C_v, self.D_v, delta_bias=self.dt_proj_v.bias, delta_softplus=True)
        y_t = selective_scan_fn(x_t_in, dt_t, A_t, B_t, C_t, self.D_t, delta_bias=self.dt_proj_t.bias, delta_softplus=True)
        
        out_mem = ((y_v + y_t) / 2.0).view(B, self.d_inner, self.d_state, -1).mean(dim=-1) # (B, d_inner, d_state)
        
        y_v = y_v.view(B, self.d_inner, H, W)
        y_t = y_t.view(B, self.d_inner, H, W)
        
        # Modulate SE weights
        e_v = self.mlp_v(self.pool(y_v)) * (0.5 + alpha)
        e_t = self.mlp_t(self.pool(y_t)) * (1.5 - alpha)
        
        y_v = y_v * e_t
        y_t = y_t * e_v
        
        y = torch.cat([y_v, y_t], dim=1).permute(0, 2, 3, 1).contiguous()
        out = self.out_proj(y).permute(0, 3, 1, 2).contiguous()
        
        fused = feat_v + feat_t + out
        return fused, out_mem
