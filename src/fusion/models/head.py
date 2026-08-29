import torch
import torch.nn as nn
import math

class Detect(nn.Module):
    def __init__(self, nc=1, anchors=None, ch=[256, 512, 1024]):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors) if anchors is not None else 3
        
        if anchors is None:
            anchors = [[10, 13, 16, 30, 33, 23], 
                       [30, 61, 62, 45, 59, 119], 
                       [116, 90, 156, 198, 373, 326]]
        self.na = len(anchors[0]) // 2
        
        # Grid sizes and anchor tensors
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))
        
        self.stride = torch.tensor([8, 16, 32])
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        
        # Init weights (Biases for objectness, etc.)
        for mi, s in zip(self.m, self.stride):
            b = mi.bias.view(self.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (self.nc - 0.999999))
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                # Decode
                grid_y, grid_x = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing='ij')
                grid = torch.stack((grid_x, grid_y), 2).view(1, 1, ny, nx, 2).float().to(x[i].device)
                
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]      # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)
