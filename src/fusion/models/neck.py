import torch
import torch.nn as nn
from fusion.models.backbone import Conv, C3

class PANet(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024]):
        super().__init__()
        c_p3, c_p4, c_p5 = in_channels
        
        # Top-down pathway
        self.conv_p5 = Conv(c_p5, c_p4, k=1)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.c3_p4 = C3(c_p5, c_p4, shortcut=False)
        
        self.conv_p4 = Conv(c_p4, c_p3, k=1)
        self.c3_p3 = C3(c_p4, c_p3, shortcut=False)
        
        # Bottom-up pathway
        self.down_conv1 = Conv(c_p3, c_p3, k=3, s=2)
        self.c3_n4 = C3(c_p3 + c_p4, c_p4, shortcut=False)
        
        self.down_conv2 = Conv(c_p4, c_p4, k=3, s=2)
        self.c3_n5 = C3(c_p4 + c_p4, c_p5, shortcut=False)

    def forward(self, features):
        f_p3, f_p4, f_p5 = features
        
        # Top-down
        p5_proj = self.conv_p5(f_p5)
        p5_up = self.up(p5_proj)
        p4_td = self.c3_p4(torch.cat([p5_up, f_p4], dim=1))
        
        p4_proj = self.conv_p4(p4_td)
        p4_up = self.up(p4_proj)
        n3 = self.c3_p3(torch.cat([p4_up, f_p3], dim=1))
        
        # Bottom-up
        n3_down = self.down_conv1(n3)
        n4 = self.c3_n4(torch.cat([n3_down, p4_td], dim=1))
        
        n4_down = self.down_conv2(n4)
        n5 = self.c3_n5(torch.cat([n4_down, p5_proj], dim=1))
        
        return n3, n4, n5
