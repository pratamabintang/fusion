import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k=1, s=1)
        self.cv2 = Conv(c_, c2, k=3, s=1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k=1, s=1)
        self.cv2 = Conv(c1, c_, k=1, s=1)
        self.cv3 = Conv(2 * c_, c2, k=1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class SPPF(nn.Module):
    def __init__(self, c1, c2=None, k=5):
        super().__init__()
        if c2 is None:
            c2 = c1
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, k=1, s=1)
        self.cv2 = Conv(c_ * 4, c2, k=1, s=1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))

class CSPDarkNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, base_depth=3):
        super().__init__()
        # Stem
        self.stem = Conv(in_channels, base_channels, k=6, s=2, p=2)
        # Stage 2
        self.stage2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, k=3, s=2),
            C3(base_channels * 2, base_channels * 2, n=base_depth)
        )
        # Stage 3
        self.stage3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, k=3, s=2),
            C3(base_channels * 4, base_channels * 4, n=base_depth * 2)
        )
        # Stage 4
        self.stage4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, k=3, s=2),
            C3(base_channels * 8, base_channels * 8, n=base_depth * 3)
        )
        # Stage 5
        self.stage5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, k=3, s=2),
            C3(base_channels * 16, base_channels * 16, n=base_depth),
            SPPF(base_channels * 16, base_channels * 16)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        p3 = self.stage3(x)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        return p3, p4, p5

class DualStreamCSPDarkNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, base_depth=3):
        super().__init__()
        self.stream_v = CSPDarkNet(in_channels, base_channels, base_depth)
        self.stream_t = CSPDarkNet(in_channels, base_channels, base_depth)

    def forward(self, feat_v, feat_t):
        return self.stream_v(feat_v), self.stream_t(feat_t)
