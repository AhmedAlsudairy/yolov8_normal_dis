import torch
import torch.nn as nn

class GaussianHead(nn.Module):
    def __init__(self, nc=80, anchors=()):
        super(GaussianHead, self).__init__()
        self.nc = nc  # Number of classes
        self.no = nc + 9  # Number of outputs per anchor (class + xywh + variances)
        self.nl = len(anchors)  # Number of detection layers
        self.na = len(anchors[0]) // 2  # Number of anchors
        self.anchors = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in [256, 512, 1024])  # Detection heads

    def forward(self, x):
        z = []  # Outputs
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # Apply conv
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            z.append(x[i])
        return z
