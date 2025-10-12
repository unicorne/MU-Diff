# backbones/regnet2d.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def grad_smooth(flow):
    dx = torch.abs(flow[:,:, :,1:] - flow[:,:,:, :-1])
    dy = torch.abs(flow[:,:,1:, :] - flow[:,:, :-1,:])
    return dx.mean() + dy.mean()

def conv(in_ch,out_ch,k=3,s=1,p=1):
    return nn.Sequential(nn.Conv2d(in_ch,out_ch,k,s,p), nn.InstanceNorm2d(out_ch, affine=True), nn.LeakyReLU(0.2, inplace=True))

class RegNet2D(nn.Module):
    def __init__(self, in_ch=2, base=32, max_disp=16):
        super().__init__()
        self.max_disp = max_disp
        # encoder
        self.e1 = conv(in_ch, base)
        self.e2 = nn.Sequential(conv(base, base*2, s=2), conv(base*2, base*2))
        self.e3 = nn.Sequential(conv(base*2, base*4, s=2), conv(base*4, base*4))
        self.e4 = nn.Sequential(conv(base*4, base*8, s=2), conv(base*8, base*8))
        # decoder
        self.u3 = conv(base*8+base*4, base*4)
        self.u2 = conv(base*4+base*2, base*2)
        self.u1 = conv(base*2+base,   base)
        self.out = nn.Conv2d(base, 2, kernel_size=3, padding=1)  # flow (dx, dy)

    def forward(self, src, tgt):
        # concat: [src, tgt]
        x = torch.cat([src, tgt], dim=1)
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        d3 = F.interpolate(e4, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = self.u3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = self.u2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = self.u1(torch.cat([d1, e1], dim=1))
        flow = torch.tanh(self.out(d1)) * self.max_disp  # (B,2,H,W)
        return flow

def warp(src, flow):
    # flow is in pixels (dx, dy). grid_sample expects normalized coordinates.
    B, C, H, W = src.shape
    # build base grid in [-1,1]
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=src.device),
        torch.linspace(-1, 1, W, device=src.device),
        indexing='ij'
    )
    base = torch.stack((xx, yy), dim=-1).unsqueeze(0).repeat(B,1,1,1)  # (B,H,W,2)
    # normalize flow
    norm = torch.stack((flow[:,0]/(W/2), flow[:,1]/(H/2)), dim=-1)     # (B,2,H,W,)->(B,2,H,W,2)? reshape:
    norm = norm.permute(0,2,3,1)  # (B,H,W,2)
    grid = base + norm
    return F.grid_sample(src, grid, mode='bilinear', padding_mode='border', align_corners=False)
