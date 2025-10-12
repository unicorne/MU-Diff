# models/registration2d.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def meshgrid_2d(h, w, device):
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device),
        indexing='ij'
    )
    grid = torch.stack((xx, yy), dim=-1)  # (H, W, 2) in [-1,1]
    return grid

class UNet2D(nn.Module):
    def __init__(self, in_ch=2, base=32, out_ch=2):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.InstanceNorm2d(cout), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.InstanceNorm2d(cout), nn.LeakyReLU(0.2, inplace=True),
            )
        self.down1 = block(in_ch, base)
        self.down2 = block(base, base*2)
        self.down3 = block(base*2, base*4)
        self.pool = nn.MaxPool2d(2)
        self.bott = block(base*4, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.dec3 = block(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.dec2 = block(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.dec1 = block(base*2, base)
        self.out = nn.Conv2d(base, out_ch, 3, padding=1)  # velocity field (dy, dx)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        b  = self.bott(self.pool(d3))
        u3 = self.up3(b)
        c3 = self.dec3(torch.cat([u3, d3], dim=1))
        u2 = self.up2(c3)
        c2 = self.dec2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(c2)
        c1 = self.dec1(torch.cat([u1, d1], dim=1))
        v  = self.out(c1)
        return v

class VecInt(nn.Module):
    """Scaling-and-squaring integration for stationary velocity fields."""
    def __init__(self, n_steps=5):
        super().__init__()
        self.n_steps = n_steps

    def forward(self, v):
        # v: (B, 2, H, W) pixel-displacements; convert to normalized grid increments
        B, _, H, W = v.shape
        # convert pixel displacement to normalized [-1,1] per axis
        vx = v[:,1] * (2.0 / (W - 1))
        vy = v[:,0] * (2.0 / (H - 1))
        phi = torch.stack([vx, vy], dim=1) / (2 ** self.n_steps)  # small step
        grid = None
        for _ in range(self.n_steps):
            # compose: phi <- phi ∘ phi + phi  (approximate exp via squaring)
            if grid is None:
                base = meshgrid_2d(H, W, v.device).permute(2,0,1).unsqueeze(0)  # (1,2,H,W)
                grid = base
            # current warp grid in normalized coords:
            cur = (grid + phi.permute(0,2,3,1)).permute(0,3,1,2)  # (B,2,H,W)
            samp = F.grid_sample(phi, cur.permute(0,2,3,1), align_corners=True, mode='bilinear', padding_mode='border')
            phi = phi + samp
        # return final displacement in normalized coords (dy,dx order preserved)
        return phi

class SpatialTransformer2D(nn.Module):
    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, x, disp_norm):
        # x: (B, 1, H, W) image; disp_norm: (B, 2, H, W) normalized displacements
        B, _, H, W = x.shape
        base = meshgrid_2d(H, W, x.device).unsqueeze(0).expand(B, -1, -1, -1)  # (B,H,W,2)
        flow = base + disp_norm.permute(0,2,3,1)  # (B,H,W,2)
        return F.grid_sample(x, flow, align_corners=True, mode=self.mode, padding_mode='border')

# in models.py
class RegNet2D(nn.Module):
    def __init__(self, in_ch=2, n_steps=5):
        super().__init__()
        self.unet = UNet2D(in_ch=in_ch, out_ch=2)
        self.integrate = VecInt(n_steps=n_steps)
        self.stn_img = SpatialTransformer2D(mode='bilinear')
        self.stn_mask_soft = SpatialTransformer2D(mode='bilinear')  # <-- for training
        self.stn_mask_hard = SpatialTransformer2D(mode='nearest')   # <-- for eval/inference

    def forward(self, moving, fixed, moving_mask=None):
        x = torch.cat([moving, fixed], dim=1)
        v = self.unet(x)
        disp_norm = self.integrate(v)
        warped = self.stn_img(moving, disp_norm)
        out = {'warped_image': warped, 'disp_norm': disp_norm, 'v': v}
        if moving_mask is not None:
            stn = self.stn_mask_soft if self.training else self.stn_mask_hard
            warped_mask = stn(moving_mask, disp_norm)                # no rounding here
            warped_mask = warped_mask.clamp(0, 1)
            out['warped_mask'] = warped_mask
        return out

