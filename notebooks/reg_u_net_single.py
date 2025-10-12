import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTransformer(nn.Module):
    """
    A spatial transformer that warps an image (or mask) with a dense deformation field (flow).
    For images use mode='bilinear'; for masks use mode='nearest'.
    """
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode
        # Create a static sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)  # (ndims, H, W)
        grid = torch.unsqueeze(grid, 0).type(torch.FloatTensor)  # (1, ndims, H, W)
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        """
        src:  (B, C, H, W)
        flow: (B, 2, H, W) displacement in (y, x) pixel units
        """
        new_locs = self.grid.to(src.device) + flow
        shape = flow.shape[2:]  # (H, W)

        # Normalize to [-1, 1] for grid_sample
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # (B, H, W, 2) and swap to (x, y) order for grid_sample
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class SelfAttention(nn.Module):
    """ Lightweight self-attention for feature maps. """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        b, c, h, w = x.size()
        q = self.query(x).view(b, -1, h * w).permute(0, 2, 1)     # (b, hw, c//8)
        k = self.key(x).view(b, -1, h * w)                        # (b, c//8, hw)
        v = self.value(x).view(b, -1, h * w)                      # (b, c, hw)

        attn = torch.bmm(q, k)                                    # (b, hw, hw)
        attn = F.softmax(attn / max(1, (self.in_channels // 8)) ** 0.5, dim=2)
        out = torch.bmm(v, attn.permute(0, 2, 1))                 # (b, c, hw)
        out = out.view(b, c, h, w)
        return x + self.gamma * out


class ResGroupNormBlock(nn.Module):
    """ Residual block with GroupNorm + SiLU. """
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.act1  = nn.SiLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.act2  = nn.SiLU()

        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x + residual


class RegistrationUNet(nn.Module):
    """
    U-Net that predicts a single 2D dense deformation field (flow) for 1:1 registration.
    Input:  2 channels (synthetic/moving, target/fixed)
    Output: 2 channels (dy, dx) flow
    """
    def __init__(self, in_channels=2, base_channels=32, num_resolutions=4, max_displacement=32.0):
        super().__init__()
        self.max_displacement = max_displacement

        # Stem
        self.in_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        ch = base_channels
        for _ in range(num_resolutions):
            self.encoder_blocks.append(
                nn.ModuleDict({
                    'res': ResGroupNormBlock(ch, ch * 2),
                    'down': nn.Conv2d(ch * 2, ch * 2, kernel_size=3, stride=2, padding=1)
                })
            )
            ch *= 2

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResGroupNormBlock(ch, ch),
            SelfAttention(ch),
            ResGroupNormBlock(ch, ch)
        )

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        for _ in range(num_resolutions):
            self.decoder_blocks.append(
                nn.ModuleDict({
                    'up': nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    'res': ResGroupNormBlock(ch * 2, ch // 2)
                })
            )
            ch //= 2

        # Final conv: output ONLY ONE flow (2 channels)
        self.out_conv = nn.Conv2d(base_channels, 2, kernel_size=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x):
        h = self.in_conv(x)
        skips = [h]

        # Encoder
        for block in self.encoder_blocks:
            h = block['res'](h)
            skips.append(h)
            h = block['down'](h)

        # Bottleneck
        h = self.bottleneck(h)

        # Decoder
        for i, block in enumerate(self.decoder_blocks):
            h = block['up'](h)
            skip = skips[-(i + 1)]
            h = torch.cat([h, skip], dim=1)
            h = block['res'](h)

        # Flow field in pixels, limited to [-max_disp, max_disp]
        raw_flow = self.out_conv(h)
        flow = torch.tanh(raw_flow) * self.max_displacement  # (B, 2, H, W)
        return flow
