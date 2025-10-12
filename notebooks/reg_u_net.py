import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpatialTransformer(nn.Module):
    """
    A spatial transformer network layer to warp an image based on a dense deformation field.
    This can be used in the training script to apply the flows predicted by the RegistrationUNet.
    """
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode
        # Create a static grid for sampling
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        new_locs = self.grid.to(src.device) + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for grid_sample
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class SelfAttention(nn.Module):
    """ Self-attention layer for feature maps. """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        q = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, height * width)
        v = self.value(x).view(batch_size, -1, height * width)

        attention_matrix = torch.bmm(q, k)
        attention_weights = F.softmax(attention_matrix / (self.in_channels // 8) ** 0.5, dim=2)
        
        attention_output = torch.bmm(v, attention_weights.permute(0, 2, 1))
        attention_output = attention_output.view(batch_size, channels, height, width)
        
        return x + self.gamma * attention_output


class ResGroupNormBlock(nn.Module):
    """ A residual block with Group Normalization and SiLU activation. """
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.act2 = nn.SiLU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x + residual


class RegistrationUNet(nn.Module):
    """
    A high-performance U-Net for registration that predicts three separate 
    deformation fields (DDFs). This model ONLY predicts the flows. 
    The warping should be handled in the training loop.
    """
    def __init__(self, in_channels=4, base_channels=32, num_resolutions=4, max_displacement=32.0):
        super().__init__()
        self.max_displacement = max_displacement
        
        # --- Initial convolution ---
        self.in_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # --- Encoder Path ---
        self.encoder_blocks = nn.ModuleList()
        ch = base_channels
        for i in range(num_resolutions):
            self.encoder_blocks.append(
                nn.ModuleDict({
                    'res': ResGroupNormBlock(ch, ch * 2),
                    'down': nn.Conv2d(ch * 2, ch * 2, kernel_size=3, stride=2, padding=1)
                })
            )
            ch *= 2
            
        # --- Bottleneck with Self-Attention ---
        self.bottleneck = nn.Sequential(
            ResGroupNormBlock(ch, ch),
            SelfAttention(ch),
            ResGroupNormBlock(ch, ch)
        )
        
        # --- Decoder Path ---
        self.decoder_blocks = nn.ModuleList()
        for i in range(num_resolutions):
            self.decoder_blocks.append(
                nn.ModuleDict({
                    'up': nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    'res': ResGroupNormBlock(ch * 2, ch // 2)
                })
            )
            ch //= 2
            
        # --- Final Output Layer ---
        self.out_conv = nn.Conv2d(base_channels, 6, kernel_size=1)
        
        self.out_conv.weight.data.zero_()
        self.out_conv.bias.data.zero_()

    def forward(self, x):
        # The input 'x' should be the concatenated tensor of moving and fixed images
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
            skip_connection = skips[-(i + 1)]
            h = torch.cat([h, skip_connection], dim=1)
            h = block['res'](h)
            
        # Final processing to get the DDFs
        raw_flow = self.out_conv(h)
        scaled_flow = torch.tanh(raw_flow) * self.max_displacement
        
        # Split the 6 channels into 3 separate 2-channel flows
        flow1, flow2, flow3 = torch.chunk(scaled_flow, 3, dim=1)
        
        # Return the individual DDFs as a tuple
        return flow1, flow2, flow3

