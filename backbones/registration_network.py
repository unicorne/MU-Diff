# In a new file, e.g., backbones/registration_network.py
import torch
import torch.nn as nn
from .ncsnpp_generator_adagn_feat import ResnetBlockDDPM

# In backbones/registration_network.py

import torch
import torch.nn as nn

class RegistrationUNet(nn.Module):
    # Add a max_displacement parameter
    def __init__(self, in_channels=4, base_channels=32, num_resolutions=4, max_displacement=16.0):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        for i in range(num_resolutions):
            self.down_blocks.append(nn.Conv2d(ch, ch*2, 3, stride=2, padding=1))
            ch *= 2

        self.bottleneck = nn.Conv2d(ch, ch, 3, padding=1)

        self.up_blocks = nn.ModuleList()
        for i in range(num_resolutions):
            self.up_blocks.append(nn.ConvTranspose2d(ch, ch//2, 4, stride=2, padding=1))
            ch //= 2

        self.out_conv = nn.Conv2d(base_channels, 6, 3, padding=1)
        
        self.final_activation = nn.Tanh()
        self.max_displacement = max_displacement
        
        # Initialize output layer to predict zero flow initially
        self.out_conv.weight.data.zero_()
        self.out_conv.bias.data.zero_()

    def forward(self, x):
        h = self.in_conv(x)
        skips = [h]
        for block in self.down_blocks:
            h = block(h)
            skips.append(h)

        h = self.bottleneck(h)

        for i, block in enumerate(self.up_blocks):
            h = block(h)
            h = h + skips[-(i+2)] 
        
        raw_flow = self.out_conv(h)
        
        # Apply the activation and scale the flow to a plausible range
        scaled_flow = self.final_activation(raw_flow) * self.max_displacement
        
        # Split the 6 channels into 3 separate 2-channel flows
        flow1, flow2, flow3 = torch.chunk(scaled_flow, 3, dim=1)
        
        return flow1, flow2, flow3