import torch
import torch.nn as nn
import torch.nn.functional as F

class OpticalFlow(nn.Module):
    def __init__(self, in_channels = 256, out_channels = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels*2, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.layers = nn.ModuleList([self.conv1, self.conv2])

    def forward(self, f1, f2):
        # f1: (B, C, H, W)
        f = torch.cat([f1, f2], dim=1)
        for layer in self.layers:
            f = layer(f)
        return f