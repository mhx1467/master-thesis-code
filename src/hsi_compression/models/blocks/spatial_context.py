import torch
import torch.nn as nn

from .decoder import ResidualConvBlock


class SpatialContextEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_channels: int = 16,
        context_channels: int = 64,
        activation_slope: float = 0.2,
    ):
        super().__init__()
        mid_channels = max(context_channels // 2, embed_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, embed_channels, kernel_size=1),
            nn.LeakyReLU(activation_slope, inplace=True),
            nn.Conv2d(embed_channels, mid_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(activation_slope, inplace=True),
            nn.Conv2d(mid_channels, context_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(activation_slope, inplace=True),
            ResidualConvBlock(context_channels, activation_slope=activation_slope),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
