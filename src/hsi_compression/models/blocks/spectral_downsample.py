import torch
import torch.nn as nn


class SpectralPreservingDownsample(nn.Module):
    """Reduce spatial size while preserving spectral channel semantics."""

    def __init__(self, in_channels: int, activation_slope: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=in_channels,
                bias=True,
            ),
            nn.LeakyReLU(activation_slope, inplace=True),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=in_channels,
                bias=True,
            ),
            nn.LeakyReLU(activation_slope, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
