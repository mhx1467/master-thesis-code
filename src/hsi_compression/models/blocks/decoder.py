import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, activation_slope: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(activation_slope, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return self.act(x + residual)


class SpectralFirstDecoder(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        activation_slope: float = 0.2,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(activation_slope, inplace=True),
            ResidualConvBlock(hidden_channels, activation_slope=activation_slope),
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(activation_slope, inplace=True),
            ResidualConvBlock(hidden_channels, activation_slope=activation_slope),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(activation_slope, inplace=True),
            ResidualConvBlock(hidden_channels, activation_slope=activation_slope),
        )
        self.head = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        x = self.stem(z_q)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2(x)
        return self.head(x)
