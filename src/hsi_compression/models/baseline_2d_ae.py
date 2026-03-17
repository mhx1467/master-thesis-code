import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Baseline2DAutoencoder(nn.Module):
    """
    Simple 2D convolutional autoencoder for HSI patches.

    Input:
      (N, C, 128, 128)

    Latent after two downsamples:
      (N, latent_channels, 32, 32)
    """
    def __init__(
        self,
        in_channels: int = 224,
        hidden_channels: tuple[int, int] = (128, 64),
        latent_channels: int = 16,
    ):
        super().__init__()

        h1, h2 = hidden_channels

        self.enc_block1 = ConvBlock(in_channels, h1)
        self.down1 = nn.Conv2d(h1, h1, kernel_size=4, stride=2, padding=1)   # 128 -> 64

        self.enc_block2 = ConvBlock(h1, h2)
        self.down2 = nn.Conv2d(h2, h2, kernel_size=4, stride=2, padding=1)   # 64 -> 32

        self.bottleneck = nn.Conv2d(h2, latent_channels, kernel_size=3, padding=1)

        self.up1 = nn.ConvTranspose2d(latent_channels, h2, kernel_size=4, stride=2, padding=1)  # 32 -> 64
        self.dec_block1 = ConvBlock(h2, h2)

        self.up2 = nn.ConvTranspose2d(h2, h1, kernel_size=4, stride=2, padding=1)  # 64 -> 128
        self.dec_block2 = ConvBlock(h1, h1)

        self.out_conv = nn.Conv2d(h1, in_channels, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc_block1(x)
        x = self.down1(x)
        x = self.enc_block2(x)
        x = self.down2(x)
        z = self.bottleneck(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.up1(z)
        x = self.dec_block1(x)
        x = self.up2(x)
        x = self.dec_block2(x)
        x_hat = self.out_conv(x)
        return torch.sigmoid(x_hat)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return {
            "x_hat": x_hat,
            "z": z,
        }

    @staticmethod
    def compression_ratio_proxy(
        input_shape: tuple[int, int, int],
        latent_shape: tuple[int, int, int],
    ) -> float:
        c, h, w = input_shape
        cz, hz, wz = latent_shape
        return (c * h * w) / (cz * hz * wz)