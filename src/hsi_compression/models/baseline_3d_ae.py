import torch
import torch.nn as nn


class Baseline3DAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 202,
        latent_channels: int = 16,
        hidden_channels: tuple[int, int] = (32, 64),
    ):
        super().__init__()
        h1, h2 = hidden_channels

        self.enc1 = nn.Sequential(
            nn.Conv3d(1, h1, kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(h1, h1, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(h1, h2, kernel_size=3, stride=(2, 1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(h2, latent_channels, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.ReLU(inplace=True),
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose3d(
                latent_channels,
                h2,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
                output_padding=(0, 1, 1),
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(
                h2,
                h1,
                kernel_size=3,
                stride=(2, 1, 1),
                padding=1,
                output_padding=(1, 0, 0),
            ),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(
                h1,
                h1,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
                output_padding=(0, 1, 1),
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(
                h1,
                1,
                kernel_size=(5, 3, 3),
                stride=(2, 1, 1),
                padding=(2, 1, 1),
                output_padding=(1, 0, 0),
            ),
        )

        self.adapt_in_depth = nn.AdaptiveAvgPool3d((in_channels, 128, 128))
        self.adapt_latent = nn.AdaptiveAvgPool3d((in_channels // 4, 32, 32))
        self.adapt_out_depth = nn.AdaptiveAvgPool3d((in_channels, 128, 128))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x3d = x.unsqueeze(1)  # (N,1,C,H,W)
        x3d = self.adapt_in_depth(x3d)
        h = self.enc1(x3d)
        z = self.enc2(h)
        z = self.adapt_latent(z)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec1(z)
        x3d_hat = self.dec2(h)
        x3d_hat = self.adapt_out_depth(x3d_hat)
        x_hat = x3d_hat.squeeze(1)
        return x_hat

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
