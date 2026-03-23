import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline3DFullBandsAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 202,
        latent_channels: int = 4,
        hidden_channels: tuple[int, int, int] = (8, 16, 32),
        output_activation: str | None = "sigmoid",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        h1, h2, h3 = hidden_channels

        self.enc3d = nn.Sequential(
            # (N,1,C,128,128) -> (N,h1,C,64,64)
            nn.Conv3d(1, h1, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(h1, h1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (N,h2,C,32,32)
            nn.Conv3d(h1, h2, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(h2, h2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # spectral compression: 202 -> 101 -> 51 -> 26 -> 13
            nn.Conv3d(h2, h3, kernel_size=3, stride=(2, 1, 1), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(h3, h3, kernel_size=3, stride=(2, 1, 1), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(h3, latent_channels, kernel_size=3, stride=(2, 1, 1), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(latent_channels, latent_channels, kernel_size=3, stride=(2, 1, 1), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        dec_layers: list[nn.Module] = [
            # 13 -> 26
            nn.ConvTranspose3d(
                latent_channels,
                latent_channels,
                kernel_size=3,
                stride=(2, 1, 1),
                padding=1,
                output_padding=(1, 0, 0),
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # 26 -> 51
            nn.ConvTranspose3d(
                latent_channels,
                h3,
                kernel_size=3,
                stride=(2, 1, 1),
                padding=1,
                output_padding=(0, 0, 0),
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # 51 -> 101
            nn.ConvTranspose3d(
                h3,
                h3,
                kernel_size=3,
                stride=(2, 1, 1),
                padding=1,
                output_padding=(0, 0, 0),
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # 101 -> 202
            nn.ConvTranspose3d(
                h3,
                h2,
                kernel_size=3,
                stride=(2, 1, 1),
                padding=1,
                output_padding=(1, 0, 0),
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # spatial upsample: 32 -> 64
            nn.ConvTranspose3d(
                h2,
                h1,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
                output_padding=(0, 1, 1),
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(h1, h1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # spatial upsample: 64 -> 128
            nn.ConvTranspose3d(
                h1,
                1,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
                output_padding=(0, 1, 1),
            ),
        ]

        if output_activation == "sigmoid":
            dec_layers.append(nn.Sigmoid())
        elif output_activation not in (None, "identity"):
            raise ValueError("output_activation must be one of: 'sigmoid', 'identity', None")

        self.dec3d = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x3d = x.unsqueeze(1)
        return self.enc3d(x3d)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x3d = self.dec3d(z)

        if x3d.shape[2] < self.in_channels:
            x3d = F.pad(x3d, (0, 0, 0, 0, 0, self.in_channels - x3d.shape[2]))
        elif x3d.shape[2] > self.in_channels:
            x3d = x3d[:, :, : self.in_channels]

        return x3d.squeeze(1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return {"x_hat": x_hat, "z": z}

    @staticmethod
    def compression_ratio_proxy(
        input_shape: tuple[int, int, int],
        latent_shape: tuple[int, ...],
    ) -> float:
        c, h, w = input_shape
        latent_els = 1
        for s in latent_shape:
            latent_els *= s
        return (c * h * w) / latent_els
