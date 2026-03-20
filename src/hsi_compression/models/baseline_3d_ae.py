import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline3DAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 202,
        latent_channels: int = 16,
        hidden_channels: tuple[int, int] = (32, 64),
    ):
        super().__init__()
        self.in_channels = in_channels
        h1, h2 = hidden_channels

        self.enc1 = nn.Sequential(
            nn.Conv3d(1,  h1, kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(h1, h1, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(h1, h2, kernel_size=3, stride=(2, 1, 1), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(h2, latent_channels, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose3d(
                latent_channels, h2,
                kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1),
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(
                h2, h1,
                kernel_size=3, stride=(2, 1, 1), padding=1, output_padding=(1, 0, 0),
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(
                h1, h1,
                kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1),
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(
                h1, 1,
                kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1), output_padding=(1, 0, 0),
            ),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x3d = x.unsqueeze(1)
        return self.enc2(self.enc1(x3d))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x3d = self.dec2(self.dec1(z))
        x3d = x3d[:, :, :self.in_channels]
        x3d = torch.sigmoid(x3d)
        return x3d.squeeze(1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z     = self.encode(x)
        x_hat = self.decode(z)
        return {"x_hat": x_hat, "z": z}

    @staticmethod
    def compression_ratio_proxy(
        input_shape:  tuple[int, int, int],
        latent_shape: tuple[int, int, int],
    ) -> float:
        c, h, w = input_shape
        cz, hz, wz = latent_shape
        return (c * h * w) / (cz * hz * wz)
