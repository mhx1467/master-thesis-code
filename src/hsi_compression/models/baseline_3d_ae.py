import torch
import torch.nn as nn


class Baseline3DAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 202,
        latent_channels: int = 16,
        hidden_channels: tuple[int, int] = (32, 64),
        spectral_reduced: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.spectral_reduced = spectral_reduced
        h1, h2 = hidden_channels

        # (N, 202, 128, 128) → (N, spectral_reduced, 128, 128)
        self.spectral_encoder_2d = nn.Sequential(
            nn.Conv2d(in_channels, spectral_reduced * 2, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(spectral_reduced * 2, spectral_reduced, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Input: (N, 1, spectral_reduced, 128, 128)
        self.enc3d = nn.Sequential(
            # (N, 1, 32, 128, 128) → (N, h1, 16, 64, 64)
            nn.Conv3d(1, h1, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(h1, h1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # (N, h1, 16, 64, 64) → (N, lc, 8, 32, 32)
            nn.Conv3d(h1, latent_channels, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.dec3d = nn.Sequential(
            # (N, lc, 8, 32, 32) → (N, h1, 16, 64, 64)
            nn.ConvTranspose3d(
                latent_channels,
                h1,
                kernel_size=3,
                stride=(2, 2, 2),
                padding=1,
                output_padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(h1, h1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # (N, h1, 16, 64, 64) → (N, 1, 32, 128, 128)
            nn.ConvTranspose3d(
                h1,
                1,
                kernel_size=3,
                stride=(2, 2, 2),
                padding=1,
                output_padding=1,
            ),
        )

        # (N, spectral_reduced, 128, 128) → (N, 202, 128, 128)
        self.spectral_decoder_2d = nn.Sequential(
            nn.Conv2d(spectral_reduced, spectral_reduced * 2, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(spectral_reduced * 2, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_s = self.spectral_encoder_2d(x)  # (N, S, 128, 128)
        # 3D encoding
        x3d = x_s.unsqueeze(1)  # (N, 1, S, 128, 128)
        return self.enc3d(x3d)  # (N, lc, S/4, 32, 32)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # 3D decoding
        x3d = self.dec3d(z)  # (N, 1, S, 128, 128)

        # Crop or pad spectral dimension if needed
        if x3d.shape[2] != self.spectral_reduced:
            x3d = x3d[:, :, : self.spectral_reduced]

        x_s = x3d.squeeze(1)  # (N, S, 128, 128)
        # Spectral expansion
        return self.spectral_decoder_2d(x_s)  # (N, 202, 128, 128)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return {"x_hat": x_hat, "z": z}

    @staticmethod
    def compression_ratio_proxy(
        input_shape: tuple[int, int, int],
        latent_shape: tuple,
    ) -> float:
        c, h, w = input_shape
        latent_els = 1
        for s in latent_shape:
            latent_els *= s
        return (c * h * w) / latent_els
