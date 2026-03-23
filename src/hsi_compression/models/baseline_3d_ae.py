import torch
import torch.nn as nn


class Baseline3DAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 202,
        latent_channels: int = 16,
        hidden_channels: tuple[int, int] = (32, 64),
        spectral_reduced: int = 32,
        output_activation: str | None = "sigmoid",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.spectral_reduced = spectral_reduced
        self.output_activation = output_activation
        h1, h2 = hidden_channels

        # (N, C, 128, 128) -> (N, S, 128, 128)
        self.spectral_encoder_2d = nn.Sequential(
            nn.Conv2d(in_channels, spectral_reduced * 2, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(spectral_reduced * 2, spectral_reduced, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # (N, 1, S, 128, 128) -> (N, lc, S/4, 32, 32)
        self.enc3d = nn.Sequential(
            nn.Conv3d(1, h1, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(h1, h2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(h2, latent_channels, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # (N, lc, S/4, 32, 32) -> (N, 1, S, 128, 128)
        self.dec3d = nn.Sequential(
            nn.ConvTranspose3d(
                latent_channels,
                h2,
                kernel_size=3,
                stride=(2, 2, 2),
                padding=1,
                output_padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(h2, h1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(
                h1,
                1,
                kernel_size=3,
                stride=(2, 2, 2),
                padding=1,
                output_padding=1,
            ),
        )

        decoder_layers = [
            nn.Conv2d(spectral_reduced, spectral_reduced * 2, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(spectral_reduced * 2, in_channels, kernel_size=1),
        ]
        if output_activation == "sigmoid":
            decoder_layers.append(nn.Sigmoid())
        self.spectral_decoder_2d = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_s = self.spectral_encoder_2d(x)
        x3d = x_s.unsqueeze(1)
        return self.enc3d(x3d)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x3d = self.dec3d(z)

        if x3d.shape[2] > self.spectral_reduced:
            x3d = x3d[:, :, : self.spectral_reduced]
        elif x3d.shape[2] < self.spectral_reduced:
            pad_d = self.spectral_reduced - x3d.shape[2]
            x3d = torch.nn.functional.pad(x3d, (0, 0, 0, 0, 0, pad_d))

        x_s = x3d.squeeze(1)
        return self.spectral_decoder_2d(x_s)

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
