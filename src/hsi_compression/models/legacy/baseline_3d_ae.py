import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck


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
        self.latent_channels = latent_channels
        self.spectral_reduced = spectral_reduced
        self.output_activation = output_activation
        h1, h2 = hidden_channels

        self.spectral_encoder_2d = nn.Sequential(
            nn.Conv2d(in_channels, spectral_reduced * 2, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(spectral_reduced * 2, spectral_reduced, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.enc3d = nn.Sequential(
            nn.Conv3d(1, h1, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(h1, h2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(h2, latent_channels, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

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

        self.entropy_bottleneck = EntropyBottleneck(latent_channels * (spectral_reduced // 4))

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
        n, c, d, h, w = z.shape
        z_4d = z.view(n, c * d, h, w)
        z_hat_4d, likelihoods = self.entropy_bottleneck(z_4d)
        z_hat = z_hat_4d.view(n, c, d, h, w)
        x_hat = self.decode(z_hat)
        return {"x_hat": x_hat, "z": z, "z_hat": z_hat, "likelihoods": likelihoods}

    def update(self, force: bool = False) -> bool:
        return self.entropy_bottleneck.update(force=force)

    def compress(self, x: torch.Tensor, **kwargs) -> dict:  # noqa: ARG002
        z = self.encode(x)
        n, c, d, h, w = z.shape
        z_4d = z.view(n, c * d, h, w)
        strings = self.entropy_bottleneck.compress(z_4d)
        return {
            "strings": strings,
            "shape": (h, w),
            "z_shape": tuple(z.shape),
            "x_shape": tuple(x.shape),
        }

    def decompress(self, strings, shape, z_shape=None, **kwargs) -> dict:  # noqa: ARG002
        if z_shape is None:
            raise ValueError("z_shape is required for Baseline3DAutoencoder.decompress()")

        z_hat_4d = self.entropy_bottleneck.decompress(strings, shape)
        z_hat = z_hat_4d.view(z_shape)
        x_hat = self.decode(z_hat)
        return {"x_hat": x_hat, "z_hat": z_hat}

    @property
    def proxy_bpppc(self) -> float:
        latent_h = 32
        latent_w = 32
        input_h = 128
        input_w = 128
        latent_d = self.spectral_reduced // 4
        return (self.latent_channels * latent_d * latent_h * latent_w) / (
            self.in_channels * input_h * input_w
        )

    @property
    def bpppc(self) -> float:
        return self.proxy_bpppc
