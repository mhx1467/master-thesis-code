import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck


class Baseline3DPatchAutoencoder(nn.Module):
    """True 3D patch baseline without a preceding 2D spectral projection."""

    def __init__(
        self,
        in_channels: int = 202,
        latent_channels: int = 16,
        hidden_channels: tuple[int, int] = (32, 64),
        output_activation: str | None = "sigmoid",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels
        self._latent_spectral_len = math.ceil(math.ceil(in_channels / 2) / 2)

        h1, h2 = hidden_channels
        self.encoder = nn.Sequential(
            nn.Conv3d(1, h1, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(h1, h2, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(h2, latent_channels, kernel_size=3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                latent_channels,
                h2,
                kernel_size=3,
                stride=(2, 2, 2),
                padding=1,
                output_padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(
                h2,
                h1,
                kernel_size=3,
                stride=(2, 2, 2),
                padding=1,
                output_padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(h1, 1, kernel_size=3, padding=1),
        )
        self.entropy_bottleneck = EntropyBottleneck(latent_channels * self._latent_spectral_len)

        if output_activation == "sigmoid":
            self.output_head = nn.Sigmoid()
        elif output_activation in (None, "identity"):
            self.output_head = nn.Identity()
        else:
            raise ValueError("output_activation must be one of: 'sigmoid', 'identity', None")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.unsqueeze(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x3d = self.decoder(z)
        if x3d.shape[2] > self.in_channels:
            x3d = x3d[:, :, : self.in_channels]
        elif x3d.shape[2] < self.in_channels:
            x3d = F.pad(x3d, (0, 0, 0, 0, 0, self.in_channels - x3d.shape[2]))
        x_hat = x3d.squeeze(1)
        return self.output_head(x_hat)

    def _to_4d(self, z: torch.Tensor) -> torch.Tensor:
        n, c, d, h, w = z.shape
        return z.view(n, c * d, h, w)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(x)
        z_4d = self._to_4d(z)
        z_hat_4d, likelihoods = self.entropy_bottleneck(z_4d)
        z_hat = z_hat_4d.view_as(z)
        x_hat = self.decode(z_hat)
        return {"x_hat": x_hat, "z": z, "z_hat": z_hat, "likelihoods": likelihoods}

    def update(self, force: bool = False) -> bool:
        return self.entropy_bottleneck.update(force=force)

    def compress(self, x: torch.Tensor, **kwargs) -> dict:  # noqa: ARG002
        z = self.encode(x)
        z_4d = self._to_4d(z)
        strings = self.entropy_bottleneck.compress(z_4d)
        return {
            "strings": strings,
            "shape": z_4d.shape[-2:],
            "z_shape": tuple(z.shape),
            "x_shape": tuple(x.shape),
        }

    def decompress(self, strings, shape, z_shape=None, **kwargs) -> dict:  # noqa: ARG002
        if z_shape is None:
            raise ValueError("z_shape is required for Baseline3DPatchAutoencoder.decompress()")
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
        return (self.latent_channels * self._latent_spectral_len * latent_h * latent_w) / (
            self.in_channels * input_h * input_w
        )

    @property
    def bpppc(self) -> float:
        return self.proxy_bpppc
