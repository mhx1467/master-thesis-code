import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck


class Baseline1DPixelAutoencoder(nn.Module):
    """
    Pure spectral baseline: each pixel spectrum is compressed independently.

    This model intentionally avoids spatial downsampling so it can serve as a
    clean point of comparison against spatial or spectral-spatial models.
    """

    def __init__(
        self,
        in_channels: int = 202,
        latent_channels: int = 16,
        hidden_channels: int = 64,
        output_activation: str | None = "sigmoid",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels
        self._latent_len = in_channels // 2 // 2
        self.entropy_bottleneck = EntropyBottleneck(latent_channels * self._latent_len)

        self.encoder = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(hidden_channels, latent_channels, kernel_size=3, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(latent_channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_channels, 1, kernel_size=5, padding=2),
        )

        if output_activation == "sigmoid":
            self.output_head = nn.Sigmoid()
        elif output_activation in (None, "identity"):
            self.output_head = nn.Identity()
        else:
            raise ValueError("output_activation must be one of: 'sigmoid', 'identity', None")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        x_seq = x.permute(0, 2, 3, 1).reshape(n * h * w, 1, c)
        z_seq = self.encoder(x_seq)
        _, lc, ll = z_seq.shape
        return z_seq.reshape(n, h, w, lc * ll).permute(0, 3, 1, 2).contiguous()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        n, ch, h, w = z.shape
        ll = ch // self.latent_channels
        z_seq = z.permute(0, 2, 3, 1).reshape(n * h * w, self.latent_channels, ll)
        x_seq = self.decoder(z_seq)

        curr_len = x_seq.shape[-1]
        if curr_len < self.in_channels:
            x_seq = F.pad(x_seq, (0, self.in_channels - curr_len))
        elif curr_len > self.in_channels:
            x_seq = x_seq[..., : self.in_channels]

        x_seq = self.output_head(x_seq)
        x_seq = x_seq[:, 0, :]
        return x_seq.reshape(n, h, w, self.in_channels).permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(x)
        z_hat, likelihoods = self.entropy_bottleneck(z)
        x_hat = self.decode(z_hat)
        return {"x_hat": x_hat, "z": z, "z_hat": z_hat, "likelihoods": likelihoods}

    def update(self, force: bool = False) -> bool:
        return self.entropy_bottleneck.update(force=force)

    def compress(self, x: torch.Tensor, **kwargs) -> dict:  # noqa: ARG002
        z = self.encode(x)
        strings = self.entropy_bottleneck.compress(z)
        return {
            "strings": strings,
            "shape": z.shape[-2:],
            "z_shape": tuple(z.shape),
            "x_shape": tuple(x.shape),
        }

    def decompress(self, strings, shape, **kwargs) -> dict:  # noqa: ARG002
        z_hat = self.entropy_bottleneck.decompress(strings, shape)
        x_hat = self.decode(z_hat)
        return {"x_hat": x_hat, "z_hat": z_hat}

    @property
    def proxy_bpppc(self) -> float:
        latent_h = 128
        latent_w = 128
        input_h = 128
        input_w = 128
        latent_channels_total = self.latent_channels * self._latent_len
        return (latent_channels_total * latent_h * latent_w) / (
            self.in_channels * input_h * input_w
        )

    @property
    def bpppc(self) -> float:
        return self.proxy_bpppc
