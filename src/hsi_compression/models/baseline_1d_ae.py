import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline1DAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 202,
        latent_channels: int = 16,
        spectral_hidden_channels: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self._latent_channels = latent_channels
        self._latent_len = in_channels // 2 // 2  # 50

        self.spectral_encoder = nn.Sequential(
            nn.Conv1d(1, spectral_hidden_channels, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(kernel_size=2),                        # 202 → 101

            nn.Conv1d(spectral_hidden_channels, latent_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(kernel_size=2),                        # 101 → 50
        )

        self.spectral_decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),                        # 50 → 100
            nn.Conv1d(latent_channels, spectral_hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),                        # 100 → 200
            nn.Conv1d(spectral_hidden_channels, 1, kernel_size=5, padding=2),
        )

        self.spatial_decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        n, c, _, _ = x.shape

        x_ds = F.avg_pool2d(x, kernel_size=4, stride=4)
        _, _, h, w = x_ds.shape

        seq = x_ds.permute(0, 2, 3, 1).reshape(n * h * w, 1, c)
        z   = self.spectral_encoder(seq)           # (N·H·W, lc, latent_len)

        lc, ll = z.shape[1], z.shape[2]
        return z.reshape(n, h, w, lc * ll).permute(0, 3, 1, 2).contiguous()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        n, ch, h, w = z.shape
        lc  = self._latent_channels
        ll  = ch // lc

        z_seq = z.permute(0, 2, 3, 1).reshape(n * h * w, lc, ll)
        x_seq = self.spectral_decoder(z_seq)       # (N·H·W, 1, ~200)

        curr_len = x_seq.shape[2]
        if curr_len < self.in_channels:
            x_seq = F.pad(x_seq, (0, self.in_channels - curr_len))
        elif curr_len > self.in_channels:
            x_seq = x_seq[:, :, :self.in_channels]

        x_seq = x_seq[:, 0, :]                     # (N·H·W, C)
        x_ds  = x_seq.reshape(n, h, w, self.in_channels).permute(0, 3, 1, 2).contiguous()

        return self.spatial_decoder(x_ds)

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