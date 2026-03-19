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

        self.encoder_1d = nn.Sequential(
            nn.Conv1d(1, spectral_hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(spectral_hidden_channels, latent_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        self.decoder_1d = nn.Sequential(
            nn.ConvTranspose1d(latent_channels, spectral_hidden_channels, kernel_size=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(spectral_hidden_channels, 1, kernel_size=3, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        n, c, _, _ = x.shape
        x_ds = F.avg_pool2d(x, kernel_size=4, stride=4)  # (N, C, 32, 32)
        n, c, h, w = x_ds.shape

        seq = x_ds.permute(0, 2, 3, 1).contiguous().view(n * h * w, 1, c)
        z_seq = self.encoder_1d(seq).squeeze(-1)  # (N*H*W, latent_channels)
        z = z_seq.view(n, h, w, -1).permute(0, 3, 1, 2).contiguous()  # (N, latent, 32, 32)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        n, _, h, w = z.shape
        z_seq = z.permute(0, 2, 3, 1).contiguous().view(n * h * w, -1, 1)
        x_seq = self.decoder_1d(z_seq).squeeze(1)  # (N*H*W, C)

        x_ds = x_seq.view(n, h, w, self.in_channels).permute(0, 3, 1, 2).contiguous()
        x_hat = F.interpolate(x_ds, size=(128, 128), mode="bilinear", align_corners=False)
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
