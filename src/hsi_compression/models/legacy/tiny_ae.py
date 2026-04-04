import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck


class TinyHSIAutoencoder(nn.Module):
    def __init__(self, bands: int = 224, latent_channels: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(bands, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, latent_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.entropy_bottleneck = EntropyBottleneck(latent_channels)
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, bands, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        z_hat, likelihoods = self.entropy_bottleneck(z)
        x_hat = self.decoder(z_hat)
        return {
            "x_hat": x_hat,
            "z": z,
            "z_hat": z_hat,
            "likelihoods": likelihoods,
        }
