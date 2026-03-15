import torch
import torch.nn as nn


class TinyHSIAutoencoder(nn.Module):
    def __init__(self, bands: int = 224, latent_channels: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(bands, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, latent_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, bands, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return {
            "x_hat": x_hat,
            "z": z,
        }