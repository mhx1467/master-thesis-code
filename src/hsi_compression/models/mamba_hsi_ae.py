import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError("Install mamba-ssm: pip install mamba-ssm")


class MambaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model)

    def forward(self, x):
        # x: (B, L, D)
        return x + self.mamba(self.norm(x))


class MambaHSIAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels=224,
        latent_channels=16,
        d_model=64,
        num_layers=2,
    ):
        super().__init__()

        self.in_channels = in_channels

        # 1. Spatial encoder
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=4, stride=2, padding=1),  # 128x64x64
            nn.ReLU(),
            nn.Conv2d(128, in_channels, kernel_size=4, stride=2, padding=1),  # Cx32x32
            nn.ReLU(),
        )

        # 2. Spectral projection
        self.input_proj = nn.Linear(1, d_model)

        # 3. Mamba layers
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model) for _ in range(num_layers)
        ])

        # 4. Output projection to latent channels
        self.output_proj = nn.Linear(d_model, 1)

        # 5. Reduce spectral dim to latent channels
        self.channel_reduction = nn.Conv2d(in_channels, latent_channels, kernel_size=1)

        # 6. Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        N, C, H, W = x.shape

        # 1. Spatial encode
        x = self.spatial_encoder(x)  # (N, C, 32, 32)
        _, _, H2, W2 = x.shape

        # 2. reshape to sequences
        x = x.permute(0, 2, 3, 1)           # (N, H, W, C)
        x = x.reshape(-1, C, 1)             # (N*H*W, C, 1)

        # 3. project
        x = self.input_proj(x)              # (B, C, D)

        # 4. Mamba
        for layer in self.mamba_layers:
            x = layer(x)

        # 5. back to scalar
        x = self.output_proj(x)             # (B, C, 1)

        # 6. reshape back
        x = x.reshape(N, H2, W2, C)
        x = x.permute(0, 3, 1, 2)           # (N, C, H, W)

        # 7. channel reduction → latent
        z = self.channel_reduction(x)

        # 8. decode
        x_hat = self.decoder(z)

        return x_hat