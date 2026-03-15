import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    """
    Removes extra right-side padding to preserve causal behavior.
    """
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)


class TCN1d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        num_layers: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = []
        in_ch = input_channels

        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_ch = hidden_channels

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ConvBlock2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TCNHSIAutoencoder(nn.Module):
    """
    HSI autoencoder with:
    - spatial Conv2d encoder
    - spectral TCN operating over channel sequences
    - spatial decoder

    Input:
        (N, in_channels, 128, 128)

    Latent:
        (N, latent_channels, 32, 32)
    """
    def __init__(
        self,
        in_channels: int = 224,
        encoder_channels: tuple[int, int] = (128, 64),
        latent_channels: int = 8,
        tcn_hidden_channels: int = 64,
        tcn_num_layers: int = 4,
        tcn_kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        e1, e2 = encoder_channels

        # Spatial encoder
        self.enc_block1 = ConvBlock2d(in_channels, e1)
        self.down1 = nn.Conv2d(e1, e1, kernel_size=4, stride=2, padding=1)   # 128 -> 64

        self.enc_block2 = ConvBlock2d(e1, e2)
        self.down2 = nn.Conv2d(e2, e2, kernel_size=4, stride=2, padding=1)   # 64 -> 32

        # Project channels before TCN
        self.pre_tcn_proj = nn.Conv2d(e2, latent_channels, kernel_size=1)

        # Spectral TCN over channel dimension
        self.tcn = TCN1d(
            input_channels=1,
            hidden_channels=tcn_hidden_channels,
            num_layers=tcn_num_layers,
            kernel_size=tcn_kernel_size,
            dropout=dropout,
        )

        self.post_tcn_proj = nn.Conv1d(
            in_channels=tcn_hidden_channels,
            out_channels=1,
            kernel_size=1,
        )

        # Spatial decoder
        self.up1 = nn.ConvTranspose2d(latent_channels, e2, kernel_size=4, stride=2, padding=1)  # 32 -> 64
        self.dec_block1 = ConvBlock2d(e2, e2)

        self.up2 = nn.ConvTranspose2d(e2, e1, kernel_size=4, stride=2, padding=1)  # 64 -> 128
        self.dec_block2 = ConvBlock2d(e1, e1)

        self.out_conv = nn.Conv2d(e1, in_channels, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc_block1(x)
        x = self.down1(x)
        x = self.enc_block2(x)
        x = self.down2(x)
        z = self.pre_tcn_proj(x)  # (N, latent_channels, 32, 32)
        return z

    def spectral_tcn(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply TCN over channel sequences independently for each spatial location.

        Input:
            z: (N, C_latent, H, W)

        Process:
            each (h, w) location has a sequence of length C_latent
        """
        n, c, h, w = z.shape

        # (N, C, H, W) -> (N, H, W, C) -> (N*H*W, 1, C)
        seq = z.permute(0, 2, 3, 1).contiguous().view(n * h * w, 1, c)

        seq = self.tcn(seq)
        seq = self.post_tcn_proj(seq)  # (N*H*W, 1, C)

        # back to (N, C, H, W)
        seq = seq.view(n, h, w, c).permute(0, 3, 1, 2).contiguous()
        return seq

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.up1(z)
        x = self.dec_block1(x)
        x = self.up2(x)
        x = self.dec_block2(x)
        x_hat = self.out_conv(x)
        return x_hat

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(x)
        z_tcn = self.spectral_tcn(z)
        x_hat = self.decode(z_tcn)
        return {
            "x_hat": x_hat,
            "z": z_tcn,
        }

    @staticmethod
    def compression_ratio_proxy(
        input_shape: tuple[int, int, int],
        latent_shape: tuple[int, int, int],
    ) -> float:
        c, h, w = input_shape
        cz, hz, wz = latent_shape
        return (c * h * w) / (cz * hz * wz)