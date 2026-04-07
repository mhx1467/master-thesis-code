import torch
import torch.nn as nn


class SSCNet(nn.Module):
    """
    Reference-style SSCNet adapted from the HySpecNet-11k benchmark code.

    Source architecture:
      SpectralSignalsCompressorNetwork
      La Grassa et al., Remote Sensing 2022
    """

    def __init__(self, in_channels: int = 202, latent_channels: int = 1024):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=3, padding=1),
            nn.PReLU(num_parameters=256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.PReLU(num_parameters=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.PReLU(num_parameters=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.PReLU(num_parameters=512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=512, out_channels=latent_channels, kernel_size=3, padding=1),
            nn.PReLU(num_parameters=latent_channels),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=latent_channels,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.PReLU(num_parameters=512),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2),
            nn.PReLU(num_parameters=256),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2),
            nn.PReLU(num_parameters=256),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2),
            nn.PReLU(num_parameters=256),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid(),
        )

        self.supports_actual_compression = False

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return {
            "x_hat": x_hat,
            "z": z,
            "z_hat": z,
        }

    def compress(self, x: torch.Tensor, **kwargs) -> dict:  # noqa: ARG002
        z = self.encode(x)
        return {
            "latent": z,
            "shape": z.shape[-2:],
            "z_shape": tuple(z.shape),
            "x_shape": tuple(x.shape),
        }

    def decompress(self, latent=None, z_shape=None, **kwargs) -> dict:
        _ = z_shape
        _ = kwargs
        if latent is None:
            raise ValueError("SSCNet.decompress() requires the latent tensor.")
        x_hat = self.decode(latent)
        return {"x_hat": x_hat, "z_hat": latent}

    def update(self, force: bool = False) -> bool:  # noqa: ARG002
        return False

    @property
    def proxy_bpppc(self) -> float:
        spatial_downsamplings = 3
        latent_h_over_input_h = 1 / (2**spatial_downsamplings)
        latent_w_over_input_w = 1 / (2**spatial_downsamplings)
        return (
            self.latent_channels
            / self.in_channels
            * latent_h_over_input_h
            * latent_w_over_input_w
        )

    @property
    def bpppc(self) -> float:
        return self.proxy_bpppc

    @property
    def float32_bpppc(self) -> float:
        return 32.0 * self.proxy_bpppc
