from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from hsi_compression.models.blocks import (
    BidirectionalMambaBlock,
    QuantizationProxy,
    SpatialConditioning,
    SpatialContextEncoder,
    SpectralAttentionPooling,
    SpectralFirstDecoder,
    SpectralPreservingDownsample,
    SpectralRefinementBlock,
)


class SpectralMambaBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        spectral_d_model: int = 32,
        spectral_out_channels: int = 64,
        num_blocks: int = 2,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.input_proj = nn.Linear(1, spectral_d_model)
        self.blocks = nn.ModuleList(
            [
                BidirectionalMambaBlock(
                    d_model=spectral_d_model,
                    d_state=mamba_d_state,
                    d_conv=mamba_d_conv,
                    expand=mamba_expand,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.pool = SpectralAttentionPooling(spectral_d_model, spectral_out_channels)

    def forward(self, x_spec: torch.Tensor) -> torch.Tensor:
        # (B, C, 32, 32) -> (B*32*32, C, 1)
        seq = rearrange(x_spec, "b c h w -> (b h w) c 1")
        seq = self.input_proj(seq)
        for block in self.blocks:
            seq = block(seq)
        pooled = self.pool(seq)
        return rearrange(pooled, "(b h w) f -> b f h w", h=32, w=32)


class SpectralFirstMambaAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 202,
        latent_channels: int = 8,
        spectral_d_model: int = 32,
        spectral_out_channels: int = 64,
        spatial_embed_channels: int = 16,
        spatial_context_channels: int = 64,
        num_spectral_blocks: int = 2,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        use_affine_conditioning: bool = False,
        output_activation: str | None = "sigmoid",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels

        self.spectral_downsample = SpectralPreservingDownsample(in_channels=in_channels)
        self.spectral_backbone = SpectralMambaBackbone(
            in_channels=in_channels,
            spectral_d_model=spectral_d_model,
            spectral_out_channels=spectral_out_channels,
            num_blocks=num_spectral_blocks,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            dropout=dropout,
        )
        self.spatial_context = SpatialContextEncoder(
            in_channels=in_channels,
            embed_channels=spatial_embed_channels,
            context_channels=spatial_context_channels,
        )
        self.conditioning = SpatialConditioning(
            channels=spectral_out_channels,
            use_affine_bias=use_affine_conditioning,
        )
        self.late_bottleneck = nn.Conv2d(
            spectral_out_channels, latent_channels, kernel_size=1, bias=True
        )
        self.quantization = QuantizationProxy()
        self.decoder = SpectralFirstDecoder(
            latent_channels=latent_channels,
            out_channels=in_channels,
            hidden_channels=spectral_out_channels,
        )
        self.spectral_refinement = SpectralRefinementBlock(in_channels=in_channels)

        if output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation in (None, "identity"):
            self.output_activation = nn.Identity()
        else:
            raise ValueError("output_activation must be one of: 'sigmoid', 'identity', None")

    def encode_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_spec = self.spectral_downsample(x)
        f_spec = self.spectral_backbone(x_spec)
        f_spat = self.spatial_context(x)
        f_joint = self.conditioning(f_spec, f_spat)
        return f_spec, f_spat, f_joint

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, _, f_joint = self.encode_features(x)
        return self.late_bottleneck(f_joint)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z_q = self.quantization(z)
        x_pre = self.decoder(z_q)
        x_refined = self.spectral_refinement(x_pre)
        return self.output_activation(x_refined)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        f_spec, f_spat, f_joint = self.encode_features(x)
        z = self.late_bottleneck(f_joint)
        z_q = self.quantization(z)
        x_pre = self.decoder(z_q)
        x_refined = self.spectral_refinement(x_pre)
        x_hat = self.output_activation(x_refined)
        return {
            "x_hat": x_hat,
            "z": z,
            "z_q": z_q,
            "f_spec": f_spec,
            "f_spat": f_spat,
            "f_joint": f_joint,
        }

    @staticmethod
    def compression_ratio_proxy(
        input_shape: tuple[int, int, int],
        latent_shape: tuple[int, int, int],
    ) -> float:
        c, h, w = input_shape
        cz, hz, wz = latent_shape
        return (c * h * w) / (cz * hz * wz)
