from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

from hsi_compression.models.hierarchical_spectral_mamba_ae import (
    HierarchicalSpectralMambaAutoencoder,
)


def _default_scale_table(
    min_scale: float = 0.11,
    max_scale: float = 256.0,
    levels: int = 64,
) -> list[float]:
    if min_scale <= 0.0:
        raise ValueError("min_scale must be positive")
    if max_scale <= min_scale:
        raise ValueError("max_scale must be greater than min_scale")
    if levels <= 1:
        raise ValueError("levels must be greater than 1")
    log_min = math.log(min_scale)
    log_max = math.log(max_scale)
    return torch.exp(torch.linspace(log_min, log_max, levels)).tolist()


class HierarchicalSpectralMambaHyperpriorAutoencoder(HierarchicalSpectralMambaAutoencoder):
    """Hierarchical spectral Mamba codec with a hyperprior entropy model.

    This variant intentionally lives beside `HierarchicalSpectralMambaAutoencoder`. The inherited
    encoder, spatial conditioning path, and decoder are unchanged, so experiments isolate the effect
    of replacing the simple `EntropyBottleneck` with a side-information hyperprior.
    """

    def __init__(
        self,
        in_channels: int,
        latent_channels: int = 96,
        group_size: int = 4,
        spectral_d_model: int = 64,
        spectral_mlp_hidden_dim: int = 128,
        spectral_out_channels: int = 96,
        num_summary_tokens: int = 4,
        num_local_blocks: int = 2,
        num_global_blocks: int = 1,
        spatial_embed_channels: int = 16,
        spatial_context_channels: int = 64,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        use_spatial_conditioning: bool = True,
        use_affine_conditioning: bool = True,
        spectral_chunk_size: int | None = 512,
        output_activation: str | None = "sigmoid",
        dropout: float = 0.0,
        hyper_channels: int = 96,
        hyper_latent_channels: int = 64,
        scale_bound: float = 0.11,
        scale_table_min: float = 0.11,
        scale_table_max: float = 256.0,
        scale_table_levels: int = 64,
    ):
        super().__init__(
            in_channels=in_channels,
            latent_channels=latent_channels,
            group_size=group_size,
            spectral_d_model=spectral_d_model,
            spectral_mlp_hidden_dim=spectral_mlp_hidden_dim,
            spectral_out_channels=spectral_out_channels,
            num_summary_tokens=num_summary_tokens,
            num_local_blocks=num_local_blocks,
            num_global_blocks=num_global_blocks,
            spatial_embed_channels=spatial_embed_channels,
            spatial_context_channels=spatial_context_channels,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            use_spatial_conditioning=use_spatial_conditioning,
            use_affine_conditioning=use_affine_conditioning,
            spectral_chunk_size=spectral_chunk_size,
            output_activation=output_activation,
            dropout=dropout,
        )
        if hyper_channels <= 0:
            raise ValueError("hyper_channels must be positive")
        if hyper_latent_channels <= 0:
            raise ValueError("hyper_latent_channels must be positive")

        self.hyper_channels = int(hyper_channels)
        self.hyper_latent_channels = int(hyper_latent_channels)
        self.scale_bound = float(scale_bound)
        self.scale_table_min = float(scale_table_min)
        self.scale_table_max = float(scale_table_max)
        self.scale_table_levels = int(scale_table_levels)

        self.hyper_encoder = nn.Sequential(
            nn.Conv2d(latent_channels, hyper_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hyper_channels, hyper_channels, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(hyper_channels, hyper_latent_channels, kernel_size=5, stride=2, padding=2),
        )
        self.hyper_decoder = nn.Sequential(
            nn.ConvTranspose2d(
                hyper_latent_channels,
                hyper_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                hyper_channels,
                hyper_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(hyper_channels, latent_channels * 2, kernel_size=3, padding=1),
        )

        # Side information is coded with an entropy bottleneck. The main latent uses a conditional
        # Gaussian whose means and scales are predicted from the decoded side information.
        self.entropy_bottleneck = EntropyBottleneck(hyper_latent_channels)
        self.gaussian_conditional = GaussianConditional(
            _default_scale_table(
                min_scale=self.scale_table_min,
                max_scale=self.scale_table_max,
                levels=self.scale_table_levels,
            ),
            scale_bound=self.scale_bound,
        )

    def _scale_table(self) -> list[float]:
        return _default_scale_table(
            min_scale=self.scale_table_min,
            max_scale=self.scale_table_max,
            levels=self.scale_table_levels,
        )

    def _hyper_params(
        self, hyper_latent_hat: torch.Tensor, latent_shape: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.hyper_decoder(hyper_latent_hat)
        if tuple(params.shape[-2:]) != tuple(latent_shape):
            params = F.interpolate(params, size=latent_shape, mode="nearest")
        means_hat, scales_hat = params.chunk(2, dim=1)
        scales_hat = F.softplus(scales_hat) + self.scale_bound
        return means_hat, scales_hat

    def forward(
        self, x: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        z = self.encode(x, valid_mask=valid_mask)
        hyper_latent = self.hyper_encoder(z)
        hyper_latent_hat, hyper_likelihoods = self.entropy_bottleneck(hyper_latent)
        means_hat, scales_hat = self._hyper_params(hyper_latent_hat, z.shape[-2:])
        z_hat, z_likelihoods = self.gaussian_conditional(z, scales_hat, means=means_hat)
        x_hat = self.decode(z_hat)
        likelihoods = torch.cat((z_likelihoods.reshape(-1), hyper_likelihoods.reshape(-1)))

        return {
            "x_hat": x_hat,
            "z": z,
            "z_hat": z_hat,
            "hyper_latent": hyper_latent,
            "hyper_latent_hat": hyper_latent_hat,
            "likelihoods": likelihoods,
            "z_likelihoods": z_likelihoods,
            "hyper_likelihoods": hyper_likelihoods,
            "entropy_scales": scales_hat,
            "entropy_means": means_hat,
        }

    def update(self, force: bool = False) -> bool:
        entropy_updated = self.entropy_bottleneck.update(force=force)
        gaussian_updated = self.gaussian_conditional.update_scale_table(
            self._scale_table(), force=force
        )
        return bool(entropy_updated or gaussian_updated)

    def compress(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> dict:
        z = self.encode(x, valid_mask=valid_mask)
        hyper_latent = self.hyper_encoder(z)
        hyper_strings = self.entropy_bottleneck.compress(hyper_latent)
        hyper_latent_hat = self.entropy_bottleneck.decompress(
            hyper_strings, hyper_latent.shape[-2:]
        )
        means_hat, scales_hat = self._hyper_params(hyper_latent_hat, z.shape[-2:])
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        z_strings = self.gaussian_conditional.compress(z, indexes, means=means_hat)
        return {
            "strings": [z_strings, hyper_strings],
            "shape": hyper_latent.shape[-2:],
            "z_shape": tuple(z.shape),
            "x_shape": tuple(x.shape),
        }

    def decompress(
        self, strings, shape, z_shape=None, valid_mask: torch.Tensor | None = None
    ) -> dict:
        del valid_mask
        if not isinstance(strings, (list, tuple)) or len(strings) != 2:
            raise ValueError("Hyperprior decompress expects [z_strings, hyper_strings].")
        z_strings, hyper_strings = strings
        hyper_latent_hat = self.entropy_bottleneck.decompress(hyper_strings, shape)
        if z_shape is None:
            latent_shape = tuple(self.hyper_decoder(hyper_latent_hat).shape[-2:])
        else:
            latent_shape = tuple(z_shape[-2:])
        means_hat, scales_hat = self._hyper_params(hyper_latent_hat, latent_shape)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        z_hat = self.gaussian_conditional.decompress(
            z_strings,
            indexes,
            dtype=means_hat.dtype,
            means=means_hat,
        )
        return {
            "x_hat": self.decode(z_hat),
            "z_hat": z_hat,
            "hyper_latent_hat": hyper_latent_hat,
        }

    @property
    def proxy_bpppc(self) -> float:
        latent_h = 32
        latent_w = 32
        hyper_h = math.ceil(math.ceil(latent_h / 2) / 2)
        hyper_w = math.ceil(math.ceil(latent_w / 2) / 2)
        input_h = 128
        input_w = 128
        latent_slots = self.latent_channels * latent_h * latent_w
        hyper_slots = self.hyper_latent_channels * hyper_h * hyper_w
        return (latent_slots + hyper_slots) / (self.in_channels * input_h * input_w)
