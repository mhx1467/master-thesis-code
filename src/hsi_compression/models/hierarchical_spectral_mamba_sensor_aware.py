from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck
from einops import rearrange

from hsi_compression.constants import FULL_BAND_COUNT, WATER_VAPOR_BANDS
from hsi_compression.models.blocks import BidirectionalMambaBlock
from hsi_compression.models.hierarchical_spectral_mamba_ae import (
    LearnedSpectralTokenAggregator,
    ResidualMLPBlock,
)


def _hyspecnet_clean_positions(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    full = torch.linspace(0.0, 1.0, FULL_BAND_COUNT, device=device, dtype=dtype)
    invalid = set(WATER_VAPOR_BANDS)
    keep = [idx for idx in range(FULL_BAND_COUNT) if idx not in invalid]
    return full[torch.as_tensor(keep, device=device, dtype=torch.long)]


def _default_positions(
    channels: int,
    *,
    preset: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if preset == "hyspecnet_clean_approx" and channels == FULL_BAND_COUNT - len(WATER_VAPOR_BANDS):
        return _hyspecnet_clean_positions(device=device, dtype=dtype)
    if preset not in {"linear_0_1", "hyspecnet_clean_approx"}:
        raise ValueError("wavelength_preset must be one of: linear_0_1, hyspecnet_clean_approx")
    return torch.linspace(0.0, 1.0, channels, device=device, dtype=dtype)


class FourierWavelengthEmbedding(nn.Module):
    """Learned embedding over normalized physical wavelength positions."""

    def __init__(self, embedding_dim: int = 16, num_frequencies: int = 8):
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if num_frequencies < 0:
            raise ValueError("num_frequencies must be non-negative")
        self.embedding_dim = int(embedding_dim)
        self.num_frequencies = int(num_frequencies)
        if num_frequencies:
            frequencies = (2.0 ** torch.arange(num_frequencies, dtype=torch.float32)) * math.pi
        else:
            frequencies = torch.empty(0, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)
        input_dim = 1 + 2 * num_frequencies
        hidden_dim = max(embedding_dim * 2, 16)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, wavelengths: torch.Tensor) -> torch.Tensor:
        wavelengths = wavelengths.reshape(-1, 1).clamp(0.0, 1.0)
        if self.num_frequencies:
            frequencies = self.frequencies.to(device=wavelengths.device, dtype=wavelengths.dtype)
            angles = wavelengths * frequencies.reshape(1, -1)
            features = torch.cat([wavelengths, torch.sin(angles), torch.cos(angles)], dim=-1)
        else:
            features = wavelengths
        return self.net(features)


class WavelengthSensorAdapter(nn.Module):
    """Small wavelength-conditioned affine adapter initialized as identity."""

    def __init__(
        self,
        wavelength_embedding_dim: int,
        hidden_dim: int = 32,
        scale_limit: float = 0.25,
        bias_limit: float = 0.05,
        enabled: bool = True,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if scale_limit < 0.0 or bias_limit < 0.0:
            raise ValueError("adapter limits must be non-negative")
        self.enabled = bool(enabled)
        self.scale_limit = float(scale_limit)
        self.bias_limit = float(bias_limit)
        self.net = nn.Sequential(
            nn.LayerNorm(wavelength_embedding_dim),
            nn.Linear(wavelength_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor, wavelength_embedding: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        params = self.net(wavelength_embedding.to(dtype=x.dtype))
        scale = torch.tanh(params[:, 0]) * self.scale_limit
        bias = torch.tanh(params[:, 1]) * self.bias_limit
        return x * (1.0 + scale.view(1, -1, 1, 1)) + bias.view(1, -1, 1, 1)


class SpectralStatsSpatialConditionPath(nn.Module):
    """Spatial conditioning path that does not depend on the number of spectral bands."""

    def __init__(
        self,
        embed_channels: int,
        context_channels: int,
        target_channels: int,
        use_affine_conditioning: bool = True,
    ):
        super().__init__()
        self.use_affine_conditioning = use_affine_conditioning
        self.conv1 = nn.Conv2d(3, embed_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(
            embed_channels, context_channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.conv3 = nn.Conv2d(
            context_channels, context_channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.act = nn.GELU()
        self.gamma = nn.Conv2d(context_channels, target_channels, kernel_size=1)
        self.beta = (
            nn.Conv2d(context_channels, target_channels, kernel_size=1)
            if use_affine_conditioning
            else None
        )

    def _spectral_stats(
        self, x: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if mask is None:
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True, unbiased=False)
            valid_fraction = torch.ones_like(mean)
            return torch.cat([mean, std, valid_fraction], dim=1), None

        mask_bool = mask.bool()
        if mask_bool.shape[1] == 1:
            mask_bool = mask_bool.expand(-1, x.shape[1], -1, -1)
        mask_f = mask_bool.to(dtype=x.dtype)
        denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (x * mask_f).sum(dim=1, keepdim=True) / denom
        centered = (x - mean) * mask_f
        std = torch.sqrt((centered.square().sum(dim=1, keepdim=True) / denom).clamp_min(0.0))
        valid_fraction = mask_f.mean(dim=1, keepdim=True)
        spatial_mask = valid_fraction.gt(0.0).to(dtype=x.dtype)
        return torch.cat([mean, std, valid_fraction], dim=1), spatial_mask

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        stats, spatial_mask = self._spectral_stats(x, mask)
        x_ctx = self.act(self.conv1(stats))
        if spatial_mask is not None:
            x_ctx = x_ctx * spatial_mask

        x_ctx = self.act(self.conv2(x_ctx))
        if spatial_mask is not None:
            spatial_mask = F.max_pool2d(spatial_mask, kernel_size=2, stride=2)
            x_ctx = x_ctx * spatial_mask

        x_ctx = self.act(self.conv3(x_ctx))
        if spatial_mask is not None:
            spatial_mask = F.max_pool2d(spatial_mask, kernel_size=2, stride=2)
            x_ctx = x_ctx * spatial_mask

        gamma = self.gamma(x_ctx)
        beta = self.beta(x_ctx) if self.beta is not None else None
        return gamma, beta


class DynamicWavelengthBandDecoder(nn.Module):
    """Decode an arbitrary output band set from shared spatial features."""

    def __init__(self, feature_channels: int, wavelength_embedding_dim: int):
        super().__init__()
        self.feature_channels = int(feature_channels)
        hidden_dim = max(wavelength_embedding_dim * 2, feature_channels)
        self.net = nn.Sequential(
            nn.LayerNorm(wavelength_embedding_dim),
            nn.Linear(wavelength_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_channels + 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        wavelength_embedding: torch.Tensor,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        params = self.net(wavelength_embedding.to(dtype=features.dtype))
        weights = params[:, : self.feature_channels] / math.sqrt(float(self.feature_channels))
        bias = params[:, self.feature_channels]
        chunk_size = chunk_size or weights.shape[0]

        chunks = []
        for start in range(0, weights.shape[0], chunk_size):
            end = start + chunk_size
            y = torch.einsum("bdhw,cd->bchw", features, weights[start:end])
            y = y + bias[start:end].view(1, -1, 1, 1)
            chunks.append(y)
        return torch.cat(chunks, dim=1)


class HierarchicalSpectralMambaSensorAwareAutoencoder(nn.Module):
    """
    Sensor-aware hierarchical Mamba autoencoder.

    Unlike ``hierarchical_spectral_mamba_ae``, this variant has no learned parameter whose
    shape depends on the number of input or output bands. Band identity is represented by
    wavelength embeddings, which lets a HySpecNet-pretrained checkpoint load into a model
    instantiated for another sensor grid such as HYPERVIEW2/PRISMA.
    """

    compression_mode = "lossy"
    supports_actual_compression = True

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
        wavelength_embedding_dim: int = 16,
        wavelength_num_frequencies: int = 8,
        wavelength_preset: str = "hyspecnet_clean_approx",
        wavelengths: Sequence[float] | None = None,
        wavelength_min_nm: float | None = None,
        wavelength_max_nm: float | None = None,
        spatial_embed_channels: int = 16,
        spatial_context_channels: int = 64,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        use_spatial_conditioning: bool = True,
        use_affine_conditioning: bool = True,
        use_sensor_adapter: bool = True,
        sensor_adapter_hidden_dim: int = 32,
        sensor_adapter_scale_limit: float = 0.25,
        sensor_adapter_bias_limit: float = 0.05,
        spectral_chunk_size: int | None = 512,
        decoder_band_chunk_size: int | None = None,
        spectral_augmentation: dict | None = None,
        output_activation: str | None = "sigmoid",
        dropout: float = 0.0,
    ):
        super().__init__()
        if group_size <= 0:
            raise ValueError("group_size must be > 0")
        if num_summary_tokens <= 0:
            raise ValueError("num_summary_tokens must be > 0")

        self.in_channels = int(in_channels)
        self.latent_channels = int(latent_channels)
        self.group_size = int(group_size)
        self.spectral_d_model = int(spectral_d_model)
        self.spectral_out_channels = int(spectral_out_channels)
        self.num_summary_tokens = int(num_summary_tokens)
        self.spectral_chunk_size = spectral_chunk_size
        self.decoder_band_chunk_size = decoder_band_chunk_size
        self.use_spatial_conditioning = bool(use_spatial_conditioning)
        self.wavelength_preset = wavelength_preset
        self.wavelength_min_nm = wavelength_min_nm
        self.wavelength_max_nm = wavelength_max_nm
        self.spectral_augmentation = dict(spectral_augmentation or {})

        default_wavelengths = self._init_default_wavelengths(wavelengths)
        self.register_buffer("default_wavelengths", default_wavelengths, persistent=False)

        self.wavelength_embedding = FourierWavelengthEmbedding(
            embedding_dim=wavelength_embedding_dim,
            num_frequencies=wavelength_num_frequencies,
        )
        self.sensor_adapter = WavelengthSensorAdapter(
            wavelength_embedding_dim=wavelength_embedding_dim,
            hidden_dim=sensor_adapter_hidden_dim,
            scale_limit=sensor_adapter_scale_limit,
            bias_limit=sensor_adapter_bias_limit,
            enabled=use_sensor_adapter,
        )

        per_band_feature_dim = 1 + wavelength_embedding_dim + 1
        self.token_embed = nn.Linear(group_size * per_band_feature_dim, spectral_d_model)

        self.local_blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "mamba": BidirectionalMambaBlock(
                            d_model=spectral_d_model,
                            d_state=mamba_d_state,
                            d_conv=mamba_d_conv,
                            expand=mamba_expand,
                            dropout=dropout,
                        ),
                        "mlp": ResidualMLPBlock(
                            d_model=spectral_d_model,
                            mlp_hidden_dim=spectral_mlp_hidden_dim,
                            dropout=dropout,
                        ),
                    }
                )
                for _ in range(num_local_blocks)
            ]
        )
        self.summary_aggregator = LearnedSpectralTokenAggregator(
            d_model=spectral_d_model,
            num_summary_tokens=num_summary_tokens,
        )
        self.global_blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "mamba": BidirectionalMambaBlock(
                            d_model=spectral_d_model,
                            d_state=mamba_d_state,
                            d_conv=mamba_d_conv,
                            expand=mamba_expand,
                            dropout=dropout,
                        ),
                        "mlp": ResidualMLPBlock(
                            d_model=spectral_d_model,
                            mlp_hidden_dim=spectral_mlp_hidden_dim,
                            dropout=dropout,
                        ),
                    }
                )
                for _ in range(num_global_blocks)
            ]
        )

        self.summary_norm = nn.LayerNorm(spectral_d_model)
        self.summary_to_grid = nn.Sequential(
            nn.LayerNorm(num_summary_tokens * spectral_d_model),
            nn.Linear(num_summary_tokens * spectral_d_model, spectral_out_channels),
            nn.GELU(),
        )
        self.spec_downsample = nn.Sequential(
            nn.Conv2d(
                spectral_out_channels, spectral_out_channels, kernel_size=3, stride=2, padding=1
            ),
            nn.GELU(),
        )

        self.spatial_condition = (
            SpectralStatsSpatialConditionPath(
                embed_channels=spatial_embed_channels,
                context_channels=spatial_context_channels,
                target_channels=spectral_out_channels,
                use_affine_conditioning=use_affine_conditioning,
            )
            if use_spatial_conditioning
            else None
        )

        self.encoder_to_latent = nn.Conv2d(spectral_out_channels, latent_channels, kernel_size=1)
        self.entropy_bottleneck = EntropyBottleneck(latent_channels)

        decoder_feature_channels = max(spectral_out_channels // 2, 32)
        self.decoder_features = nn.Sequential(
            nn.Conv2d(latent_channels, spectral_out_channels, kernel_size=1),
            nn.GELU(),
            nn.ConvTranspose2d(
                spectral_out_channels,
                spectral_out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                spectral_out_channels,
                decoder_feature_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GELU(),
        )
        self.band_decoder = DynamicWavelengthBandDecoder(
            feature_channels=decoder_feature_channels,
            wavelength_embedding_dim=wavelength_embedding_dim,
        )

        if output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation in (None, "identity"):
            self.output_activation = nn.Identity()
        else:
            raise ValueError("output_activation must be one of: 'sigmoid', 'identity', None")

    def _init_default_wavelengths(self, wavelengths: Sequence[float] | None) -> torch.Tensor:
        if wavelengths is None:
            return _default_positions(
                self.in_channels,
                preset=self.wavelength_preset,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        values = torch.as_tensor(list(wavelengths), dtype=torch.float32)
        if values.ndim != 1:
            raise ValueError("wavelengths must be a one-dimensional sequence")
        if values.numel() != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} default wavelengths, got {values.numel()}"
            )
        return self._normalize_wavelengths(values)

    def _normalize_wavelengths(self, wavelengths: torch.Tensor) -> torch.Tensor:
        wavelengths = wavelengths.float()
        if self.wavelength_min_nm is not None or self.wavelength_max_nm is not None:
            if self.wavelength_min_nm is None or self.wavelength_max_nm is None:
                raise ValueError(
                    "wavelength_min_nm and wavelength_max_nm must be provided together"
                )
            denom = max(float(self.wavelength_max_nm) - float(self.wavelength_min_nm), 1e-6)
            wavelengths = (wavelengths - float(self.wavelength_min_nm)) / denom
            return wavelengths.clamp(0.0, 1.0)
        if wavelengths.numel() > 0 and (
            wavelengths.min() < -1e-6 or wavelengths.max() > 1.0 + 1e-6
        ):
            denom = (wavelengths.max() - wavelengths.min()).clamp_min(1e-6)
            wavelengths = (wavelengths - wavelengths.min()) / denom
        return wavelengths.clamp(0.0, 1.0)

    def _wavelengths(
        self,
        channels: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        wavelengths: torch.Tensor | Sequence[float] | None = None,
    ) -> torch.Tensor:
        if wavelengths is None:
            if self.default_wavelengths.numel() == channels:
                values = self.default_wavelengths.to(device=device, dtype=dtype)
            else:
                values = _default_positions(
                    channels, preset="linear_0_1", device=device, dtype=dtype
                )
        else:
            values = torch.as_tensor(wavelengths, device=device, dtype=dtype).reshape(-1)
            if values.numel() != channels:
                raise ValueError(f"Expected {channels} wavelengths, got {values.numel()}")
            values = self._normalize_wavelengths(values).to(device=device, dtype=dtype)
        return values

    def _band_mask(self, x: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
        if valid_mask is None:
            return torch.ones_like(x, dtype=torch.bool)
        mask = valid_mask.bool()
        if mask.shape[1] == 1:
            return mask.expand(-1, x.shape[1], -1, -1)
        if mask.shape[1] != x.shape[1]:
            raise ValueError(f"valid_mask has {mask.shape[1]} channels, expected 1 or {x.shape[1]}")
        return mask

    def _augment_encoder_input(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        wavelengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.spectral_augmentation
        if not self.training or not cfg or not cfg.get("enabled", True):
            return x, mask, wavelengths

        x_aug = x
        mask_aug = mask
        wavelengths_aug = wavelengths

        if (
            cfg.get("spectral_blur_probability", 0.0) > 0.0
            and x.shape[1] > 2
            and torch.rand((), device=x.device) < float(cfg["spectral_blur_probability"])
        ):
            prev_band = torch.cat([x_aug[:, :1], x_aug[:, :-1]], dim=1)
            next_band = torch.cat([x_aug[:, 1:], x_aug[:, -1:]], dim=1)
            x_aug = 0.25 * prev_band + 0.5 * x_aug + 0.25 * next_band

        dropout_p = float(cfg.get("band_dropout_probability", 0.0))
        if dropout_p > 0.0 and x.shape[1] > 1:
            drop = torch.rand(x.shape[1], device=x.device) < dropout_p
            min_keep_fraction = float(cfg.get("min_keep_fraction", 0.75))
            max_drop = max(0, int(round(x.shape[1] * (1.0 - min_keep_fraction))))
            if int(drop.sum().item()) > max_drop:
                keep_drop_indices = drop.nonzero(as_tuple=False).flatten()
                order = torch.randperm(keep_drop_indices.numel(), device=x.device)
                drop = torch.zeros_like(drop)
                drop[keep_drop_indices[order[:max_drop]]] = True
            if drop.any():
                x_aug = x_aug.masked_fill(drop.view(1, -1, 1, 1), 0.0)
                mask_aug = mask_aug & (~drop.view(1, -1, 1, 1))

        jitter_std = float(cfg.get("wavelength_jitter_std", 0.0))
        if jitter_std > 0.0:
            wavelengths_aug = (
                wavelengths_aug + torch.randn_like(wavelengths_aug) * jitter_std
            ).clamp(0.0, 1.0)

        scale_std = float(cfg.get("reflectance_scale_std", 0.0))
        if scale_std > 0.0:
            scale = torch.randn(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype) * scale_std
            x_aug = (x_aug * (1.0 + scale)).clamp(0.0, 1.0)

        return x_aug, mask_aug, wavelengths_aug

    def _spectral_token_mask(self, mask: torch.Tensor | None, channels: int) -> torch.Tensor | None:
        if mask is None:
            return None
        mask_pix = rearrange(mask, "b c h w -> (b h w) c")
        pad_bands = int(math.ceil(channels / self.group_size) * self.group_size) - channels
        if pad_bands > 0:
            mask_pix = F.pad(mask_pix, (0, pad_bands), mode="constant", value=0.0)
        if self.group_size > 1:
            return rearrange(mask_pix, "n (t g) -> n t g", g=self.group_size).amax(dim=-1)
        return mask_pix

    def _encode_token_chunk(
        self, h_tok: torch.Tensor, mask_tok: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.local_blocks:
            h_tok = block["mamba"](h_tok)
            h_tok = block["mlp"](h_tok)

        summary, attn = self.summary_aggregator(h_tok, mask=mask_tok)

        for block in self.global_blocks:
            summary = block["mamba"](summary)
            summary = block["mlp"](summary)

        summary = self.summary_norm(summary)
        summary = rearrange(summary, "n k d -> n (k d)")
        feat = self.summary_to_grid(summary)
        return feat, attn

    def _spectral_encode_grid(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        wavelengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, channels, h, w = x.shape
        x_pix = rearrange(x, "b c h w -> (b h w) c")
        mask_pix = rearrange(mask, "b c h w -> (b h w) c")

        wave_emb = self.wavelength_embedding(wavelengths).to(dtype=x.dtype)
        pad_bands = int(math.ceil(channels / self.group_size) * self.group_size) - channels
        if pad_bands > 0:
            x_pix = F.pad(x_pix, (0, pad_bands), mode="constant", value=0.0)
            mask_pix = F.pad(mask_pix, (0, pad_bands), mode="constant", value=0.0)
            wave_emb = F.pad(wave_emb, (0, 0, 0, pad_bands), mode="constant", value=0.0)

        n_pix = x_pix.shape[0]
        wave_features = wave_emb.unsqueeze(0).expand(n_pix, -1, -1)
        band_features = torch.cat(
            [
                x_pix.unsqueeze(-1),
                wave_features,
                mask_pix.to(dtype=x.dtype).unsqueeze(-1),
            ],
            dim=-1,
        )
        tokens = rearrange(band_features, "n (t g) f -> n t (g f)", g=self.group_size).contiguous()
        h_tok = self.token_embed(tokens)
        mask_tok = self._spectral_token_mask(mask, channels)
        chunk_size = self.spectral_chunk_size or h_tok.shape[0]

        feat_chunks = []
        attn_chunks = []
        for start in range(0, h_tok.shape[0], chunk_size):
            end = start + chunk_size
            h_chunk = h_tok[start:end]
            mask_chunk = mask_tok[start:end] if mask_tok is not None else None
            feat_chunk, attn_chunk = self._encode_token_chunk(h_chunk, mask_chunk)
            feat_chunks.append(feat_chunk)
            attn_chunks.append(attn_chunk)

        feat = torch.cat(feat_chunks, dim=0)
        attn = torch.cat(attn_chunks, dim=0)
        feat = rearrange(feat, "(b h w) c -> b c h w", b=b, h=h, w=w)
        return feat, attn

    def encode(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        wavelengths: torch.Tensor | Sequence[float] | None = None,
    ) -> torch.Tensor:
        band_mask = self._band_mask(x, valid_mask)
        wavelengths_t = self._wavelengths(
            x.shape[1], device=x.device, dtype=x.dtype, wavelengths=wavelengths
        )
        wave_emb = self.wavelength_embedding(wavelengths_t)
        x_adapted = self.sensor_adapter(x, wave_emb)
        x_enc, mask_enc, wavelengths_enc = self._augment_encoder_input(
            x_adapted, band_mask, wavelengths_t
        )
        x_enc = x_enc * mask_enc.to(dtype=x_enc.dtype)
        mask_float = mask_enc.to(dtype=x_enc.dtype)

        if self.spatial_condition is not None:
            gamma, beta = self.spatial_condition(x_enc, mask=mask_float)
        else:
            gamma = torch.zeros(
                x.shape[0],
                self.spectral_out_channels,
                x.shape[-2] // 4,
                x.shape[-1] // 4,
                device=x.device,
                dtype=x.dtype,
            )
            beta = None

        x_low = F.avg_pool2d(x_enc, kernel_size=2, stride=2)
        mask_low = F.max_pool2d(mask_float, kernel_size=2, stride=2).gt(0.0)
        spec_feat, _ = self._spectral_encode_grid(x_low, mask_low, wavelengths_enc)
        spec_feat = self.spec_downsample(spec_feat)

        fused = spec_feat * (1.0 + gamma)
        if beta is not None:
            fused = fused + beta
        return self.encoder_to_latent(fused)

    def decode(
        self,
        z_hat: torch.Tensor,
        output_wavelengths: torch.Tensor | Sequence[float] | None = None,
        output_channels: int | None = None,
    ) -> torch.Tensor:
        if output_wavelengths is not None:
            out_channels = len(output_wavelengths)
        else:
            out_channels = int(output_channels or self.in_channels)
        wavelengths_t = self._wavelengths(
            out_channels,
            device=z_hat.device,
            dtype=z_hat.dtype,
            wavelengths=output_wavelengths,
        )
        wave_emb = self.wavelength_embedding(wavelengths_t)
        features = self.decoder_features(z_hat)
        x_hat = self.band_decoder(
            features,
            wave_emb,
            chunk_size=self.decoder_band_chunk_size,
        )
        return self.output_activation(x_hat)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        wavelengths: torch.Tensor | Sequence[float] | None = None,
        output_wavelengths: torch.Tensor | Sequence[float] | None = None,
    ) -> dict[str, torch.Tensor]:
        band_mask = self._band_mask(x, valid_mask)
        wavelengths_t = self._wavelengths(
            x.shape[1], device=x.device, dtype=x.dtype, wavelengths=wavelengths
        )
        wave_emb = self.wavelength_embedding(wavelengths_t)
        x_adapted = self.sensor_adapter(x, wave_emb)
        x_enc, mask_enc, wavelengths_enc = self._augment_encoder_input(
            x_adapted, band_mask, wavelengths_t
        )
        x_enc = x_enc * mask_enc.to(dtype=x_enc.dtype)
        mask_float = mask_enc.to(dtype=x_enc.dtype)

        if self.spatial_condition is not None:
            gamma, beta = self.spatial_condition(x_enc, mask=mask_float)
        else:
            gamma = torch.zeros(
                x.shape[0],
                self.spectral_out_channels,
                x.shape[-2] // 4,
                x.shape[-1] // 4,
                device=x.device,
                dtype=x.dtype,
            )
            beta = None

        x_low = F.avg_pool2d(x_enc, kernel_size=2, stride=2)
        mask_low = F.max_pool2d(mask_float, kernel_size=2, stride=2).gt(0.0)
        spec_feat, summary_attn = self._spectral_encode_grid(x_low, mask_low, wavelengths_enc)
        spec_feat = self.spec_downsample(spec_feat)

        fused = spec_feat * (1.0 + gamma)
        if beta is not None:
            fused = fused + beta

        z = self.encoder_to_latent(fused)
        z_hat, likelihoods = self.entropy_bottleneck(z)
        x_hat = self.decode(
            z_hat,
            output_wavelengths=output_wavelengths,
            output_channels=x.shape[1] if output_wavelengths is None else None,
        )
        return {
            "x_hat": x_hat,
            "z": z,
            "z_hat": z_hat,
            "likelihoods": likelihoods,
            "summary_attn": summary_attn,
        }

    def update(self, force: bool = False) -> bool:
        return self.entropy_bottleneck.update(force=force)

    def compress(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        wavelengths: torch.Tensor | Sequence[float] | None = None,
    ) -> dict:
        z = self.encode(x, valid_mask=valid_mask, wavelengths=wavelengths)
        strings = self.entropy_bottleneck.compress(z)
        return {
            "strings": [strings] if isinstance(strings, bytes) else strings,
            "shape": z.shape[-2:],
            "z_shape": tuple(z.shape),
            "x_shape": tuple(x.shape),
            "output_channels": int(x.shape[1]),
        }

    def decompress(
        self,
        strings,
        shape,
        z_shape=None,
        output_channels: int | None = None,
        output_wavelengths: torch.Tensor | Sequence[float] | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> dict:
        _ = z_shape
        _ = valid_mask
        z_hat = self.entropy_bottleneck.decompress(strings, shape)
        return {
            "x_hat": self.decode(
                z_hat,
                output_wavelengths=output_wavelengths,
                output_channels=output_channels,
            ),
            "z_hat": z_hat,
        }

    @property
    def proxy_bpppc(self) -> float:
        latent_h = 32
        latent_w = 32
        input_h = 128
        input_w = 128
        return (self.latent_channels * latent_h * latent_w) / (self.in_channels * input_h * input_w)

    @property
    def bpppc(self) -> float:
        return self.proxy_bpppc
