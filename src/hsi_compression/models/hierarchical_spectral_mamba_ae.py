from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck
from einops import rearrange

from hsi_compression.models.blocks import BidirectionalMambaBlock


class ResidualMLPBlock(nn.Module):
    def __init__(self, d_model: int, mlp_hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, mlp_hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.fc2 = nn.Linear(mlp_hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc2(self.drop(self.act(self.fc1(self.norm(x)))))
        return x + y


class SpatialConditionPath(nn.Module):
    """Lightweight spatial context path for affine conditioning of spectral features."""

    def __init__(
        self,
        in_channels: int,
        embed_channels: int,
        context_channels: int,
        target_channels: int,
        use_affine_conditioning: bool = True,
    ):
        super().__init__()
        self.use_affine_conditioning = use_affine_conditioning
        self.conv1 = nn.Conv2d(in_channels, embed_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(
            embed_channels, context_channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.conv3 = nn.Conv2d(
            context_channels, context_channels, kernel_size=3, padding=1, bias=False
        )
        self.act = nn.GELU()

        self.gamma = nn.Conv2d(context_channels, target_channels, kernel_size=1)
        self.beta = (
            nn.Conv2d(context_channels, target_channels, kernel_size=1)
            if use_affine_conditioning
            else None
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if mask is not None and mask.shape[1] > 1:
            mask = mask.amax(dim=1, keepdim=True)

        x = self.act(self.conv1(x))
        if mask is not None:
            x = x * mask

        x = self.act(self.conv2(x))
        if mask is not None:
            mask = F.max_pool2d(mask, kernel_size=2, stride=2)
            x = x * mask

        x = self.act(self.conv3(x))
        if mask is not None:
            x = x * mask

        gamma = self.gamma(x)
        beta = self.beta(x) if self.beta is not None else None
        return gamma, beta


class LearnedSpectralTokenAggregator(nn.Module):
    """Maps a long spectral sequence into a small set of learned summary tokens."""

    def __init__(self, d_model: int, num_summary_tokens: int):
        super().__init__()
        self.num_summary_tokens = num_summary_tokens
        self.query = nn.Parameter(torch.randn(num_summary_tokens, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # tokens: (N, T, D)
        tokens = self.norm(tokens)
        scores = torch.einsum("kd,ntd->nkt", self.query, tokens)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        summary = torch.einsum("nkt,ntd->nkd", attn, tokens)
        return summary, attn


class HierarchicalSpectralMambaAutoencoder(nn.Module):
    """
    Spectral-first HSI autoencoder with a structured multi-token spectral latent.

    Novelty relative to the active spectral Mamba baseline:
    - avoids collapsing the full spectrum to a single pooled token per location
    - learns K spectral summary tokens for every spatial location
    - keeps the entropy model simple in stage 1 so gains can be attributed to latent structure
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
        use_affine_conditioning: bool = True,
        spectral_chunk_size: int | None = 512,
        output_activation: str | None = "sigmoid",
        dropout: float = 0.0,
    ):
        super().__init__()
        if group_size <= 0:
            raise ValueError("group_size must be > 0")
        if num_summary_tokens <= 0:
            raise ValueError("num_summary_tokens must be > 0")

        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.group_size = group_size
        self.spectral_d_model = spectral_d_model
        self.spectral_out_channels = spectral_out_channels
        self.num_summary_tokens = num_summary_tokens
        self.spectral_chunk_size = spectral_chunk_size

        self.c_pad = int(math.ceil(in_channels / group_size) * group_size)
        self.pad_bands = self.c_pad - in_channels
        self.num_tokens = self.c_pad // group_size

        self.token_embed = nn.Linear(group_size, spectral_d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, spectral_d_model))

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
            nn.Conv2d(spectral_out_channels, spectral_out_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

        self.spatial_condition = SpatialConditionPath(
            in_channels=in_channels,
            embed_channels=spatial_embed_channels,
            context_channels=spatial_context_channels,
            target_channels=spectral_out_channels,
            use_affine_conditioning=use_affine_conditioning,
        )

        self.encoder_to_latent = nn.Conv2d(spectral_out_channels, latent_channels, kernel_size=1)
        self.entropy_bottleneck = EntropyBottleneck(latent_channels)

        self.decoder = nn.Sequential(
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
                max(spectral_out_channels // 2, 32),
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(max(spectral_out_channels // 2, 32), in_channels, kernel_size=3, padding=1),
        )

        if output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation in (None, "identity"):
            self.output_activation = nn.Identity()
        else:
            raise ValueError("output_activation must be one of: 'sigmoid', 'identity', None")

    def _spectral_token_mask(self, mask: torch.Tensor | None) -> torch.Tensor | None:
        if mask is None:
            return None

        mask_pix = rearrange(mask, "b c h w -> (b h w) c")
        if mask_pix.shape[1] == 1 and self.num_tokens > 1:
            return mask_pix.expand(-1, self.num_tokens)
        if mask_pix.shape[1] > 1 and self.group_size > 1:
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
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = x.shape
        x_pix = rearrange(x, "b c h w -> (b h w) c")

        if self.pad_bands > 0:
            x_pix = F.pad(x_pix, (0, self.pad_bands), mode="constant", value=0.0)

        tokens = rearrange(x_pix, "n (t g) -> n t g", g=self.group_size)
        h_tok = self.token_embed(tokens) + self.pos_embed

        mask_tok = self._spectral_token_mask(mask)
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

    def encode(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        mask_float = valid_mask.float() if valid_mask is not None else None
        gamma, beta = self.spatial_condition(x, mask=mask_float)

        x_low = F.avg_pool2d(x, kernel_size=2, stride=2)
        mask_low = None
        if mask_float is not None:
            mask_low = F.max_pool2d(mask_float, kernel_size=2, stride=2)

        spec_feat, _ = self._spectral_encode_grid(x_low, mask=mask_low)
        spec_feat = self.spec_downsample(spec_feat)

        fused = spec_feat * (1.0 + gamma)
        if beta is not None:
            fused = fused + beta
        return self.encoder_to_latent(fused)

    def decode(self, z_hat: torch.Tensor) -> torch.Tensor:
        x_hat = self.decoder(z_hat)
        return self.output_activation(x_hat)

    def forward(
        self, x: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        mask_float = valid_mask.float() if valid_mask is not None else None
        gamma, beta = self.spatial_condition(x, mask=mask_float)

        x_low = F.avg_pool2d(x, kernel_size=2, stride=2)
        mask_low = None
        if mask_float is not None:
            mask_low = F.max_pool2d(mask_float, kernel_size=2, stride=2)

        spec_feat, summary_attn = self._spectral_encode_grid(x_low, mask=mask_low)
        spec_feat = self.spec_downsample(spec_feat)

        fused = spec_feat * (1.0 + gamma)
        if beta is not None:
            fused = fused + beta

        z = self.encoder_to_latent(fused)
        z_hat, likelihoods = self.entropy_bottleneck(z)
        x_hat = self.decode(z_hat)

        return {
            "x_hat": x_hat,
            "z": z,
            "z_hat": z_hat,
            "likelihoods": likelihoods,
            "summary_attn": summary_attn,
        }

    def update(self, force: bool = False) -> bool:
        return self.entropy_bottleneck.update(force=force)

    def compress(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> dict:
        z = self.encode(x, valid_mask=valid_mask)
        strings = self.entropy_bottleneck.compress(z)
        return {
            "strings": [strings] if isinstance(strings, bytes) else strings,
            "shape": z.shape[-2:],
            "z_shape": tuple(z.shape),
            "x_shape": tuple(x.shape),
        }

    def decompress(
        self, strings, shape, z_shape=None, valid_mask: torch.Tensor | None = None
    ) -> dict:
        _ = z_shape
        _ = valid_mask
        z_hat = self.entropy_bottleneck.decompress(strings, shape)
        return {
            "x_hat": self.decode(z_hat),
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
