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
        # this is a transformer-style feed-forward residual block.
        # it lets each token refine its own feature vector after mamba mixed sequence context.
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
            context_channels, context_channels, kernel_size=3, stride=2, padding=1, bias=False
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
        # if the mask is per-band, collapse it to one spatial mask before convolutions.
        # the spatial branch only needs to know which pixels are valid.
        if mask is not None and mask.shape[1] > 1:
            mask = mask.amax(dim=1, keepdim=True)

        # the first pointwise convolution mixes spectral bands at the original resolution.
        x = self.act(self.conv1(x))
        if mask is not None:
            # invalid pixels are zeroed so they do not create spatial context.
            x = x * mask

        # two strided convolutions move this branch to the same 32x32 grid as the latent.
        x = self.act(self.conv2(x))
        if mask is not None:
            mask = F.max_pool2d(mask, kernel_size=2, stride=2)
            x = x * mask

        x = self.act(self.conv3(x))
        if mask is not None:
            mask = F.max_pool2d(mask, kernel_size=2, stride=2)
            x = x * mask

        gamma = self.gamma(x)
        beta = self.beta(x) if self.beta is not None else None
        # gamma and beta later scale and shift the spectral features.
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
        # tokens are shaped as sequence batch, spectral token count, feature dimension
        tokens = self.norm(tokens)
        # each learned query produces one summary token by attending over the spectrum
        scores = torch.einsum("kd,ntd->nkt", self.query, tokens)

        if mask is not None:
            mask_fill = torch.finfo(scores.dtype).min
            # masked spectral tokens receive almost zero attention after softmax.
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, mask_fill)

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
        self.use_spatial_conditioning = use_spatial_conditioning

        self.c_pad = int(math.ceil(in_channels / group_size) * group_size)
        self.pad_bands = self.c_pad - in_channels
        self.num_tokens = self.c_pad // group_size

        # if the number of bands is not divisible by group_size, padded bands are zeros.
        # these bands exist only to make reshaping into equal tokens possible.
        # group adjacent bands into short spectral patches before sequence modeling
        self.token_embed = nn.Linear(group_size, spectral_d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, spectral_d_model))

        # local blocks process the full grouped spectrum, for example 51 tokens for 202 bands
        # with group_size=4. this is where neighboring and distant bands can influence each other.
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

        # the aggregator is the actual multi-token bottleneck idea.
        # it learns k different summaries of the same spectrum instead of one average vector.
        self.summary_aggregator = LearnedSpectralTokenAggregator(
            d_model=spectral_d_model,
            num_summary_tokens=num_summary_tokens,
        )

        # after aggregation, global blocks operate only on the k summary tokens.
        # this is cheaper than processing the full spectral sequence again.
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

        # this converts the 64x64 spectral feature grid to the 32x32 latent grid.
        self.spec_downsample = nn.Sequential(
            nn.Conv2d(
                spectral_out_channels, spectral_out_channels, kernel_size=3, stride=2, padding=1
            ),
            nn.GELU(),
        )

        self.spatial_condition = (
            SpatialConditionPath(
                in_channels=in_channels,
                embed_channels=spatial_embed_channels,
                context_channels=spatial_context_channels,
                target_channels=spectral_out_channels,
                use_affine_conditioning=use_affine_conditioning,
            )
            if use_spatial_conditioning
            else None
        )

        self.encoder_to_latent = nn.Conv2d(spectral_out_channels, latent_channels, kernel_size=1)
        # entropy bottleneck is the compressai module that learns a probability model for z.
        # during training it simulates quantization and estimates bitrate through likelihoods.
        self.entropy_bottleneck = EntropyBottleneck(latent_channels)

        # decoder mirrors the latent path by upsampling from 32x32 back to 128x128.
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

        # reduce a band mask to the same token grid used by grouped spectral patches
        mask_pix = rearrange(mask, "b c h w -> (b h w) c")
        if self.pad_bands > 0 and mask_pix.shape[1] > 1:
            mask_pix = F.pad(mask_pix, (0, self.pad_bands), mode="constant", value=0.0)
        if mask_pix.shape[1] == 1 and self.num_tokens > 1:
            return mask_pix.expand(-1, self.num_tokens)
        if mask_pix.shape[1] > 1 and self.group_size > 1:
            return rearrange(mask_pix, "n (t g) -> n t g", g=self.group_size).amax(dim=-1)
        return mask_pix

    def _encode_token_chunk(
        self, h_tok: torch.Tensor, mask_tok: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # local blocks model dependencies across the detailed spectral patch sequence
        for block in self.local_blocks:
            h_tok = block["mamba"](h_tok)
            h_tok = block["mlp"](h_tok)

        summary, attn = self.summary_aggregator(h_tok, mask=mask_tok)

        # global blocks let the learned summary tokens exchange information
        for block in self.global_blocks:
            summary = block["mamba"](summary)
            summary = block["mlp"](summary)

        summary = self.summary_norm(summary)
        # flatten k summary tokens into one feature vector before returning to image grid form.
        summary = rearrange(summary, "n k d -> n (k d)")
        feat = self.summary_to_grid(summary)
        return feat, attn

    def _spectral_encode_grid(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = x.shape
        # every spatial location is treated as one spectral sequence
        x_pix = rearrange(x, "b c h w -> (b h w) c")

        if self.pad_bands > 0:
            # padding only makes the channel count divisible by the spectral group size
            x_pix = F.pad(x_pix, (0, self.pad_bands), mode="constant", value=0.0)

        tokens = rearrange(x_pix, "n (t g) -> n t g", g=self.group_size)
        # each small group of raw bands is projected to d_model features, then position
        # embeddings tell the model where this group lies in the spectrum.
        h_tok = self.token_embed(tokens) + self.pos_embed

        mask_tok = self._spectral_token_mask(mask)
        chunk_size = self.spectral_chunk_size or h_tok.shape[0]

        # chunking keeps memory bounded while preserving independent per pixel sequences
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
        if self.spatial_condition is not None:
            gamma, beta = self.spatial_condition(x, mask=mask_float)
        else:
            # when spatial conditioning is disabled, use neutral affine parameters.
            gamma = torch.zeros(
                x.shape[0],
                self.spectral_out_channels,
                x.shape[-2] // 4,
                x.shape[-1] // 4,
                device=x.device,
                dtype=x.dtype,
            )
            beta = None

        x_low = F.avg_pool2d(x, kernel_size=2, stride=2)
        mask_low = None
        if mask_float is not None:
            # max pooling marks a low-resolution pixel valid if any source pixel was valid.
            mask_low = F.max_pool2d(mask_float, kernel_size=2, stride=2)

        spec_feat, _ = self._spectral_encode_grid(x_low, mask=mask_low)
        spec_feat = self.spec_downsample(spec_feat)

        # spatial conditioning scales and shifts spectral features at the latent grid
        fused = spec_feat * (1.0 + gamma)
        if beta is not None:
            fused = fused + beta
        return self.encoder_to_latent(fused)

    def decode(self, z_hat: torch.Tensor) -> torch.Tensor:
        # z_hat is already quantized or simulated as quantized by the entropy bottleneck.
        x_hat = self.decoder(z_hat)
        return self.output_activation(x_hat)

    def forward(
        self, x: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        mask_float = valid_mask.float() if valid_mask is not None else None
        if self.spatial_condition is not None:
            gamma, beta = self.spatial_condition(x, mask=mask_float)
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

        x_low = F.avg_pool2d(x, kernel_size=2, stride=2)
        mask_low = None
        if mask_float is not None:
            mask_low = F.max_pool2d(mask_float, kernel_size=2, stride=2)

        spec_feat, summary_attn = self._spectral_encode_grid(x_low, mask=mask_low)
        spec_feat = self.spec_downsample(spec_feat)

        # fusion combines spectral summaries with spatial context before entropy coding.
        fused = spec_feat * (1.0 + gamma)
        if beta is not None:
            fused = fused + beta

        z = self.encoder_to_latent(fused)
        # likelihoods are later converted into a bitrate term in the loss and evaluation.
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
