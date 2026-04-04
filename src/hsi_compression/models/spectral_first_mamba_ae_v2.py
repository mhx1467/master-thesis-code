from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck
from einops import rearrange

from hsi_compression.models.blocks import BidirectionalMambaBlock


class SpectralTokenAttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1, bias=False)

    def forward(
        self, h: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # h: (N, T, D)
        scores = self.score(h).squeeze(-1)  # (N, T)

        if mask is not None:
            # negative weighting of background pixels ensures that after softmax attention = 0.0
            scores = scores.masked_fill(mask == 0, -1e9)

        alpha = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("nt,ntd->nd", alpha, h)
        return pooled, alpha


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


class ResidualSpectralRefinementBlock(nn.Module):
    """Local spectral correction applied on reconstructed spectra."""

    def __init__(self, hidden_channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C)
        return x + self.net(x.unsqueeze(1)).squeeze(1)


class SpectralRefinementStack(nn.Module):
    def __init__(self, depth: int = 3, hidden_channels: int = 16):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ResidualSpectralRefinementBlock(hidden_channels=hidden_channels) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class SpatialConditionPath(nn.Module):
    """Spatial context extractor that respects the valid mask (doesn't blur background)."""

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
        self.conv4 = nn.Conv2d(
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
        # make sure mask is float and has same spatial dimensions as x
        if mask is not None and mask.shape[1] > 1:
            mask = mask.amax(dim=1, keepdim=True)

        x = self.act(self.conv1(x))
        if mask is not None:
            x = x * mask

        x = self.conv2(x)
        if mask is not None:
            mask = F.max_pool2d(mask, kernel_size=2, stride=2)
        x = self.act(x)
        if mask is not None:
            x = x * mask

        x = self.conv3(x)
        if mask is not None:
            mask = F.max_pool2d(mask, kernel_size=2, stride=2)
        x = self.act(x)
        if mask is not None:
            x = x * mask

        x = self.act(self.conv4(x))
        if mask is not None:
            x = x * mask

        gamma = self.gamma(x)
        beta = self.beta(x) if self.beta is not None else None
        return gamma, beta


class SpectralFirstMambaAutoencoderV2(nn.Module):
    """
    Conservative v2 upgrade over spectral_first_mamba_ae.

    Key changes:
    - 3 Bi-Mamba spectral blocks by default
    - affine spatial conditioning: spec * (1 + gamma) + beta
    - deeper residual spectral refinement stack on output spectra
    - same latent grid resolution (32x32 for 128x128 inputs) and same bpppc proxy as v1
    """

    def __init__(
        self,
        in_channels: int,
        latent_channels: int = 96,
        group_size: int = 1,
        spectral_d_model: int = 64,
        spectral_mlp_hidden_dim: int = 128,
        spectral_out_channels: int = 96,
        spatial_embed_channels: int = 16,
        spatial_context_channels: int = 64,
        num_spectral_blocks: int = 3,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        pooling: str = "attention",
        use_affine_conditioning: bool = True,
        refinement_depth: int = 3,
        refinement_hidden_channels: int = 16,
        output_activation: str | None = "sigmoid",
        dropout: float = 0.0,
    ):
        super().__init__()
        if group_size <= 0:
            raise ValueError("group_size must be > 0")
        if pooling not in {"attention", "mean"}:
            raise ValueError("pooling must be 'attention' or 'mean'")

        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.group_size = group_size
        self.spectral_d_model = spectral_d_model
        self.spectral_out_channels = spectral_out_channels
        self.pooling = pooling
        self.c_pad = int(math.ceil(in_channels / group_size) * group_size)
        self.pad_bands = self.c_pad - in_channels
        self.num_tokens = self.c_pad // group_size

        self.token_embed = nn.Linear(group_size, spectral_d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, spectral_d_model))
        self.spectral_blocks = nn.ModuleList(
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
                for _ in range(num_spectral_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(spectral_d_model)
        self.attn_pool = (
            SpectralTokenAttentionPooling(spectral_d_model) if pooling == "attention" else None
        )
        self.spectral_proj = nn.Sequential(
            nn.LayerNorm(spectral_d_model),
            nn.Linear(spectral_d_model, spectral_out_channels),
            nn.GELU(),
        )
        self.downsample_spec = nn.Conv2d(
            spectral_out_channels, spectral_out_channels, kernel_size=2, stride=2
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

        self.spectral_refinement = SpectralRefinementStack(
            depth=refinement_depth,
            hidden_channels=refinement_hidden_channels,
        )

        if output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation in (None, "identity"):
            self.output_activation = nn.Identity()
        else:
            raise ValueError("output_activation must be one of: 'sigmoid', 'identity', None")

    def _spectral_encode_grid(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, c, h, w = x.shape
        x_pix = rearrange(x, "b c h w -> (b h w) c")

        if self.pad_bands > 0:
            x_pix = F.pad(x_pix, (0, self.pad_bands), mode="constant", value=0.0)

        tokens = rearrange(x_pix, "n (t g) -> n t g", g=self.group_size)
        h_tok = self.token_embed(tokens) + self.pos_embed

        for block in self.spectral_blocks:
            h_tok = block["mamba"](h_tok)
            h_tok = block["mlp"](h_tok)

        h_tok = self.final_norm(h_tok)
        alpha = None

        # prepare mask for spectral tokens
        mask_pix = None
        if mask is not None:
            mask_pix = rearrange(mask, "b c h w -> (b h w) c")
            if mask_pix.shape[1] == 1 and self.num_tokens > 1:
                mask_pix = mask_pix.expand(-1, self.num_tokens)
            elif mask_pix.shape[1] > 1 and self.group_size > 1:
                mask_pix = rearrange(mask_pix, "n (t g) -> n t g", g=self.group_size).amax(dim=-1)

        if self.attn_pool is not None:
            pooled, alpha = self.attn_pool(h_tok, mask=mask_pix)
        else:
            if mask_pix is not None:
                h_tok = h_tok * mask_pix.unsqueeze(-1)
                valid_counts = mask_pix.sum(dim=1, keepdim=True).clamp_min(1.0)
                pooled = h_tok.sum(dim=1) / valid_counts
            else:
                pooled = h_tok.mean(dim=1)

        feat = self.spectral_proj(pooled)
        feat = rearrange(feat, "(b h w) c -> b c h w", b=b, h=h, w=w)
        return feat, alpha

    def _refine_output(self, x_hat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x_hat.shape
        x_pix = rearrange(x_hat, "b c h w -> (b h w) c")
        x_pix = self.spectral_refinement(x_pix)
        return rearrange(x_pix, "(b h w) c -> b c h w", b=b, h=h, w=w)

    def encode(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        mask_float = valid_mask.float() if valid_mask is not None else None
        gamma, beta = self.spatial_condition(x, mask=mask_float)

        x_low = F.avg_pool2d(x, kernel_size=2, stride=2)
        mask_low = None
        if mask_float is not None:
            mask_low = F.max_pool2d(mask_float, kernel_size=2, stride=2)

        spec_feat, _ = self._spectral_encode_grid(x_low, mask=mask_low)

        spec_feat = self.downsample_spec(spec_feat)

        fused = spec_feat * (1.0 + gamma)
        if beta is not None:
            fused = fused + beta
        z = self.encoder_to_latent(fused)
        return z

    def decode(self, z_hat: torch.Tensor) -> torch.Tensor:
        x_hat = self.decoder(z_hat)
        x_hat = self._refine_output(x_hat)
        x_hat = self.output_activation(x_hat)
        return x_hat

    def forward(
        self, x: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        mask_float = valid_mask.float() if valid_mask is not None else None

        # 1. spatial branch (with mask awareness)
        gamma, beta = self.spatial_condition(x, mask=mask_float)

        # 2. spectral branch (with mask-aware pooling)
        x_low = F.avg_pool2d(x, kernel_size=2, stride=2)
        mask_low = None
        if mask_float is not None:
            # max_pool2d keeps mask as 1.0 if at least one sub-pixel was valid, ensuring that the spectral branch receives valid context for all pixels that have any valid neighbors
            mask_low = F.max_pool2d(mask_float, kernel_size=2, stride=2)

        spec_feat, attn = self._spectral_encode_grid(x_low, mask=mask_low)
        spec_feat = self.downsample_spec(spec_feat)

        # 3. fusion of spatial and spectral features (affine modulation)
        fused = spec_feat * (1.0 + gamma)
        if beta is not None:
            fused = fused + beta

        # 4. Latent & Entropy Bottleneck
        z = self.encoder_to_latent(fused)
        z_hat, likelihoods = self.entropy_bottleneck(z)

        # 5. Decoder
        x_hat = self.decoder(z_hat)
        x_hat = self._refine_output(x_hat)
        x_hat = self.output_activation(x_hat)

        out = {"x_hat": x_hat, "z": z, "z_hat": z_hat, "likelihoods": likelihoods}
        if attn is not None:
            out["attn_weights"] = attn
        return out

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
        x_hat = self.decode(z_hat)
        return {
            "x_hat": x_hat,
            "z_hat": z_hat,
        }

    @property
    def bpppc(self) -> float:
        latent_h = 32
        latent_w = 32
        input_h = 128
        input_w = 128
        return (self.latent_channels * latent_h * latent_w) / (self.in_channels * input_h * input_w)
