from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck
from einops import rearrange

from hsi_compression.models.blocks import BidirectionalMambaBlock


class PixelwiseSpectralAttentionPooling(nn.Module):
    """Attention pooling over spectral tokens."""

    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1, bias=False)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(h).squeeze(-1)
        alpha = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("nt,ntd->nd", alpha, h)
        return pooled, alpha


class PixelwiseSpectralRefinementBlock(nn.Module):
    """Residual local spectral correction on reconstructed spectra."""

    def __init__(self, hidden_channels: int = 16):
        super().__init__()
        self.conv1 = nn.Conv1d(1, hidden_channels, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(hidden_channels, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C)
        seq = x.unsqueeze(1)
        delta = self.conv2(self.act(self.conv1(seq))).squeeze(1)
        return x + delta


class PixelwiseSpectralMambaAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 202,
        latent_channels: int = 16,
        group_size: int = 4,
        d_model: int = 64,
        mlp_hidden_dim: int = 256,
        num_mamba_blocks: int = 4,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        pooling: str = "attention",
        refinement_hidden_channels: int = 16,
        pixels_per_patch: int | None = 512,
        eval_chunk_size: int = 8192,
        output_activation: str | None = "sigmoid",
        dropout: float = 0.0,
    ):
        super().__init__()
        if group_size <= 0:
            raise ValueError("group_size must be > 0")
        if pooling not in {"attention", "mean"}:
            raise ValueError("pooling must be one of: 'attention', 'mean'")

        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.group_size = group_size
        self.d_model = d_model
        self.mlp_hidden_dim = mlp_hidden_dim
        self.num_mamba_blocks = num_mamba_blocks
        self.pooling = pooling
        self.pixels_per_patch = pixels_per_patch
        self.eval_chunk_size = eval_chunk_size

        self.c_pad = int(math.ceil(in_channels / group_size) * group_size)
        self.num_tokens = self.c_pad // group_size
        self.pad_bands = self.c_pad - in_channels

        self.token_embed = nn.Linear(group_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, d_model))
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    BidirectionalMambaBlock(
                        d_model=d_model,
                        d_state=mamba_d_state,
                        d_conv=mamba_d_conv,
                        expand=mamba_expand,
                        dropout=dropout,
                    ),
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, mlp_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                    nn.Linear(mlp_hidden_dim, d_model),
                )
                for _ in range(num_mamba_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.attn_pool = (
            PixelwiseSpectralAttentionPooling(d_model) if pooling == "attention" else None
        )
        self.encoder_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, latent_channels),
        )
        self.entropy_bottleneck = EntropyBottleneck(latent_channels)
        self.decoder_head = nn.Sequential(
            nn.LayerNorm(latent_channels),
            nn.Linear(latent_channels, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, in_channels),
        )
        self.spectral_refinement = PixelwiseSpectralRefinementBlock(
            hidden_channels=refinement_hidden_channels
        )
        if output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation in (None, "identity"):
            self.output_activation = nn.Identity()
        else:
            raise ValueError("output_activation must be one of: 'sigmoid', 'identity', None")

    def _pad_input(self, x_pix: torch.Tensor) -> torch.Tensor:
        if self.pad_bands <= 0:
            return x_pix
        return F.pad(x_pix, (0, self.pad_bands), mode="constant", value=0.0)

    def _tokenize(self, x_pix: torch.Tensor) -> torch.Tensor:
        x_pad = self._pad_input(x_pix)
        tokens = rearrange(x_pad, "n (t g) -> n t g", g=self.group_size)
        h = self.token_embed(tokens) + self.pos_embed
        return h

    def _run_backbone(self, h: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            h = block[0](h)
            ff_in = block[1](h)
            ff_out = block[5](block[4](block[3](block[2](ff_in))))
            h = h + ff_out
        return self.final_norm(h)

    def _aggregate(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.pooling == "attention":
            return self.attn_pool(h)
        pooled = h.mean(dim=1)
        return pooled, None

    def encode_pixels(self, x_pix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self._tokenize(x_pix)
        h = self._run_backbone(h)
        pooled, alpha = self._aggregate(h)
        z = self.encoder_head(pooled)
        return z, alpha

    def decode_pixels(self, z_hat: torch.Tensor) -> torch.Tensor:
        x_hat = self.decoder_head(z_hat)
        x_hat = self.spectral_refinement(x_hat)
        x_hat = self.output_activation(x_hat)
        return x_hat

    def _sample_pixel_indices(
        self, valid_mask: torch.Tensor | None, h: int, w: int, device: torch.device
    ) -> torch.Tensor:
        num_pixels = h * w
        if self.pixels_per_patch is None or self.pixels_per_patch >= num_pixels:
            return torch.arange(num_pixels, device=device)
        if valid_mask is None:
            perm = torch.randperm(num_pixels, device=device)
            return perm[: self.pixels_per_patch]
        valid = valid_mask.reshape(-1).bool().nonzero(as_tuple=False).squeeze(-1)
        if valid.numel() == 0:
            perm = torch.randperm(num_pixels, device=device)
            return perm[: self.pixels_per_patch]
        if valid.numel() >= self.pixels_per_patch:
            perm = torch.randperm(valid.numel(), device=device)
            return valid[perm[: self.pixels_per_patch]]
        extra = torch.randint(
            0, valid.numel(), (self.pixels_per_patch - valid.numel(),), device=device
        )
        return torch.cat([valid, valid[extra]], dim=0)

    def _forward_train_sampled(
        self, x: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        b, c, h, w = x.shape
        x_bhwc = rearrange(x, "b c h w -> b (h w) c")
        mask_flat = None
        if valid_mask is not None:
            pixel_mask = self._collapse_valid_mask(valid_mask, x)
            mask_flat = rearrange(pixel_mask, "b 1 h w -> b (h w)")

        x_samples = []
        z_samples = []
        xhat_samples = []
        sampled_masks = []
        attn_weights = []
        for bi in range(b):
            idx = self._sample_pixel_indices(
                None if mask_flat is None else mask_flat[bi],
                h=h,
                w=w,
                device=x.device,
            )
            x_pix = x_bhwc[bi, idx]
            z, alpha = self.encode_pixels(x_pix)
            x_samples.append(x_pix)
            z_samples.append(z)
            sampled_masks.append(idx)
            if alpha is not None:
                attn_weights.append(alpha)

        x_sampled = torch.stack(x_samples, dim=0)
        z_sampled = torch.stack(z_samples, dim=0)
        z_4d = rearrange(z_sampled, "b k l -> b l k 1")
        z_hat_4d, likelihoods = self.entropy_bottleneck(z_4d)
        z_hat_sampled = rearrange(z_hat_4d, "b l k 1 -> b k l")

        for bi in range(b):
            x_hat = self.decode_pixels(z_hat_sampled[bi])
            xhat_samples.append(x_hat)

        xhat_sampled = torch.stack(xhat_samples, dim=0)

        x_sampled_4d = rearrange(x_sampled, "b k c -> b c k 1")
        xhat_sampled_4d = rearrange(xhat_sampled, "b k c -> b c k 1")
        mask_for_loss = torch.ones_like(xhat_sampled_4d, dtype=torch.bool)

        return {
            "x_hat": xhat_sampled_4d,
            "x_hat_for_loss": xhat_sampled_4d,
            "x_target": x_sampled_4d,
            "mask_for_loss": mask_for_loss,
            "z": z_4d,
            "z_hat": z_hat_4d,
            "likelihoods": likelihoods,
            "sampled_indices": torch.stack(sampled_masks, dim=0),
            "attn_weights": torch.stack(attn_weights, dim=0) if attn_weights else None,
        }

    def _forward_eval_chunked(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(x)
        z_hat, likelihoods = self.entropy_bottleneck(z)
        x_hat = self.decode(z_hat)
        return {"x_hat": x_hat, "z": z, "z_hat": z_hat, "likelihoods": likelihoods}

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_bhwc = rearrange(x, "b c h w -> b (h w) c")
        num_pixels = h * w
        chunk = max(1, self.eval_chunk_size)
        z_batches = []
        for bi in range(b):
            x_pix = x_bhwc[bi]
            z_chunks = []
            for start in range(0, num_pixels, chunk):
                x_chunk = x_pix[start : start + chunk]
                z_chunk, _ = self.encode_pixels(x_chunk)
                z_chunks.append(z_chunk)
            z_pix = torch.cat(z_chunks, dim=0)
            z_batches.append(rearrange(z_pix, "(h w) l -> l h w", h=h, w=w))
        return torch.stack(z_batches, dim=0)

    def forward(
        self, x: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if self.training and self.pixels_per_patch is not None:
            return self._forward_train_sampled(x, valid_mask=valid_mask)
        return self._forward_eval_chunked(x)

    def _collapse_valid_mask(self, valid_mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Convert mask to shape (B, 1, H, W)
        Supports:
        - (B, 1, H, W)
        - (B, C, H, W)
        """
        if valid_mask is None:
            return torch.ones(
                x.shape[0],
                1,
                x.shape[2],
                x.shape[3],
                device=x.device,
                dtype=torch.bool,
            )

        if valid_mask.dim() != 4:
            raise ValueError(f"valid_mask must be 4D, got shape={valid_mask.shape}")

        # already pixel mask
        if valid_mask.shape[1] == 1:
            return valid_mask > 0

        # per-band mask → collapse to pixel mask
        if valid_mask.shape[1] == x.shape[1]:
            return (valid_mask > 0).all(dim=1, keepdim=True)

        raise ValueError(
            f"Unexpected valid_mask shape {valid_mask.shape}. "
            f"Expected channel dim 1 or {x.shape[1]}."
        )

    @property
    def bpppc(self) -> float:
        return self.latent_channels / self.in_channels
