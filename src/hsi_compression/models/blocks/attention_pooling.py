import torch
import torch.nn as nn


class SpectralAttentionPooling(nn.Module):
    """Learned token pooling across the spectral sequence dimension."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.score = nn.Linear(input_dim, 1)
        self.proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B_seq, L_seq, D)
        attn_logits = self.score(x)
        attn = torch.softmax(attn_logits, dim=1)
        pooled = torch.sum(attn * x, dim=1)
        return self.proj(pooled)
