import torch
import torch.nn as nn


class SpectralAttentionPooling(nn.Module):
    """Learned token pooling across the spectral sequence dimension."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.score = nn.Linear(input_dim, 1)
        self.proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x contains a batch of spectral token sequences.
        # the score layer decides which tokens matter most for the final summary.
        attn_logits = self.score(x)
        attn = torch.softmax(attn_logits, dim=1)
        # weighted sum turns a variable-length sequence into one fixed-size vector.
        pooled = torch.sum(attn * x, dim=1)
        return self.proj(pooled)
