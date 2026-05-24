import torch
import torch.nn as nn


class SpectralRefinementBlock(nn.Module):
    """Per-pixel residual refinement along the spectral axis."""

    def __init__(self, in_channels: int, hidden_channels: int | None = None):
        super().__init__()
        hidden = hidden_channels or in_channels
        self.conv1 = nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(hidden, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is a hyperspectral batch with channels as spectral bands.
        b, c, h, w = x.shape
        # reshape spatial positions into a sequence so conv1d can refine across them.
        seq = x.reshape(b, c, h * w)
        delta = self.conv1(seq)
        delta = self.act(delta)
        delta = self.conv2(delta)
        delta = delta.reshape(b, c, h, w)
        return x + delta
