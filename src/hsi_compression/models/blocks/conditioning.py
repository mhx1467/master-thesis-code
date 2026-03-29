import torch
import torch.nn as nn


class SpatialConditioning(nn.Module):
    def __init__(self, channels: int, use_affine_bias: bool = False):
        super().__init__()
        self.gate = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

        self.use_affine_bias = use_affine_bias
        if use_affine_bias:
            self.bias = nn.Conv2d(channels, channels, kernel_size=1)
            nn.init.zeros_(self.bias.weight)
            nn.init.zeros_(self.bias.bias)
        else:
            self.bias = None

    def forward(self, f_spec: torch.Tensor, f_spat: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(f_spat))
        out = f_spec * (1.0 + gate)
        if self.bias is not None:
            out = out + self.bias(f_spat)
        return out
