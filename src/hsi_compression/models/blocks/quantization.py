import torch
import torch.nn as nn


class QuantizationProxy(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.training:
            return z + (torch.rand_like(z) - 0.5)
        return torch.round(z)
