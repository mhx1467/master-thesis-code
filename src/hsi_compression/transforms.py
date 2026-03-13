import torch


class BandStandardize:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6):
        self.mean = mean.float()
        self.std = std.float()
        self.eps = eps

    def __call__(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = (x - self.mean[:, None, None]) / (self.std[:, None, None] + self.eps)

        if valid_mask is not None:
            x = x.clone()
            x[~valid_mask] = 0.0

        return x