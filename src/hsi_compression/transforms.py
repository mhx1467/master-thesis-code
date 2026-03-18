import torch


class GlobalMinMaxNormalize:
    def __init__(
        self,
        global_min: float,
        global_max: float,
        eps: float = 1e-8,
    ):
        if global_max <= global_min:
            raise ValueError(
                f"global_max ({global_max}) must be > global_min ({global_min})"
            )
        self.global_min = global_min
        self.global_max = global_max
        self.eps = eps
        self._range = global_max - global_min

    def __call__(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = (x.float() - self.global_min) / (self._range + self.eps)
        x = torch.clamp(x, 0.0, 1.0)

        if valid_mask is not None:
            x = x.clone()
            x[~valid_mask] = 0.0

        return x

    def inverse(self, x_norm: torch.Tensor) -> torch.Tensor:
        return x_norm * self._range + self.global_min

    def __repr__(self) -> str:
        return (
            f"GlobalMinMaxNormalize("
            f"min={self.global_min:.2f}, max={self.global_max:.2f})"
        )