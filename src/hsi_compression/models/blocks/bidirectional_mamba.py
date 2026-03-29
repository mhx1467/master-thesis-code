import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError as exc:  # pragma: no cover - depends on user environment
    Mamba = None
    _MAMBA_IMPORT_ERROR = exc
else:
    _MAMBA_IMPORT_ERROR = None


class BidirectionalMambaBlock(nn.Module):
    """Bidirectional Mamba block for non-causal spectral sequences."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if Mamba is None:  # pragma: no cover - depends on user environment
            raise ImportError(
                "mamba-ssm is required for SpectralFirstMambaAutoencoder. "
                "Install and verify it in the active environment before using this model."
            ) from _MAMBA_IMPORT_ERROR

        self.norm = nn.LayerNorm(d_model)
        self.forward_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.backward_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.merge = nn.Linear(2 * d_model, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)

        y_fwd = self.forward_mamba(x_norm)

        x_rev = torch.flip(x_norm, dims=[1])
        y_bwd = self.backward_mamba(x_rev)
        y_bwd = torch.flip(y_bwd, dims=[1])

        y = torch.cat([y_fwd, y_bwd], dim=-1)
        y = self.merge(y)
        y = self.dropout(y)
        return residual + y
