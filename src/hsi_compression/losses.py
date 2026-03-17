import torch
import torch.nn as nn
import torch.nn.functional as F

from hsi_compression.metrics import masked_mse


class MaskedHybridLoss(nn.Module):
    """
    MSE + alpha * spectral cosine loss

    Note:
    - this is not exact SAM loss
    - it is a cosine-based spectral regularizer
    """
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mse_val = masked_mse(x_hat, x, mask)

        # (N, B, H, W) -> (N, H, W, B)
        x_hat_p = x_hat.permute(0, 2, 3, 1)
        x_p = x.permute(0, 2, 3, 1)
        mask_p = mask.permute(0, 2, 3, 1)

        # only compare pixels where all bands are valid
        pixel_mask = mask_p.all(dim=-1)  # (N, H, W)

        if pixel_mask.any():
            x_hat_valid = x_hat_p[pixel_mask]  # (num_valid_pixels, B)
            x_valid = x_p[pixel_mask]

            cos = F.cosine_similarity(x_hat_valid, x_valid, dim=-1, eps=1e-8)
            cos = torch.clamp(cos, -1.0, 1.0)

            spectral_loss = (1.0 - cos).mean()
        else:
            spectral_loss = torch.tensor(0.0, device=x.device)

        return mse_val + self.alpha * spectral_loss


LOSS_REGISTRY = {
    "masked_mse": lambda x_hat, x, mask: masked_mse(x_hat, x, mask),
    "hybrid_mse_sam": MaskedHybridLoss(alpha=0.1),
}


def build_loss(loss_name: str):
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss_name: {loss_name}. "
            f"Available: {list(LOSS_REGISTRY.keys())}"
        )
    return LOSS_REGISTRY[loss_name]