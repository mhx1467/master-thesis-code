import torch
import torch.nn as nn
import torch.nn.functional as F
from hsi_compression.metrics import masked_mse

class MaskedHybridLoss(nn.Module):
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mse_val = masked_mse(x_hat, x, mask)

        x_hat_p = x_hat.permute(0, 2, 3, 1)
        x_p = x.permute(0, 2, 3, 1)
        mask_p = mask.permute(0, 2, 3, 1)

        x_hat_p = x_hat_p * mask_p
        x_p = x_p * mask_p

        cos = F.cosine_similarity(x_hat_p, x_p, dim=-1) # (N, H, W)
        
        pixel_mask = mask_p.any(dim=-1) # (N, H, W)

        if pixel_mask.any():
            sam_loss = (1.0 - cos[pixel_mask]).mean()
        else:
            sam_loss = torch.tensor(0.0, device=x.device)

        return mse_val + self.alpha * sam_loss

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