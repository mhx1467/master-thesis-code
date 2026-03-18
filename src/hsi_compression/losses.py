import torch
import torch.nn as nn
import torch.nn.functional as F

from hsi_compression.metrics import masked_mse


class MSELoss(nn.Module):
    def forward(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
        _: torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(x_hat, x)


class MaskedMSELoss(nn.Module):
    def forward(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return masked_mse(x_hat, x, mask)


class MaskedHybridLoss(nn.Module):
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mse_val = masked_mse(x_hat, x, mask)

        x_hat_p = x_hat.permute(0, 2, 3, 1)
        x_p = x.permute(0, 2, 3, 1)
        mask_p = mask.permute(0, 2, 3, 1)
        pixel_mask = mask_p.all(dim=-1)

        if pixel_mask.any():
            cos = F.cosine_similarity(x_hat_p[pixel_mask], x_p[pixel_mask], dim=-1, eps=1e-8)
            spectral_loss = (1.0 - cos.clamp(-1.0, 1.0)).mean()
        else:
            spectral_loss = torch.tensor(0.0, device=x.device)

        return mse_val + self.alpha * spectral_loss


LOSS_REGISTRY = {
    "mse": MSELoss(),
    "masked_mse": MaskedMSELoss(),
    "hybrid_mse_sam": MaskedHybridLoss(alpha=0.1),
}


def build_loss(loss_name: str) -> nn.Module:
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss name: '{loss_name}'. Available: {list(LOSS_REGISTRY.keys())}"
        )
    return LOSS_REGISTRY[loss_name]
