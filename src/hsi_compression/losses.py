import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hsi_compression.metrics import masked_mse


class MSELoss(nn.Module):
    def forward(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
        _: torch.Tensor | None,
    ) -> torch.Tensor:
        return F.mse_loss(x_hat, x)


class RMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
        _: torch.Tensor | None,
    ) -> torch.Tensor:
        return torch.sqrt(F.mse_loss(x_hat, x) + self.eps)


class MaskedMSELoss(nn.Module):
    def forward(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if mask is None:
            return F.mse_loss(x_hat, x)
        return masked_mse(x_hat, x, mask)


class MaskedHybridLoss(nn.Module):
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if mask is None:
            mse_val = F.mse_loss(x_hat, x)
            x_hat_p = x_hat.permute(0, 2, 3, 1)
            x_p = x.permute(0, 2, 3, 1)
            cos = F.cosine_similarity(
                x_hat_p.reshape(-1, x_hat_p.shape[-1]),
                x_p.reshape(-1, x_p.shape[-1]),
                dim=-1,
                eps=1e-8,
            )
            spectral_loss = (1.0 - cos.clamp(-1.0, 1.0)).mean()
            return mse_val + self.alpha * spectral_loss

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


class RateDistortionLoss(nn.Module):
    def __init__(self, lmbda: float = 0.01, distortion_metric: str = "masked_mse"):
        super().__init__()
        self.lmbda = lmbda

        self.distortion_fn = LOSS_REGISTRY.get(distortion_metric, MaskedMSELoss())

    def forward(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        likelihoods: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        D = self.distortion_fn(x_hat, x, mask)

        N, C, H, W = x.shape
        num_pixels = N * C * H * W

        bits = torch.log(likelihoods).sum() / -math.log(2.0)
        R = bits / num_pixels

        loss = D + self.lmbda * R

        return loss, D, R


LOSS_REGISTRY = {
    "mse": MSELoss(),
    "rmse": RMSELoss(),
    "masked_mse": MaskedMSELoss(),
    "hybrid_mse_sam": MaskedHybridLoss(alpha=0.1),
    "rate_distortion": RateDistortionLoss,
}


def build_loss(loss_name: str, **kwargs) -> nn.Module:
    if loss_name == "rate_distortion":
        return RateDistortionLoss(**kwargs)
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss name: '{loss_name}'. Available: {list(LOSS_REGISTRY.keys())}"
        )
    return LOSS_REGISTRY[loss_name]
