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
        # plain mse ignores valid masks and is mostly used for simple baselines.
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
        # rmse keeps the same units as the normalized image values.
        return torch.sqrt(F.mse_loss(x_hat, x) + self.eps)


class MaskedMSELoss(nn.Module):
    def forward(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if mask is None:
            # fall back to standard mse when the dataset has no mask.
            return F.mse_loss(x_hat, x)
        return masked_mse(x_hat, x, mask)


class SymbolCodeLengthLoss(nn.Module):
    """Differentiable proxy for coding integer prediction residuals.

    The forward pass uses rounded integer symbols, while the backward pass treats rounding as
    identity. This aligns TCN fine-tuning with the lossless residual stream more directly than
    normalized-value MSE.
    """

    select_by_loss = True

    def __init__(
        self,
        symbol_scale: int = 10000,
        code_weight: float = 1e-4,
        mse_weight: float = 1.0,
        value_min: float = -1.0,
        value_max: float = 1.0,
    ):
        super().__init__()
        if symbol_scale <= 0:
            raise ValueError("symbol_scale must be positive")
        if code_weight < 0.0:
            raise ValueError("code_weight must be non-negative")
        if mse_weight < 0.0:
            raise ValueError("mse_weight must be non-negative")
        if value_min >= value_max:
            raise ValueError("value_min must be smaller than value_max")

        self.symbol_scale = int(symbol_scale)
        self.code_weight = float(code_weight)
        self.mse_weight = float(mse_weight)
        self.value_min = float(value_min)
        self.value_max = float(value_max)

    def forward(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x_hat_fp32 = x_hat.float()
        x_fp32 = x.float()

        # the straight through round keeps symbol training differentiable
        pred_symbols = self._round_ste(
            x_hat_fp32.clamp(self.value_min, self.value_max) * self.symbol_scale
        )
        target_symbols = torch.round(
            x_fp32.clamp(self.value_min, self.value_max) * self.symbol_scale
        ).detach()

        residual = target_symbols - pred_symbols
        # log residual magnitude is a simple proxy for how expensive residuals are to code
        code_proxy = torch.log1p(residual.abs()) / math.log(2.0)
        code_loss = self._masked_mean(code_proxy, mask)

        if self.mse_weight == 0.0:
            mse_loss = torch.zeros((), device=x_hat.device, dtype=torch.float32)
        elif mask is None:
            mse_loss = F.mse_loss(x_hat_fp32, x_fp32)
        else:
            mse_loss = masked_mse(x_hat_fp32, x_fp32, mask)

        return self.code_weight * code_loss + self.mse_weight * mse_loss

    @staticmethod
    def _round_ste(x: torch.Tensor) -> torch.Tensor:
        # forward uses rounded values, backward sees identity for usable gradients.
        return x + (torch.round(x) - x).detach()

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return values.mean()
        valid = mask.to(device=values.device, dtype=torch.bool)
        if not valid.any():
            # empty masks should not create nan losses during unusual batches.
            return torch.zeros((), device=values.device, dtype=values.dtype)
        return values[valid].mean()


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
            # cosine loss penalizes spectral shape errors even when mse is small.
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
        # spectral angle is meaningful only where the full spectrum is valid.
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

        # likelihoods are probabilities, so negative log two gives estimated bits
        bits = torch.log(likelihoods.clamp_min(1e-12)).sum() / -math.log(2.0)
        R = bits / num_pixels

        loss = D + self.lmbda * R

        return loss, D, R


LOSS_REGISTRY = {
    "mse": MSELoss(),
    "rmse": RMSELoss(),
    "masked_mse": MaskedMSELoss(),
    "hybrid_mse_sam": MaskedHybridLoss(alpha=0.1),
    "rate_distortion": RateDistortionLoss,
    "symbol_code_length": SymbolCodeLengthLoss,
}


def build_loss(loss_name: str, **kwargs) -> nn.Module:
    if loss_name == "rate_distortion":
        # rate-distortion needs constructor arguments such as lambda and distortion metric.
        return RateDistortionLoss(**kwargs)
    if loss_name == "symbol_code_length":
        return SymbolCodeLengthLoss(**kwargs)
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss name: '{loss_name}'. Available: {list(LOSS_REGISTRY.keys())}"
        )
    if kwargs:
        raise ValueError(f"Loss '{loss_name}' does not accept kwargs: {sorted(kwargs)}")
    return LOSS_REGISTRY[loss_name]
