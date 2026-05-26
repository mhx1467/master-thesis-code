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


class MaskedSpectralFeatureLoss(nn.Module):
    """Pixel distortion plus low-order spectral feature preservation.

    The feature terms intentionally mirror the simple downstream HYPERVIEW2 feature family
    used in diagnostics: per-band spatial mean, per-band spatial standard deviation, and
    first/second differences of the mean spectrum. This keeps the loss task-aware without
    introducing a learned regressor into the compression training loop.
    """

    def __init__(
        self,
        pixel_weight: float = 1.0,
        mean_weight: float = 0.5,
        std_weight: float = 0.25,
        first_derivative_weight: float = 1.0,
        second_derivative_weight: float = 0.25,
        spectral_cosine_weight: float = 0.01,
        eps: float = 1e-8,
    ):
        super().__init__()
        weights = {
            "pixel_weight": pixel_weight,
            "mean_weight": mean_weight,
            "std_weight": std_weight,
            "first_derivative_weight": first_derivative_weight,
            "second_derivative_weight": second_derivative_weight,
            "spectral_cosine_weight": spectral_cosine_weight,
        }
        for name, value in weights.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.pixel_weight = float(pixel_weight)
        self.mean_weight = float(mean_weight)
        self.std_weight = float(std_weight)
        self.first_derivative_weight = float(first_derivative_weight)
        self.second_derivative_weight = float(second_derivative_weight)
        self.spectral_cosine_weight = float(spectral_cosine_weight)
        self.eps = float(eps)

    def forward(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x_hat_fp32 = torch.nan_to_num(x_hat.float(), nan=0.0, posinf=1.0, neginf=0.0)
        x_fp32 = torch.nan_to_num(x.float(), nan=0.0, posinf=1.0, neginf=0.0)

        loss = torch.zeros((), device=x_hat.device, dtype=torch.float32)
        if self.pixel_weight > 0.0:
            if mask is None:
                pixel_loss = F.mse_loss(x_hat_fp32, x_fp32)
            else:
                pixel_loss = masked_mse(x_hat_fp32, x_fp32, mask)
            loss = loss + self.pixel_weight * pixel_loss

        mean_hat, std_hat = self._spatial_mean_std(x_hat_fp32, mask)
        mean_x, std_x = self._spatial_mean_std(x_fp32, mask)

        if self.mean_weight > 0.0:
            loss = loss + self.mean_weight * F.mse_loss(mean_hat, mean_x)
        if self.std_weight > 0.0:
            loss = loss + self.std_weight * F.mse_loss(std_hat, std_x)
        if self.first_derivative_weight > 0.0 and mean_hat.shape[1] > 1:
            loss = loss + self.first_derivative_weight * F.mse_loss(
                mean_hat.diff(dim=1), mean_x.diff(dim=1)
            )
        if self.second_derivative_weight > 0.0 and mean_hat.shape[1] > 2:
            loss = loss + self.second_derivative_weight * F.mse_loss(
                mean_hat.diff(n=2, dim=1), mean_x.diff(n=2, dim=1)
            )
        if self.spectral_cosine_weight > 0.0:
            cosine = F.cosine_similarity(mean_hat, mean_x, dim=1, eps=self.eps)
            cosine_loss = (1.0 - cosine.clamp(-1.0, 1.0)).mean()
            loss = loss + self.spectral_cosine_weight * cosine_loss

        return loss

    def _spatial_mean_std(
        self,
        values: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            mean = values.mean(dim=(2, 3))
            std = values.std(dim=(2, 3), unbiased=False)
            return mean, std

        mask_f = mask.to(device=values.device, dtype=values.dtype)
        denom = mask_f.sum(dim=(2, 3)).clamp_min(1.0)
        mean = (values * mask_f).sum(dim=(2, 3)) / denom
        centered = values - mean[:, :, None, None]
        var = (centered.square() * mask_f).sum(dim=(2, 3)) / denom
        std = torch.sqrt(var.clamp_min(0.0) + self.eps)
        return mean, std


class RateDistortionLoss(nn.Module):
    def __init__(
        self,
        lmbda: float = 0.01,
        distortion_metric: str = "masked_mse",
        distortion_kwargs: dict | None = None,
    ):
        super().__init__()
        self.lmbda = lmbda
        self.distortion_metric = distortion_metric
        self.distortion_kwargs = dict(distortion_kwargs or {})
        if distortion_metric == "rate_distortion":
            raise ValueError("rate_distortion cannot be nested as its own distortion metric")

        self.distortion_fn = build_loss(distortion_metric, **self.distortion_kwargs)

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
    "spectral_feature": MaskedSpectralFeatureLoss,
    "rate_distortion": RateDistortionLoss,
    "symbol_code_length": SymbolCodeLengthLoss,
}


def build_loss(loss_name: str, **kwargs) -> nn.Module:
    if loss_name == "rate_distortion":
        # rate-distortion needs constructor arguments such as lambda and distortion metric.
        return RateDistortionLoss(**kwargs)
    if loss_name == "symbol_code_length":
        return SymbolCodeLengthLoss(**kwargs)
    if loss_name == "spectral_feature":
        return MaskedSpectralFeatureLoss(**kwargs)
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss name: '{loss_name}'. Available: {list(LOSS_REGISTRY.keys())}"
        )
    if kwargs:
        raise ValueError(f"Loss '{loss_name}' does not accept kwargs: {sorted(kwargs)}")
    return LOSS_REGISTRY[loss_name]
