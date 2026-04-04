import math

import torch
import torch.nn.functional as F


def _to_mask_float(mask: torch.Tensor) -> torch.Tensor:
    return mask.float() if mask.dtype != torch.float32 else mask


def psnr(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    data_range: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    mse_val = torch.mean((x_hat - x) ** 2)
    return 10.0 * torch.log10(torch.tensor(data_range**2, device=x.device) / (mse_val + eps))


def masked_psnr(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
    data_range: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    mse_val = masked_mse(x_hat, x, mask, eps=eps)
    return 10.0 * torch.log10(torch.tensor(data_range**2, device=x.device) / (mse_val + eps))


def mse(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_hat, x)


def mae(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(x_hat, x)


def masked_mse(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    mask_f = _to_mask_float(mask)
    se = (x_hat - x) ** 2 * mask_f
    return se.sum() / mask_f.sum().clamp_min(eps)


def masked_mae(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    mask_f = _to_mask_float(mask)
    ae = (x_hat - x).abs() * mask_f
    return ae.sum() / mask_f.sum().clamp_min(eps)


def invalid_region_mae(
    x_hat: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    invalid = (~mask.bool()).float()
    return (x_hat.abs() * invalid).sum() / invalid.sum().clamp_min(eps)


def masked_rmse(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    return torch.sqrt(masked_mse(x_hat, x, mask, eps=eps))


def sam(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    x_hat_p = x_hat.permute(0, 2, 3, 1)
    x_p = x.permute(0, 2, 3, 1)

    dot = torch.sum(x_hat_p * x_p, dim=-1)
    norm_hat = torch.norm(x_hat_p, dim=-1).clamp_min(eps)
    norm_x = torch.norm(x_p, dim=-1).clamp_min(eps)

    cos = (dot / (norm_hat * norm_x)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos).mean()


def masked_sam(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    x_hat_p = x_hat.permute(0, 2, 3, 1)
    x_p = x.permute(0, 2, 3, 1)
    mask_p = mask.permute(0, 2, 3, 1).bool()

    # valid spectrum only if at least one valid band exists; keep masked multiplication
    x_hat_p = x_hat_p * mask_p
    x_p = x_p * mask_p

    dot = torch.sum(x_hat_p * x_p, dim=-1)
    norm_hat = torch.norm(x_hat_p, dim=-1).clamp_min(eps)
    norm_x = torch.norm(x_p, dim=-1).clamp_min(eps)

    pixel_mask = mask_p.any(dim=-1)
    cos = (dot / (norm_hat * norm_x)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angles = torch.acos(cos)

    if pixel_mask.sum() == 0:
        return torch.tensor(float("nan"), device=x.device)
    return angles[pixel_mask].mean()


def sam_deg(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    return sam(x_hat, x, eps=eps) * (180.0 / math.pi)


def masked_sam_deg(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    return masked_sam(x_hat, x, mask, eps=eps) * (180.0 / math.pi)


def compute_true_bpppc(
    likelihoods: torch.Tensor, original_shape: tuple[int, int, int, int]
) -> float:
    N, C, H, W = original_shape
    num_pixels = N * C * H * W

    bits = torch.log(likelihoods).sum() / -math.log(2.0)

    return (bits / num_pixels).item()


def sid(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # 1. Transform to HWC to operate on spectral vectors (dim=-1)
    x_hat_p = x_hat.permute(0, 2, 3, 1)
    x_p = x.permute(0, 2, 3, 1)

    # 3. Protection against NaNs/Infs - if any value is NaN/Inf, the whole SID becomes NaN, which is undesirable.
    x_hat_pos = torch.clamp(x_hat_p, min=eps)
    x_pos = torch.clamp(x_p, min=eps)

    # 3. Normalization of each spectral vector to sum to 1 (distributions p and q)
    p = x_pos / torch.sum(x_pos, dim=-1, keepdim=True)
    q = x_hat_pos / torch.sum(x_hat_pos, dim=-1, keepdim=True)

    # 4. Calculation of KL divergence D(p||q) and D(q||p) for each pixel, then sum to get SID, and finally average over all pixels.
    d_pq = torch.sum(p * torch.log(p / q), dim=-1)
    d_qp = torch.sum(q * torch.log(q / p), dim=-1)

    sid_val = d_pq + d_qp

    return sid_val.mean()


def masked_sid(
    x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    x_hat_p = x_hat.permute(0, 2, 3, 1)
    x_p = x.permute(0, 2, 3, 1)
    mask_p = mask.permute(0, 2, 3, 1).bool()

    x_hat_pos = torch.clamp(x_hat_p, min=eps)
    x_pos = torch.clamp(x_p, min=eps)

    p = x_pos / torch.sum(x_pos, dim=-1, keepdim=True)
    q = x_hat_pos / torch.sum(x_hat_pos, dim=-1, keepdim=True)

    d_pq = torch.sum(p * torch.log(p / q), dim=-1)
    d_qp = torch.sum(q * torch.log(q / p), dim=-1)

    sid_val = d_pq + d_qp

    # masking values - if any band is invalid, the whole pixel is invalid; keep masked multiplication for correct SID values
    pixel_mask = mask_p.any(dim=-1)

    if pixel_mask.sum() == 0:
        return torch.tensor(float("nan"), device=x.device)

    return sid_val[pixel_mask].mean()
