import math
from collections.abc import Sequence

import pytorch_msssim
import torch
import torch.nn.functional as F


def _to_mask_float(mask: torch.Tensor) -> torch.Tensor:
    # metric formulas multiply by the mask, so boolean masks are converted to 0.0 or 1.0.
    return mask.float() if mask.dtype != torch.float32 else mask


def psnr(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    data_range: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    # psnr is derived from mse, with eps preventing division by zero on perfect reconstructions.
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


def ref_ssim(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    data_range: float = 1.0,
    channels: int | None = None,
) -> torch.Tensor:
    if channels is None:
        channels = int(x.shape[1])
    spatial_min = min(int(x.shape[-2]), int(x.shape[-1]))
    # ssim needs an odd window size that fits inside the spatial patch
    win_size = min(11, spatial_min if spatial_min % 2 == 1 else spatial_min - 1)
    if win_size < 3:
        return torch.tensor(float("nan"), device=x.device)
    metric = pytorch_msssim.SSIM(data_range=data_range, channel=channels, win_size=win_size).to(
        x.device
    )
    return metric(x_hat, x)


def ssim(x_hat: torch.Tensor, x: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    return ref_ssim(x_hat, x, data_range=data_range, channels=int(x.shape[1]))


def masked_mse(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    mask_f = _to_mask_float(mask)
    # only valid values contribute to both numerator and denominator.
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
    # this diagnostic checks whether the model produces unwanted signal in masked areas.
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
    # sam treats each pixel spectrum as a vector and measures the angle between spectra.
    x_hat_p = x_hat.permute(0, 2, 3, 1)
    x_p = x.permute(0, 2, 3, 1)

    dot = torch.sum(x_hat_p * x_p, dim=-1)
    norm_hat = torch.norm(x_hat_p, dim=-1).clamp_min(eps)
    norm_x = torch.norm(x_p, dim=-1).clamp_min(eps)

    cos = (dot / (norm_hat * norm_x)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos).mean()


def ref_sam(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # reference sam follows the same channel-first convention used by benchmark reports.
    numerator = torch.sum(x_hat * x, dim=1)
    denominator = torch.sqrt(torch.sum(x_hat**2, dim=1) * torch.sum(x**2, dim=1)).clamp_min(1e-12)
    fraction = (numerator / denominator).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    sa = torch.acos(fraction)
    return sa.mean()


def masked_sam(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    x_hat_p = x_hat.permute(0, 2, 3, 1)
    x_p = x.permute(0, 2, 3, 1)
    mask_p = mask.permute(0, 2, 3, 1).bool()

    # masked sam ignores invalid bands but keeps a pixel if at least one band is valid
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


def ref_sam_deg(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.rad2deg(ref_sam(x_hat, x))


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
    n, c, h, w = original_shape
    num_values = n * c * h * w
    # probability likelihoods are converted to bits with negative log base two.
    bits = torch.log(likelihoods.clamp_min(1e-12)).sum() / -math.log(2.0)
    return (bits / num_values).item()


def sid(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x_hat_p = x_hat.permute(0, 2, 3, 1)
    x_p = x.permute(0, 2, 3, 1)

    x_hat_pos = torch.clamp(x_hat_p, min=eps)
    x_pos = torch.clamp(x_p, min=eps)

    # normalize each spectrum so sid compares spectral distributions, not raw scale.
    p = x_pos / torch.sum(x_pos, dim=-1, keepdim=True)
    q = x_hat_pos / torch.sum(x_hat_pos, dim=-1, keepdim=True)

    # sid is symmetric kl divergence between normalized spectra
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

    pixel_mask = mask_p.any(dim=-1)
    if pixel_mask.sum() == 0:
        return torch.tensor(float("nan"), device=x.device)

    return sid_val[pixel_mask].mean()


def sum_string_bytes(obj) -> int:
    """
    Recursively sum byte lengths for CompressAI-style bitstream containers.

    Supported forms:
    - bytes
    - [bytes, bytes, ...]
    - [[bytes, ...], [bytes, ...], ...]
    - tuples with the same nesting
    """
    if isinstance(obj, (bytes, bytearray)):
        # a raw byte string is the simplest bitstream container.
        return len(obj)

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        # compressai can return nested lists when multiple streams are produced.
        total = 0
        for item in obj:
            total += sum_string_bytes(item)
        return total

    raise TypeError(f"Unsupported strings container type: {type(obj)!r}")


def compute_actual_bpppc_from_strings(
    strings,
    original_shape: tuple[int, ...],
) -> float:
    """
    Compute actual bits per pixel per channel from compressed bitstreams.

    original_shape should be the original input tensor shape, typically (N, C, H, W).
    """
    if strings is None:
        raise ValueError("strings must not be None")

    total_bytes = sum_string_bytes(strings)
    total_bits = total_bytes * 8

    if len(original_shape) != 4:
        raise ValueError(f"Expected original_shape to be (N, C, H, W), got {original_shape}")

    # divide by all original spectral values to report bits per pixel per channel
    n, c, h, w = original_shape
    num_values = n * c * h * w
    if num_values <= 0:
        raise ValueError(f"Invalid original_shape: {original_shape}")

    return total_bits / num_values


def compute_compression_ratio_from_bpppc(
    bpppc: float | None,
    original_bits_per_channel: float = 16.0,
) -> float | None:
    # compression ratio compares the original fixed-width sample cost to coded cost.
    if bpppc is None:
        return None
    if bpppc <= 0.0:
        return None
    return original_bits_per_channel / bpppc
