import torch
import torch.nn.functional as F


def psnr(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    data_range: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    mse = torch.mean((x_hat - x) ** 2)
    return 10.0 * torch.log10(torch.tensor(data_range**2, device=x.device) / (mse + eps))


def masked_psnr(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
    data_range: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    mse = masked_mse(x_hat, x, mask, eps=eps)
    return 10.0 * torch.log10(torch.tensor(data_range**2, device=x.device) / (mse + eps))


def mse(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_hat, x)


def masked_mse(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    mask_f = mask.float()
    se = (x_hat - x) ** 2 * mask_f
    return se.sum() / mask_f.sum().clamp_min(eps)


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
    # (N, C, H, W) to (N, H, W, C)
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
    mask_p = mask.permute(0, 2, 3, 1)

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


def masked_sam_deg(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    return masked_sam(x_hat, x, mask, eps=eps) * (180.0 / torch.pi)


def sam_deg(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return sam(x_hat, x, eps=eps) * (180.0 / torch.pi)


def estimate_bpppc(
    latent: torch.Tensor,
    num_bands: int,
    quantization_bits: int = 8,
) -> float:
    patch_pixels = 128 * 128
    latent_elements = latent[0].numel()  # C_l x H_l x W_l per patch
    latent_bits = latent_elements * quantization_bits
    return latent_bits / (patch_pixels * num_bands)
