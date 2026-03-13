import torch


def masked_mse(x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    mask = mask.float()
    se = (x_hat - x) ** 2
    se = se * mask
    denom = mask.sum().clamp_min(eps)
    return se.sum() / denom


def masked_rmse(x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(masked_mse(x_hat, x, mask, eps=eps))


def masked_psnr(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
    data_range: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    mse = masked_mse(x_hat, x, mask, eps=eps)
    return 10.0 * torch.log10((data_range ** 2) / (mse + eps))


def masked_sam(x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    x_hat, x, mask: (N, B, H, W)
    SAM is computed only for pixels where all bands are valid.
    """
    pixel_mask = mask.all(dim=1)  # (N, H, W)

    x_hat = x_hat.permute(0, 2, 3, 1)  # (N, H, W, B)
    x = x.permute(0, 2, 3, 1)

    dot = torch.sum(x_hat * x, dim=-1)
    norm_hat = torch.norm(x_hat, dim=-1)
    norm = torch.norm(x, dim=-1)

    cos = dot / (norm_hat * norm + eps)
    cos = torch.clamp(cos, -1.0, 1.0)

    sam = torch.acos(cos)

    if pixel_mask.sum() == 0:
        return torch.tensor(float("nan"), device=x.device)

    return sam[pixel_mask].mean()


def masked_sam_deg(x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return masked_sam(x_hat, x, mask, eps=eps) * 180.0 / torch.pi