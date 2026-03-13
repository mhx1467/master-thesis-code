import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_band_stats(dataset, max_samples=None, num_workers: int = 0):
    """
    Compute per-band mean/std using only valid pixels.

    Expects dataset items to be dicts with:
      - "x": (B, H, W)
      - "valid_mask": (B, H, W)
    """
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )

    sum_ = None
    sumsq_ = None
    count_ = None

    for i, batch in enumerate(loader):
        if max_samples is not None and i >= max_samples:
            break

        x = batch["x"].squeeze(0)          # (B, H, W)
        m = batch["valid_mask"].squeeze(0) # (B, H, W)

        B = x.shape[0]

        if sum_ is None:
            sum_ = torch.zeros(B, dtype=torch.float64)
            sumsq_ = torch.zeros(B, dtype=torch.float64)
            count_ = torch.zeros(B, dtype=torch.float64)

        for b in range(B):
            xb = x[b][m[b]]
            if xb.numel() == 0:
                continue

            sum_[b] += xb.double().sum()
            sumsq_[b] += (xb.double() ** 2).sum()
            count_[b] += xb.numel()

    mean = sum_ / torch.clamp(count_, min=1.0)
    var = sumsq_ / torch.clamp(count_, min=1.0) - mean ** 2
    var = torch.clamp(var, min=1e-12)
    std = torch.sqrt(var)

    return mean.float(), std.float(), count_