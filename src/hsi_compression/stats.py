import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_global_minmax(
    dataset,
    max_samples: int | None = None,
    num_workers: int = 0,
) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )

    global_min = float("inf")
    global_max = float("-inf")
    num_valid = 0

    for i, batch in enumerate(loader):
        if max_samples is not None and i >= max_samples:
            break

        x = batch["x"].squeeze(0).float()       # (C, H, W)
        m = batch["valid_mask"].squeeze(0)       # (C, H, W) bool

        valid_vals = x[m]
        if valid_vals.numel() == 0:
            continue

        batch_min = valid_vals.min().item()
        batch_max = valid_vals.max().item()

        global_min = min(global_min, batch_min)
        global_max = max(global_max, batch_max)
        num_valid += valid_vals.numel()

    if global_min == float("inf"):
        raise RuntimeError("There are no valid pixels in the dataset to compute statistics.")

    return {
        "global_min": global_min,
        "global_max": global_max,
        "num_valid_pixels": num_valid,
    }