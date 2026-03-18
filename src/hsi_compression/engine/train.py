import torch
from tqdm.auto import tqdm

from hsi_compression.metrics import masked_psnr, masked_rmse
from hsi_compression.utils.distributed import is_main_process, reduce_mean


def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device: torch.device,
    epoch: int | None = None,
    total_epochs: int | None = None,
    show_progress: bool = True,
    grad_clip_max_norm: float = 1.0,
):
    model.train()

    total_loss = 0.0
    total_rmse = 0.0
    total_psnr = 0.0
    num_batches = 0

    use_progress = show_progress and is_main_process()
    progress = loader
    if use_progress:
        desc = f"Train {epoch}/{total_epochs}" if epoch and total_epochs else "Train"
        progress = tqdm(loader, desc=desc, leave=False)

    for batch in progress:
        x = batch["x"].to(device, non_blocking=True)
        mask = batch["valid_mask"].to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(x)
        x_hat = outputs["x_hat"]

        loss = loss_fn(x_hat, x, mask)
        loss.backward()

        if grad_clip_max_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)

        optimizer.step()

        with torch.no_grad():
            rmse_val = masked_rmse(x_hat, x, mask)
            psnr_val = masked_psnr(x_hat, x, mask, data_range=1.0)

        total_loss += loss.item()
        total_rmse += rmse_val.item()
        total_psnr += psnr_val.item()
        num_batches += 1

        if use_progress:
            progress.set_postfix(
                {
                    "loss": f"{loss.item():.5f}",
                    "psnr": f"{psnr_val.item():.2f}",
                }
            )

    n = max(num_batches, 1)
    return {
        "loss": reduce_mean(total_loss / n, device),
        "rmse": reduce_mean(total_rmse / n, device),
        "psnr": reduce_mean(total_psnr / n, device),
    }
