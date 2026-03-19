import torch
from tqdm.auto import tqdm

from hsi_compression.metrics import psnr, sam_deg
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

    total_loss = total_mse = total_psnr = total_sam = 0.0
    num_batches = 0

    use_progress = show_progress and is_main_process()
    progress = (
        tqdm(loader, desc=f"Train {epoch}/{total_epochs}", leave=False) if use_progress else loader
    )

    for batch in progress:
        x = (
            batch["x"].to(device, non_blocking=True)
            if isinstance(batch, dict)
            else batch.to(device, non_blocking=True)
        )

        optimizer.zero_grad(set_to_none=True)
        outputs = model(x)
        x_hat = outputs["x_hat"]

        loss = loss_fn(x_hat, x, None)
        loss.backward()

        if grad_clip_max_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
        optimizer.step()

        with torch.no_grad():
            mse_val = torch.mean((x_hat - x) ** 2)
            psnr_val = psnr(x_hat, x, data_range=1.0)
            sam_val = sam_deg(x_hat, x)

        total_loss += loss.item()
        total_mse += mse_val.item()
        total_psnr += psnr_val.item()
        total_sam += sam_val.item()
        num_batches += 1

        if use_progress:
            progress.set_postfix(
                {
                    "loss": f"{loss.item():.5f}",
                    "psnr": f"{psnr_val.item():.2f}dB",
                }
            )

    n = max(num_batches, 1)
    return {
        "loss": reduce_mean(total_loss / n, device),
        "mse": reduce_mean(total_mse / n, device),
        "psnr": reduce_mean(total_psnr / n, device),
        "sam_deg": reduce_mean(total_sam / n, device),
    }
