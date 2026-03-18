import torch
from tqdm.auto import tqdm

from hsi_compression.metrics import (
    masked_rmse,
    masked_psnr,
    masked_sam_deg,
    estimate_bpppc,
)
from hsi_compression.utils.distributed import is_main_process, reduce_mean


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    loss_fn,
    device: torch.device,
    num_input_bands: int = 202,
    quantization_bits: int = 8,
    epoch: int | None = None,
    total_epochs: int | None = None,
    show_progress: bool = True,
):
    model.eval()

    total_loss = 0.0
    total_rmse = 0.0
    total_psnr = 0.0
    total_sam_deg = 0.0
    total_bpppc = 0.0
    num_batches = 0
    latent_shape = None

    use_progress = show_progress and is_main_process()
    progress = loader
    if use_progress:
        desc = f"Val {epoch}/{total_epochs}" if epoch and total_epochs else "Val"
        progress = tqdm(loader, desc=desc, leave=False)

    for batch in progress:
        x = batch["x"].to(device, non_blocking=True)
        mask = batch["valid_mask"].to(device, non_blocking=True)

        outputs = model(x)
        x_hat = outputs["x_hat"]
        z = outputs.get("z")

        loss_val  = loss_fn(x_hat, x, mask)
        rmse_val  = masked_rmse(x_hat, x, mask)
        psnr_val  = masked_psnr(x_hat, x, mask, data_range=1.0)
        sam_val   = masked_sam_deg(x_hat, x, mask)

        total_loss    += loss_val.item()
        total_rmse    += rmse_val.item()
        total_psnr    += psnr_val.item()
        total_sam_deg += sam_val.item()
        num_batches   += 1

        if z is not None:
            if latent_shape is None:
                latent_shape = tuple(z.shape[1:])
            total_bpppc += estimate_bpppc(
                z, num_bands=num_input_bands, quantization_bits=quantization_bits
            )

        if use_progress:
            progress.set_postfix({
                "loss": f"{loss_val.item():.5f}",
                "psnr": f"{psnr_val.item():.2f}dB",
                "sam":  f"{sam_val.item():.2f}°",
            })

    n = max(num_batches, 1)
    return {
        "loss":      reduce_mean(total_loss / n, device),
        "rmse":      reduce_mean(total_rmse / n, device),
        "psnr":      reduce_mean(total_psnr / n, device),
        "sam_deg":   reduce_mean(total_sam_deg / n, device),
        "bpppc":     total_bpppc / n if num_batches > 0 else 0.0,
        "latent_shape": latent_shape,
    }
