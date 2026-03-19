import torch
from tqdm.auto import tqdm

from hsi_compression.metrics import (
    estimate_bpppc,
    psnr,
    sam_deg,
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
    compute_sam: bool = True,
):
    model.eval()

    total_loss = total_mse = total_psnr = total_sam = total_bpppc = 0.0
    num_batches = 0
    latent_shape = None

    use_progress = show_progress and is_main_process()
    desc = f"Val {epoch}/{total_epochs}" + ("" if compute_sam else " (fast)")
    progress = tqdm(loader, desc=desc, leave=False) if use_progress else loader

    for batch in progress:
        x = (
            batch["x"].to(device, non_blocking=True)
            if isinstance(batch, dict)
            else batch.to(device, non_blocking=True)
        )

        outputs = model(x)
        x_hat = outputs["x_hat"]
        z = outputs.get("z")

        loss_val = loss_fn(x_hat, x, None)
        mse_val = torch.mean((x_hat - x) ** 2)
        psnr_val = psnr(x_hat, x, data_range=1.0)

        total_loss += loss_val.item()
        total_mse += mse_val.item()
        total_psnr += psnr_val.item()
        num_batches += 1

        sam_val = None
        if compute_sam:
            sam_val = sam_deg(x_hat, x)
            total_sam += sam_val.item()

        if z is not None:
            if latent_shape is None:
                latent_shape = tuple(z.shape[1:])
            total_bpppc += estimate_bpppc(
                z,
                num_bands=num_input_bands,
                quantization_bits=quantization_bits,
            )

        if use_progress:
            postfix = {"loss": f"{loss_val.item():.5f}", "psnr": f"{psnr_val.item():.2f}dB"}
            if compute_sam:
                postfix["sam"] = f"{sam_val.item():.2f}°"
            progress.set_postfix(postfix)

    n = max(num_batches, 1)
    return {
        "loss": reduce_mean(total_loss / n, device),
        "mse": reduce_mean(total_mse / n, device),
        "psnr": reduce_mean(total_psnr / n, device),
        "sam_deg": reduce_mean(total_sam / n, device) if compute_sam else None,
        "bpppc": total_bpppc / n if num_batches > 0 else 0.0,
        "latent_shape": latent_shape,
    }
