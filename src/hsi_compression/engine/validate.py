import time

import torch
from tqdm.auto import tqdm

from hsi_compression.metrics import (
    estimate_bpppc,
    invalid_region_mae,
    mae,
    masked_mae,
    masked_mse,
    masked_psnr,
    masked_sam_deg,
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
    use_amp: bool = False,
):
    model.eval()
    totals = {
        "loss": 0.0,
        "masked_mse": 0.0,
        "masked_mae": 0.0,
        "masked_psnr": 0.0,
        "masked_sam_deg": 0.0,
        "mse": 0.0,
        "mae": 0.0,
        "psnr": 0.0,
        "sam_deg": 0.0,
        "invalid_mae": 0.0,
        "bpppc": 0.0,
    }
    num_batches = 0
    latent_shape = None
    start_time = time.perf_counter()

    use_progress = show_progress and is_main_process()
    desc = f"Val {epoch}/{total_epochs}" + ("" if compute_sam else " (fast)")
    progress = tqdm(loader, desc=desc, leave=False) if use_progress else loader

    for batch in progress:
        if isinstance(batch, dict):
            x = batch["x"].to(device, non_blocking=True)
            mask = batch.get("valid_mask")
            mask = mask.to(device, non_blocking=True) if mask is not None else None
        else:
            x = batch.to(device, non_blocking=True)
            mask = None

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        ):
            try:
                outputs = model(x, valid_mask=mask)
            except TypeError:
                outputs = model(x)
            x_hat = outputs["x_hat"].float()
            z = outputs.get("z")
            loss_val = loss_fn(x_hat, x, mask)

        masked_mse_val = (
            masked_mse(x_hat, x, mask) if mask is not None else torch.mean((x_hat - x) ** 2)
        )
        masked_mae_val = (
            masked_mae(x_hat, x, mask) if mask is not None else torch.mean((x_hat - x).abs())
        )
        masked_psnr_val = (
            masked_psnr(x_hat, x, mask, data_range=1.0)
            if mask is not None
            else psnr(x_hat, x, data_range=1.0)
        )
        masked_sam_val = (
            masked_sam_deg(x_hat, x, mask)
            if (compute_sam and mask is not None)
            else (sam_deg(x_hat, x) if compute_sam else None)
        )
        mse_val = torch.mean((x_hat - x) ** 2)
        mae_val = mae(x_hat, x)
        psnr_val = psnr(x_hat, x, data_range=1.0)
        sam_val = sam_deg(x_hat, x) if compute_sam else None
        invalid_mae_val = (
            invalid_region_mae(x_hat, mask)
            if mask is not None
            else torch.tensor(0.0, device=device)
        )

        totals["loss"] += loss_val.item()
        totals["masked_mse"] += masked_mse_val.item()
        totals["masked_mae"] += masked_mae_val.item()
        totals["masked_psnr"] += masked_psnr_val.item()
        totals["mse"] += mse_val.item()
        totals["mae"] += mae_val.item()
        totals["psnr"] += psnr_val.item()
        totals["invalid_mae"] += invalid_mae_val.item()
        if compute_sam:
            totals["masked_sam_deg"] += masked_sam_val.item()
            totals["sam_deg"] += sam_val.item()

        num_batches += 1
        if z is not None:
            if latent_shape is None:
                latent_shape = tuple(z.shape[1:])
            totals["bpppc"] += estimate_bpppc(
                z, num_bands=num_input_bands, quantization_bits=quantization_bits
            )

        if use_progress:
            postfix = {"loss": f"{loss_val.item():.5f}", "mPSNR": f"{masked_psnr_val.item():.2f}dB"}
            if compute_sam:
                postfix["mSAM"] = f"{masked_sam_val.item():.2f}°"
            progress.set_postfix(postfix)

    n = max(num_batches, 1)
    out = {
        k: reduce_mean(v / n, device)
        for k, v in totals.items()
        if k not in {"sam_deg", "masked_sam_deg"} or compute_sam
    }
    if not compute_sam:
        out["sam_deg"] = None
        out["masked_sam_deg"] = None
    out["latent_shape"] = latent_shape
    out["epoch_time_sec"] = reduce_mean(time.perf_counter() - start_time, device)
    return out
