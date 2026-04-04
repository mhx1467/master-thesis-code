import time

import torch
from compressai.entropy_models import EntropyBottleneck
from tqdm.auto import tqdm

from hsi_compression.losses import RateDistortionLoss
from hsi_compression.metrics import (
    invalid_region_mae,
    mae,
    masked_mae,
    masked_mse,
    masked_psnr,
    masked_sam_deg,
    masked_sid,
    psnr,
    sam_deg,
    sid,
    ssim,
)
from hsi_compression.utils.distributed import is_main_process, reduce_mean


def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device: torch.device,
    aux_optimizer=None,
    epoch: int | None = None,
    total_epochs: int | None = None,
    show_progress: bool = True,
    grad_clip_max_norm: float = 1.0,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
):
    model.train()
    totals = {
        "loss": 0.0,
        "D": 0.0,
        "R": 0.0,
        "masked_mse": 0.0,
        "masked_mae": 0.0,
        "masked_psnr": 0.0,
        "masked_sam_deg": 0.0,
        "masked_sid": 0.0,
        "mse": 0.0,
        "mae": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "sam_deg": 0.0,
        "sid": 0.0,
        "invalid_mae": 0.0,
    }
    num_batches = 0
    start_time = time.perf_counter()

    use_progress = show_progress and is_main_process()
    progress = (
        tqdm(loader, desc=f"Train {epoch}/{total_epochs}", leave=False) if use_progress else loader
    )

    for batch in progress:
        if isinstance(batch, dict):
            x = batch["x"].to(device, non_blocking=True)
            mask = batch.get("valid_mask")
            mask = mask.to(device, non_blocking=True) if mask is not None else None
        else:
            x = batch.to(device, non_blocking=True)
            mask = None

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            enabled=use_amp,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        ):
            try:
                outputs = model(x, valid_mask=mask)
            except TypeError:
                outputs = model(x)
            x_hat = outputs.get("x_hat_for_loss", outputs["x_hat"]).float()
            x_target = outputs.get("x_target", x)
            mask_for_loss = outputs.get("mask_for_loss", mask)
            likelihoods = outputs.get("likelihoods")

            if likelihoods is not None and isinstance(loss_fn, RateDistortionLoss):
                loss, D, R = loss_fn(x_hat, x_target, mask_for_loss, likelihoods)
            else:
                loss = loss_fn(x_hat, x_target, mask_for_loss)
                D, R = loss, torch.tensor(0.0)

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            if grad_clip_max_norm > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_max_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            optimizer.step()

        if aux_optimizer is not None:
            aux_loss = sum(
                m.loss()
                for m in model.modules()
                if hasattr(m, "loss") and isinstance(m, EntropyBottleneck)
            )
            if aux_loss != 0 and aux_loss is not None:
                aux_optimizer.zero_grad()
                aux_loss.backward()
                aux_optimizer.step()

        with torch.no_grad():
            masked_mse_val = (
                masked_mse(x_hat, x_target, mask_for_loss)
                if mask_for_loss is not None
                else torch.mean((x_hat - x_target) ** 2)
            )
            masked_mae_val = (
                masked_mae(x_hat, x_target, mask_for_loss)
                if mask_for_loss is not None
                else torch.mean((x_hat - x_target).abs())
            )
            masked_psnr_val = (
                masked_psnr(x_hat, x_target, mask_for_loss, data_range=1.0)
                if mask_for_loss is not None
                else psnr(x_hat, x_target, data_range=1.0)
            )
            masked_sam_val = (
                masked_sam_deg(x_hat, x_target, mask_for_loss)
                if mask_for_loss is not None
                else sam_deg(x_hat, x_target)
            )
            mse_val = torch.mean((x_hat - x_target) ** 2)
            mae_val = mae(x_hat, x_target)
            psnr_val = psnr(x_hat, x_target, data_range=1.0)
            ssim_val = ssim(x_hat, x_target, data_range=1.0)
            sam_val = sam_deg(x_hat, x_target)
            masked_sid_val = (
                masked_sid(x_hat, x_target, mask_for_loss)
                if mask_for_loss is not None
                else sid(x_hat, x_target)
            )
            sid_val = sid(x_hat, x_target)
            invalid_mae_val = (
                invalid_region_mae(x_hat, mask_for_loss)
                if mask_for_loss is not None
                else torch.tensor(0.0, device=device)
            )

        metrics = {
            "loss": loss.item(),
            "D": D.item() if isinstance(D, torch.Tensor) else float(D),
            "R": R.item() if isinstance(R, torch.Tensor) else float(R),
            "masked_mse": masked_mse_val.item(),
            "masked_mae": masked_mae_val.item(),
            "masked_psnr": masked_psnr_val.item(),
            "masked_sam_deg": masked_sam_val.item(),
            "masked_sid": masked_sid_val.item(),
            "mse": mse_val.item(),
            "mae": mae_val.item(),
            "psnr": psnr_val.item(),
            "ssim": ssim_val.item(),
            "sam_deg": sam_val.item(),
            "sid": sid_val.item(),
            "invalid_mae": invalid_mae_val.item(),
        }
        for k, v in metrics.items():
            totals[k] += v
        num_batches += 1

        if use_progress:
            progress.set_postfix(
                {"loss": f"{loss.item():.5f}", "mPSNR": f"{masked_psnr_val.item():.2f}dB"}
            )

    n = max(num_batches, 1)
    out = {k: reduce_mean(v / n, device) for k, v in totals.items()}
    out["epoch_time_sec"] = reduce_mean(time.perf_counter() - start_time, device)
    return out
