import time

import torch
from tqdm.auto import tqdm

from hsi_compression.metrics import (
    compute_actual_bpppc_from_strings,
    compute_compression_ratio_from_bpppc,
    compute_true_bpppc,
    invalid_region_mae,
    mae,
    masked_mae,
    masked_mse,
    masked_psnr,
    masked_sam_deg,
    masked_sid,
    psnr,
    ref_sam_deg,
    ref_ssim,
    sam_deg,
    sid,
)
from hsi_compression.utils.distributed import is_main_process, reduce_mean

ORIGINAL_BITS_PER_CHANNEL = 16.0


def _model_kwargs_from_batch(batch: dict, device: torch.device) -> dict:
    kwargs = {}
    for key in ("wavelengths", "output_wavelengths"):
        value = batch.get(key)
        if value is None:
            continue
        if torch.is_tensor(value):
            value = value.to(device, non_blocking=True)
        kwargs[key] = value
    return kwargs


def _call_model_compress(
    model,
    x: torch.Tensor,
    mask: torch.Tensor | None,
    wavelengths=None,
):
    # newer models accept masks during compression, older baselines do not.
    try:
        return model.compress(x, valid_mask=mask, wavelengths=wavelengths)
    except TypeError:
        try:
            return model.compress(x, valid_mask=mask)
        except TypeError:
            return model.compress(x)


def _call_model_decompress(model, packed: dict, output_wavelengths=None):
    kwargs = {"strings": packed["strings"], "shape": packed["shape"]}
    if "z_shape" in packed and packed["z_shape"] is not None:
        kwargs["z_shape"] = packed["z_shape"]
    if "output_channels" in packed and packed["output_channels"] is not None:
        kwargs["output_channels"] = packed["output_channels"]
    if output_wavelengths is not None:
        kwargs["output_wavelengths"] = output_wavelengths
    try:
        return model.decompress(**kwargs)
    except TypeError:
        kwargs.pop("output_wavelengths", None)
        return model.decompress(**kwargs)


def _supports_actual_compression(model) -> bool:
    model_raw = model.module if hasattr(model, "module") else model
    return bool(getattr(model_raw, "supports_actual_compression", False))


def _exact_reconstruction_target(model, x: torch.Tensor) -> torch.Tensor:
    model_raw = model.module if hasattr(model, "module") else model
    target_fn = getattr(model_raw, "exact_reconstruction_target", None)
    if callable(target_fn):
        return target_fn(x)
    return x


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    loss_fn,
    device: torch.device,
    epoch: int | None = None,
    total_epochs: int | None = None,
    show_progress: bool = True,
    compute_sam: bool = True,
    use_amp: bool = False,
    actual_codec_eval_batches: int = 0,
):
    model.eval()

    def _get_proxy_bpppc(model_obj) -> float | None:
        model_raw = model_obj.module if hasattr(model_obj, "module") else model_obj
        # proxy bitrate is only a diagnostic when no bitstream has been measured
        proxy = getattr(model_raw, "proxy_bpppc", None)
        if proxy is not None:
            return float(proxy)
        legacy = getattr(model_raw, "bpppc", None)
        return float(legacy) if legacy is not None else None

    totals = {
        "loss": 0.0,
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
        "proxy_bpppc": 0.0,
        "ref_bpppc": 0.0,
        "likelihood_bpppc": 0.0,
        "actual_bpppc": 0.0,
    }
    num_batches = 0
    actual_batches = 0
    latent_shape = None
    has_likelihoods = False
    actual_mismatch_count = 0
    actual_max_abs_error = 0.0
    encode_times_ms = []
    decode_times_ms = []
    start_time = time.perf_counter()
    run_actual_codec = actual_codec_eval_batches > 0 and _supports_actual_compression(model)

    use_progress = show_progress and is_main_process()
    desc = f"Val {epoch}/{total_epochs}" + ("" if compute_sam else " (fast)")
    progress = tqdm(loader, desc=desc, leave=False) if use_progress else loader

    for batch in progress:
        if isinstance(batch, dict):
            # validation accepts the same batch formats as training.
            x = batch["x"].to(device, non_blocking=True)
            mask = batch.get("valid_mask")
            mask = mask.to(device, non_blocking=True) if mask is not None else None
            model_kwargs = _model_kwargs_from_batch(batch, device)
        else:
            x = batch.to(device, non_blocking=True)
            mask = None
            model_kwargs = {}

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        ):
            try:
                outputs = model(x, valid_mask=mask, **model_kwargs)
            except TypeError:
                outputs = model(x)
            x_hat = outputs["x_hat"].float()
            x_hat_for_loss = outputs.get("x_hat_for_loss", x_hat).float()
            x_target = outputs.get("x_target", x)
            mask_for_loss = outputs.get("mask_for_loss", mask)
            # some models train on a transformed target but report metrics on the input domain
            metric_target = x_target if tuple(x_hat.shape) == tuple(x_target.shape) else x
            metric_mask = mask_for_loss if tuple(x_hat.shape) == tuple(x_target.shape) else mask
            z = outputs.get("z")
            likelihoods = outputs.get("likelihoods")
            if likelihoods is not None and hasattr(loss_fn, "lmbda"):
                loss_val, _, _ = loss_fn(x_hat_for_loss, x_target, mask_for_loss, likelihoods)
            else:
                loss_val = loss_fn(x_hat_for_loss, x_target, mask_for_loss)

        masked_mse_val = (
            masked_mse(x_hat, metric_target, metric_mask)
            if metric_mask is not None
            else torch.mean((x_hat - metric_target) ** 2)
        )
        masked_mae_val = (
            masked_mae(x_hat, metric_target, metric_mask)
            if metric_mask is not None
            else torch.mean((x_hat - metric_target).abs())
        )
        masked_psnr_val = (
            masked_psnr(x_hat, metric_target, metric_mask, data_range=1.0)
            if metric_mask is not None
            else psnr(x_hat, metric_target, data_range=1.0)
        )
        masked_sam_val = (
            # sam is optional because it is slower than pixelwise errors.
            masked_sam_deg(x_hat, metric_target, metric_mask)
            if (compute_sam and metric_mask is not None)
            else (sam_deg(x_hat, metric_target) if compute_sam else None)
        )
        mse_val = torch.mean((x_hat - metric_target) ** 2)
        mae_val = mae(x_hat, metric_target)
        psnr_val = psnr(x_hat, metric_target, data_range=1.0)
        ssim_val = ref_ssim(x_hat, metric_target, data_range=1.0, channels=x_hat.shape[1])
        sam_val = ref_sam_deg(x_hat, metric_target) if compute_sam else None
        invalid_mae_val = (
            invalid_region_mae(x_hat, metric_mask)
            if metric_mask is not None
            else torch.tensor(0.0, device=device)
        )

        totals["loss"] += loss_val.item()
        totals["masked_mse"] += masked_mse_val.item()
        totals["masked_mae"] += masked_mae_val.item()
        totals["masked_psnr"] += masked_psnr_val.item()
        totals["mse"] += mse_val.item()
        totals["mae"] += mae_val.item()
        totals["psnr"] += psnr_val.item()
        totals["ssim"] += ssim_val.item()
        totals["invalid_mae"] += invalid_mae_val.item()
        if compute_sam:
            # sid is grouped with sam because both are spectral metrics and relatively costly.
            totals["masked_sam_deg"] += masked_sam_val.item()
            totals["sam_deg"] += sam_val.item()
            totals["masked_sid"] += (
                masked_sid(x_hat, metric_target, metric_mask)
                if metric_mask is not None
                else sid(x_hat, metric_target)
            ).item()
            totals["sid"] += sid(x_hat, metric_target).item()

        num_batches += 1
        if z is not None:
            if latent_shape is None:
                latent_shape = tuple(z.shape[1:])

            if likelihoods is not None:
                # likelihood bitrate is estimated from the entropy model, not from real bytes
                has_likelihoods = True
                totals["likelihood_bpppc"] += compute_true_bpppc(likelihoods, x.shape)

        model_proxy_bpppc = _get_proxy_bpppc(model)
        if model_proxy_bpppc is not None:
            # proxy and ref fields are kept equal until an actual bitstream is measured.
            totals["proxy_bpppc"] += model_proxy_bpppc
            totals["ref_bpppc"] += model_proxy_bpppc

        if run_actual_codec and actual_batches < actual_codec_eval_batches:
            if device.type == "cuda":
                torch.cuda.synchronize()
            encode_start = time.perf_counter()
            packed = _call_model_compress(
                model,
                x,
                mask,
                wavelengths=model_kwargs.get("wavelengths"),
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            encode_times_ms.append((time.perf_counter() - encode_start) * 1000.0)

            if not isinstance(packed, dict) or "strings" not in packed or "shape" not in packed:
                raise RuntimeError("model.compress() must return strings and shape")

            if device.type == "cuda":
                torch.cuda.synchronize()
            decode_start = time.perf_counter()
            output_wavelengths = model_kwargs.get("output_wavelengths")
            if output_wavelengths is None:
                output_wavelengths = model_kwargs.get("wavelengths")
            decoded = _call_model_decompress(
                model,
                packed,
                output_wavelengths=output_wavelengths,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            decode_times_ms.append((time.perf_counter() - decode_start) * 1000.0)

            if not isinstance(decoded, dict) or "x_hat" not in decoded:
                raise RuntimeError("model.decompress() must return a dict containing 'x_hat'")

            x_hat_actual = decoded["x_hat"].to(device=device, dtype=x.dtype)
            exact_target = _exact_reconstruction_target(model, x).to(device=device, dtype=x.dtype)
            actual_mismatch_count += int((x_hat_actual != exact_target).sum().item())
            actual_max_abs_error = max(
                actual_max_abs_error,
                float((x_hat_actual - exact_target).abs().max().item()),
            )
            totals["actual_bpppc"] += compute_actual_bpppc_from_strings(
                packed["strings"], tuple(x.shape)
            )
            actual_batches += 1

        if use_progress:
            postfix = {"loss": f"{loss_val.item():.5f}", "mPSNR": f"{masked_psnr_val.item():.2f}dB"}
            if compute_sam:
                postfix["mSAM"] = f"{masked_sam_val.item():.2f}°"
            if actual_batches:
                avg_actual_bpppc = totals["actual_bpppc"] / actual_batches
                postfix["act_bpppc"] = f"{avg_actual_bpppc:.4f}"
                postfix["exact"] = str(actual_mismatch_count == 0)
            progress.set_postfix(postfix)

    n = max(num_batches, 1)
    # average each accumulated metric over validation batches.
    out = {
        k: reduce_mean(v / n, device)
        for k, v in totals.items()
        if k not in {"sam_deg", "masked_sam_deg", "sid", "masked_sid"} or compute_sam
    }
    if not compute_sam:
        out["sam_deg"] = None
        out["masked_sam_deg"] = None
        out["sid"] = None
        out["masked_sid"] = None
    if not has_likelihoods:
        # keep the field explicit so downstream reports do not confuse missing rate estimates
        out["likelihood_bpppc"] = None
    if actual_batches:
        avg_actual_bpppc = totals["actual_bpppc"] / actual_batches
        out["actual_bpppc"] = reduce_mean(avg_actual_bpppc, device)
        out["actual_compression_ratio"] = compute_compression_ratio_from_bpppc(
            out["actual_bpppc"], ORIGINAL_BITS_PER_CHANNEL
        )
        out["actual_exact_reconstruction"] = actual_mismatch_count == 0
        out["actual_mismatch_count"] = actual_mismatch_count
        out["actual_max_abs_error"] = actual_max_abs_error
        out["actual_codec_batches"] = actual_batches
        out["encode_ms_per_batch"] = (
            sum(encode_times_ms) / len(encode_times_ms) if encode_times_ms else None
        )
        out["decode_ms_per_batch"] = (
            sum(decode_times_ms) / len(decode_times_ms) if decode_times_ms else None
        )
    else:
        out["actual_bpppc"] = None
        out["actual_compression_ratio"] = None
        out["actual_exact_reconstruction"] = None
        out["actual_mismatch_count"] = None
        out["actual_max_abs_error"] = None
        out["actual_codec_batches"] = 0
        out["encode_ms_per_batch"] = None
        out["decode_ms_per_batch"] = None
    out["latent_shape"] = latent_shape
    out["epoch_time_sec"] = reduce_mean(time.perf_counter() - start_time, device)
    return out
