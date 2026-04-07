import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset
from torchmetrics.functional.image import structural_similarity_index_measure as tm_ssim
from tqdm.auto import tqdm

from hsi_compression.data import build_dataloader, build_dataset
from hsi_compression.engine.checkpointing import load_checkpoint
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
from hsi_compression.models.registry import build_model
from hsi_compression.paths import ensure_artifact_dirs, logs_dir
from hsi_compression.utils import load_project_env
from hsi_compression.utils.wandb_utils import init_wandb

ORIGINAL_BITS_PER_CHANNEL = 16.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a model checkpoint on a specified dataset split"
    )
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("dataset_root", nargs="?", default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--difficulty", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _call_model_forward(model, x, mask):
    try:
        return model(x, valid_mask=mask)
    except TypeError:
        return model(x)


def _call_model_compress(model, x, mask):
    try:
        return model.compress(x, valid_mask=mask)
    except TypeError:
        return model.compress(x)


def _call_model_decompress(model, packed, mask):
    if "latent" in packed:
        return model.decompress(latent=packed["latent"], z_shape=packed.get("z_shape"))

    kwargs = {
        "strings": packed["strings"],
        "shape": packed["shape"],
    }
    if "z_shape" in packed and packed["z_shape"] is not None:
        kwargs["z_shape"] = packed["z_shape"]

    _ = mask
    return model.decompress(**kwargs)


def _validate_packed_output(packed: dict):
    if not isinstance(packed, dict):
        raise RuntimeError("model.compress() must return a dict")
    if "latent" in packed:
        return
    if "strings" not in packed:
        raise RuntimeError("model.compress() output must contain 'strings'")
    if "shape" not in packed:
        raise RuntimeError("model.compress() output must contain 'shape'")
    if packed["strings"] is None:
        raise RuntimeError("model.compress() returned strings=None")


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device,
    show_progress=True,
    split_name="eval",
    use_amp=False,
):
    model.eval()

    def _get_proxy_bpppc(model_obj) -> float | None:
        model_raw = model_obj.module if hasattr(model_obj, "module") else model_obj
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
        "actual_masked_mse": 0.0,
        "actual_masked_mae": 0.0,
        "actual_masked_psnr": 0.0,
        "actual_masked_sam_deg": 0.0,
        "actual_masked_sid": 0.0,
        "actual_mse": 0.0,
        "actual_mae": 0.0,
        "actual_psnr": 0.0,
        "actual_sam_deg": 0.0,
        "actual_sid": 0.0,
        "actual_invalid_mae": 0.0,
        "actual_bpppc": 0.0,
        "actual_ssim": 0.0,
    }
    num_batches = 0
    latent_shape = None
    has_likelihoods = False
    actual_available = False

    inference_times = []
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    progress = tqdm(loader, desc=f"Evaluate [{split_name}]") if show_progress else loader

    for batch in progress:
        x = (
            batch["x"].to(device, non_blocking=True)
            if isinstance(batch, dict)
            else batch.to(device, non_blocking=True)
        )
        mask = batch.get("valid_mask") if isinstance(batch, dict) else None
        mask = mask.to(device, non_blocking=True) if mask is not None else None

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        ):
            if device.type == "cuda":
                start_event.record()

            outputs = _call_model_forward(model, x, mask)

            if device.type == "cuda":
                end_event.record()
                torch.cuda.synchronize()
                inference_times.append(start_event.elapsed_time(end_event))

        if not isinstance(outputs, dict):
            raise RuntimeError("Model output must be a dict containing at least 'x_hat'.")

        x_hat = outputs["x_hat"].float()
        z = outputs.get("z")
        likelihoods = outputs.get("likelihoods")

        totals["loss"] += (
            masked_mse(x_hat, x, mask).item()
            if mask is not None
            else torch.mean((x_hat - x) ** 2).item()
        )
        totals["masked_mse"] += (
            masked_mse(x_hat, x, mask) if mask is not None else torch.mean((x_hat - x) ** 2)
        ).item()
        totals["masked_mae"] += (
            masked_mae(x_hat, x, mask) if mask is not None else torch.mean((x_hat - x).abs())
        ).item()
        totals["masked_psnr"] += (
            masked_psnr(x_hat, x, mask, data_range=1.0) if mask is not None else psnr(x_hat, x)
        ).item()
        totals["psnr"] += psnr(x_hat, x, data_range=1.0).item()
        totals["ssim"] += ref_ssim(x_hat, x, data_range=1.0, channels=x.shape[1]).item()

        totals["masked_sam_deg"] += (
            masked_sam_deg(x_hat, x, mask) if mask is not None else sam_deg(x_hat, x)
        ).item()
        totals["masked_sid"] += (
            masked_sid(x_hat, x, mask) if mask is not None else sid(x_hat, x)
        ).item()
        totals["mse"] += torch.mean((x_hat - x) ** 2).item()
        totals["mae"] += mae(x_hat, x).item()

        totals["sam_deg"] += ref_sam_deg(x_hat, x).item()
        totals["sid"] += sid(x_hat, x).item()
        totals["invalid_mae"] += (
            invalid_region_mae(x_hat, mask)
            if mask is not None
            else torch.tensor(0.0, device=device)
        ).item()
        if likelihoods is not None:
            has_likelihoods = True
            totals["likelihood_bpppc"] += compute_true_bpppc(likelihoods, x.shape)
        model_proxy_bpppc = _get_proxy_bpppc(model)
        if model_proxy_bpppc is not None:
            totals["proxy_bpppc"] += model_proxy_bpppc
            totals["ref_bpppc"] += model_proxy_bpppc

        if latent_shape is None and z is not None:
            latent_shape = tuple(z.shape[1:])

        supports_actual = bool(
            getattr(
                model.module if hasattr(model, "module") else model,
                "supports_actual_compression",
                True,
            )
        )
        if supports_actual:
            packed = _call_model_compress(model, x, mask)
            _validate_packed_output(packed)
            decoded = _call_model_decompress(model, packed, mask)

            if not isinstance(decoded, dict) or "x_hat" not in decoded:
                raise RuntimeError("model.decompress() must return a dict containing 'x_hat'")

            x_hat_actual = decoded["x_hat"].float()
            actual_available = True

            totals["actual_masked_mse"] += (
                masked_mse(x_hat_actual, x, mask)
                if mask is not None
                else torch.mean((x_hat_actual - x) ** 2)
            ).item()
            totals["actual_masked_mae"] += (
                masked_mae(x_hat_actual, x, mask)
                if mask is not None
                else torch.mean((x_hat_actual - x).abs())
            ).item()
            totals["actual_masked_psnr"] += (
                masked_psnr(x_hat_actual, x, mask, data_range=1.0)
                if mask is not None
                else psnr(x_hat_actual, x)
            ).item()
            totals["actual_psnr"] += psnr(x_hat_actual, x, data_range=1.0).item()

            totals["actual_masked_sam_deg"] += (
                masked_sam_deg(x_hat_actual, x, mask)
                if mask is not None
                else sam_deg(x_hat_actual, x)
            ).item()
            totals["actual_masked_sid"] += (
                masked_sid(x_hat_actual, x, mask) if mask is not None else sid(x_hat_actual, x)
            ).item()
            totals["actual_mse"] += torch.mean((x_hat_actual - x) ** 2).item()
            totals["actual_mae"] += mae(x_hat_actual, x).item()

            with torch.no_grad():
                B, C, H, W = x.shape

                x_ssim = x.view(B * C, 1, H, W)
                x_hat_ssim = x_hat_actual.view(B * C, 1, H, W)

                _, ssim_map = tm_ssim(x_hat_ssim, x_ssim, data_range=1.0, return_full_image=True)

                if mask is not None:
                    mask_expanded = mask.expand(-1, C, -1, -1) if mask.shape[1] == 1 else mask

                    mask_ssim = mask_expanded.reshape(B * C, 1, H, W).bool()

                    if ssim_map.shape == mask_ssim.shape:
                        valid_ssim_values = ssim_map[mask_ssim]
                        if valid_ssim_values.numel() > 0:
                            batch_masked_ssim = valid_ssim_values.mean().item()
                        else:
                            batch_masked_ssim = 0.0
                    else:
                        batch_masked_ssim = ssim_map.mean().item()
                else:
                    batch_masked_ssim = ssim_map.mean().item()

            totals["actual_ssim"] = totals.get("actual_ssim", 0.0) + batch_masked_ssim
            totals["actual_sam_deg"] += sam_deg(x_hat_actual, x).item()
            totals["actual_sid"] += sid(x_hat_actual, x).item()
            totals["actual_invalid_mae"] += (
                invalid_region_mae(x_hat_actual, mask)
                if mask is not None
                else torch.tensor(0.0, device=device)
            ).item()
            if "strings" in packed:
                totals["actual_bpppc"] += compute_actual_bpppc_from_strings(
                    packed["strings"], x.shape
                )

        num_batches += 1

        if show_progress and actual_available:
            avg_actual_bpppc = totals["actual_bpppc"] / num_batches
            avg_actual_cr = compute_compression_ratio_from_bpppc(
                avg_actual_bpppc, ORIGINAL_BITS_PER_CHANNEL
            )
            progress.set_postfix(
                {
                    "mPSNR": f"{totals['actual_masked_psnr'] / num_batches:.2f}dB",
                    "act_bpppc": f"{avg_actual_bpppc:.4f}",
                    "CR(act)": f"{avg_actual_cr:.2f}:1" if avg_actual_cr is not None else "n/a",
                }
            )

    n = max(num_batches, 1)
    out = {k: v / n for k, v in totals.items()}
    out["actual_ssim"] = totals["actual_ssim"] / n
    out["latent_shape"] = latent_shape
    out["num_batches"] = num_batches
    if not has_likelihoods:
        out["likelihood_bpppc"] = None
    if not actual_available:
        for key in (
            "actual_masked_mse",
            "actual_masked_mae",
            "actual_masked_psnr",
            "actual_masked_sam_deg",
            "actual_masked_sid",
            "actual_mse",
            "actual_mae",
            "actual_psnr",
            "actual_sam_deg",
            "actual_sid",
            "actual_invalid_mae",
            "actual_bpppc",
            "actual_ssim",
        ):
            out[key] = None
    out["proxy_compression_ratio"] = compute_compression_ratio_from_bpppc(
        out["proxy_bpppc"], ORIGINAL_BITS_PER_CHANNEL
    )
    out["ref_compression_ratio"] = out["proxy_compression_ratio"]
    out["likelihood_compression_ratio"] = compute_compression_ratio_from_bpppc(
        out["likelihood_bpppc"], ORIGINAL_BITS_PER_CHANNEL
    )
    out["actual_compression_ratio"] = compute_compression_ratio_from_bpppc(
        out["actual_bpppc"], ORIGINAL_BITS_PER_CHANNEL
    )

    if len(inference_times) > 5:
        out["inference_ms_per_batch"] = sum(inference_times[5:]) / len(inference_times[5:])
    elif len(inference_times) > 0:
        out["inference_ms_per_batch"] = sum(inference_times) / len(inference_times)
    else:
        out["inference_ms_per_batch"] = 0.0

    return out


def main():
    load_project_env()
    args = parse_args()
    ensure_artifact_dirs()

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        print(f"Error: checkpoint does not exist: {checkpoint_path}")
        sys.exit(1)

    dataset_root = Path(
        args.dataset_root or os.environ.get("DATASET_ROOT") or "/data/hyspecnet-11k"
    )
    if not dataset_root.exists():
        print(f"Error: dataset_root does not exist: {dataset_root}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_config = ckpt_raw.get("config", {})
    model_section = ckpt_config.get("model", {})
    data_section = ckpt_config.get("data", {})
    training_section = ckpt_config.get("training", {})
    experiment_section = ckpt_config.get("experiment", {})

    model_name = model_section.get("model_name")
    model_kwargs = model_section.get("model_kwargs", {})
    difficulty = args.difficulty or data_section.get("difficulty", "easy")
    use_amp = training_section.get("use_amp", True) and device.type == "cuda"
    eval_seed = experiment_section.get("seed", 42)

    ds = build_dataset(
        dataset_root=dataset_root,
        split_name=args.split,
        difficulty=difficulty,
        normalized=True,
        return_mask=True,
        drop_invalid_channels=data_section.get("drop_invalid_channels", True),
        prefer_npy=True,
    )
    if args.subset_size:
        ds = Subset(ds, list(range(min(args.subset_size, len(ds)))))

    loader = build_dataloader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=eval_seed,
    )

    sample = ds[0] if not args.subset_size else ds.dataset[0]
    sample_x = sample["x"] if isinstance(sample, dict) else sample
    num_input_bands = sample_x.shape[0]

    print(f"Input bands: {num_input_bands}")
    print("Evaluation dataset source: split-resolved benchmark DATA.npy artifacts")
    print(f"Original bits per channel for CR estimation: {ORIGINAL_BITS_PER_CHANNEL:.0f}")

    model = build_model(
        model_name=model_name,
        in_channels=num_input_bands,
        **{k: v for k, v in model_kwargs.items() if k != "in_channels"},
    ).to(device)

    load_checkpoint(path=checkpoint_path, model=model, optimizer=None, map_location=device)

    if hasattr(model, "update"):
        model.update(force=True)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    metrics = evaluate_model(
        model=model,
        loader=loader,
        device=device,
        show_progress=not args.no_progress,
        split_name=args.split,
        use_amp=use_amp,
    )

    print(f"\n{'=' * 55}")
    print(f"  Model:        {model_name}")
    print(f"  Split:        {args.split} [{difficulty}]")
    print(f"  Samples:      {len(ds)}")
    print(f"  Params:       {num_params:,}")
    print(f"{'-' * 55}")
    print("  Reference metrics")
    print(f"  PSNR:         {metrics['psnr']:.4f} dB")
    print(f"  SSIM:         {metrics['ssim']:.4f}")
    print(f"  SA:           {metrics['sam_deg']:.4f} °")
    print(f"  proxy bpppc:  {metrics['proxy_bpppc']:.6f}")
    print(
        f"  Proxy CR:     {metrics['proxy_compression_ratio']:.4f}:1"
        if metrics["proxy_compression_ratio"] is not None
        else "  Ref. CR:      n/a"
    )
    print(f"{'-' * 55}")
    print("  Additional metrics")
    print(f"  mPSNR:        {metrics['masked_psnr']:.4f} dB")
    print(f"  mSAM:         {metrics['masked_sam_deg']:.4f} °")
    print(f"  mSID:         {metrics['masked_sid']:.6f}")
    print(
        f"  likel. bpppc: {metrics['likelihood_bpppc']:.6f}"
        if metrics["likelihood_bpppc"] is not None
        else "  likel. bpppc: n/a"
    )
    print(
        f"  Likel. CR:    {metrics['likelihood_compression_ratio']:.4f}:1"
        if metrics["likelihood_compression_ratio"] is not None
        else "  Likel. CR:    n/a"
    )
    print(
        f"  actual mPSNR: {metrics['actual_masked_psnr']:.4f} dB"
        if metrics["actual_masked_psnr"] is not None
        else "  actual mPSNR: n/a"
    )
    print(
        f"  actual mSSIM: {metrics['actual_ssim']:.4f}"
        if metrics["actual_ssim"] is not None
        else "  actual mSSIM: n/a"
    )
    print(
        f"  actual mSAM:  {metrics['actual_masked_sam_deg']:.4f} °"
        if metrics["actual_masked_sam_deg"] is not None
        else "  actual mSAM:  n/a"
    )
    print(
        f"  actual mSID:  {metrics['actual_masked_sid']:.6f}"
        if metrics["actual_masked_sid"] is not None
        else "  actual mSID:  n/a"
    )
    print(
        f"  actual mMAE:  {metrics['actual_masked_mae']:.6f}"
        if metrics["actual_masked_mae"] is not None
        else "  actual mMAE:  n/a"
    )
    print(
        f"  Actual bpppc: {metrics['actual_bpppc']:.6f}"
        if metrics["actual_bpppc"] is not None
        else "  Actual bpppc: n/a"
    )
    print(
        f"  Actual CR:    {metrics['actual_compression_ratio']:.4f}:1"
        if metrics["actual_compression_ratio"] is not None
        else "  Actual CR:    n/a"
    )
    print(f"{'-' * 55}")
    print(f"  Latent:       {metrics['latent_shape']}")
    print(f"  Infer Time:   {metrics['inference_ms_per_batch']:.2f} ms / batch")
    print(f"{'=' * 55}\n")

    result = {
        "checkpoint_path": str(checkpoint_path),
        "split": args.split,
        "difficulty": difficulty,
        "model_name": model_name,
        "num_samples": len(ds),
        "num_input_bands": num_input_bands,
        "num_params": num_params,
        "original_bits_per_channel": ORIGINAL_BITS_PER_CHANNEL,
        **metrics,
    }

    if args.save_json:
        out = logs_dir() / f"eval_{model_name}_{difficulty}_{args.split}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    k: (
                        list(v)
                        if isinstance(v, tuple)
                        else str(v)
                        if not isinstance(v, (int, float, str, type(None), list, dict))
                        else v
                    )
                    for k, v in result.items()
                },
                f,
                indent=2,
            )
        print(f"Saved: {out}")

    if not args.disable_wandb:
        log_payload = {
            "eval/ref_psnr": metrics["psnr"],
            "eval/ref_ssim": metrics["ssim"],
            "eval/ref_sa_deg": metrics["sam_deg"],
            "eval/proxy_bpppc": metrics["proxy_bpppc"],
            "eval/proxy_compression_ratio": metrics["proxy_compression_ratio"],
            "eval/ref_bpppc": metrics["ref_bpppc"],
            "eval/ref_compression_ratio": metrics["ref_compression_ratio"],
            "eval/likelihood_bpppc": metrics["likelihood_bpppc"],
            "eval/likelihood_compression_ratio": metrics["likelihood_compression_ratio"],
            "eval/masked_psnr": metrics["masked_psnr"],
            "eval/masked_sam_deg": metrics["masked_sam_deg"],
            "eval/masked_sid": metrics["masked_sid"],
            "eval/masked_mse": metrics["masked_mse"],
            "eval/psnr": metrics["psnr"],
            "eval/sam_deg": metrics["sam_deg"],
            "eval/sid": metrics["sid"],
            "eval/mse": metrics["mse"],
            "eval/invalid_mae": metrics["invalid_mae"],
            "eval/actual_ssim": metrics["actual_ssim"],
            "eval/actual_masked_psnr": metrics["actual_masked_psnr"],
            "eval/actual_masked_sam_deg": metrics["actual_masked_sam_deg"],
            "eval/actual_masked_sid": metrics["actual_masked_sid"],
            "eval/actual_masked_mse": metrics["actual_masked_mse"],
            "eval/actual_psnr": metrics["actual_psnr"],
            "eval/actual_sam_deg": metrics["actual_sam_deg"],
            "eval/actual_sid": metrics["actual_sid"],
            "eval/actual_mse": metrics["actual_mse"],
            "eval/actual_invalid_mae": metrics["actual_invalid_mae"],
            "eval/actual_bpppc": metrics["actual_bpppc"],
            "eval/actual_compression_ratio": metrics["actual_compression_ratio"],
            "eval/inference_ms_per_batch": metrics["inference_ms_per_batch"],
        }
        with init_wandb(
            project="hsi-compression-paper",
            run_name=args.run_name or f"eval_{model_name}_{args.split}",
            config=result,
        ) as run:
            run.log({k: v for k, v in log_payload.items() if v is not None})


if __name__ == "__main__":
    main()
