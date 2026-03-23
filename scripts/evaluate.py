import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset
from tqdm.auto import tqdm

from hsi_compression.data import build_dataloader, build_dataset
from hsi_compression.engine.checkpointing import load_checkpoint
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
from hsi_compression.models.registry import build_model
from hsi_compression.paths import ensure_artifact_dirs, logs_dir
from hsi_compression.utils import load_project_env
from hsi_compression.utils.wandb_utils import init_wandb


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a model checkpoint on a specified dataset split"
    )
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("dataset_root", nargs="?", default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--difficulty", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--quantization-bits", type=int, default=8)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device,
    num_input_bands,
    quantization_bits,
    show_progress=True,
    split_name="eval",
    use_amp=False,
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

            outputs = model(x)

            if device.type == "cuda":
                end_event.record()
                torch.cuda.synchronize()
                inference_times.append(start_event.elapsed_time(end_event))

            x_hat = outputs["x_hat"].float()
            z = outputs.get("z")

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
        totals["masked_sam_deg"] += (
            masked_sam_deg(x_hat, x, mask) if mask is not None else sam_deg(x_hat, x)
        ).item()
        totals["mse"] += torch.mean((x_hat - x) ** 2).item()
        totals["mae"] += mae(x_hat, x).item()
        totals["psnr"] += psnr(x_hat, x, data_range=1.0).item()
        totals["sam_deg"] += sam_deg(x_hat, x).item()
        totals["invalid_mae"] += (
            invalid_region_mae(x_hat, mask)
            if mask is not None
            else torch.tensor(0.0, device=device)
        ).item()
        num_batches += 1

        if z is not None:
            if latent_shape is None:
                latent_shape = tuple(z.shape[1:])
            totals["bpppc"] += estimate_bpppc(
                z,
                num_bands=num_input_bands,
                quantization_bits=quantization_bits,
            )

        if show_progress:
            progress.set_postfix(
                {
                    "mPSNR": f"{totals['masked_psnr'] / num_batches:.2f}dB",
                    "PSNR": f"{totals['psnr'] / num_batches:.2f}dB",
                }
            )

    n = max(num_batches, 1)
    out = {k: v / n for k, v in totals.items()}
    out["latent_shape"] = latent_shape
    out["num_batches"] = num_batches

    # first 5 batches may be outliers due to warmup, so we exclude them if we have enough batches
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_config = ckpt_raw.get("config", {})
    model_section = ckpt_config.get("model", {})
    data_section = ckpt_config.get("data", {})
    training_section = ckpt_config.get("training", {})

    model_name = model_section.get("model_name")
    model_kwargs = model_section.get("model_kwargs", {})
    difficulty = args.difficulty or data_section.get("difficulty", "easy")
    quant_bits = args.quantization_bits
    use_amp = training_section.get("use_amp", True) and device.type == "cuda"

    ds = build_dataset(
        dataset_root=dataset_root,
        split_name=args.split,
        difficulty=difficulty,
        normalized=True,
        return_mask=True,
        drop_invalid_channels=True,
    )
    if args.subset_size:
        ds = Subset(ds, list(range(min(args.subset_size, len(ds)))))

    loader = build_dataloader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    sample_x = ds[0] if not args.subset_size else ds.dataset[0]
    sample_x = sample_x["x"] if isinstance(sample_x, dict) else sample_x
    num_input_bands = sample_x.shape[0]
    print(f"Input bands: {num_input_bands}")

    model = build_model(
        model_name=model_name,
        in_channels=num_input_bands,
        **{k: v for k, v in model_kwargs.items() if k != "in_channels"},
    ).to(device)

    load_checkpoint(path=checkpoint_path, model=model, optimizer=None, map_location=device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    metrics = evaluate_model(
        model=model,
        loader=loader,
        device=device,
        num_input_bands=num_input_bands,
        quantization_bits=quant_bits,
        show_progress=not args.no_progress,
        split_name=args.split,
        use_amp=use_amp,
    )

    cr_proxy = None
    if hasattr(model, "compression_ratio_proxy") and metrics["latent_shape"]:
        cr_proxy = model.compression_ratio_proxy(
            input_shape=(num_input_bands, 128, 128),
            latent_shape=metrics["latent_shape"],
        )

    print(f"\n{'=' * 55}")
    print(f"  Model:      {model_name}")
    print(f"  Split:      {args.split} [{difficulty}]")
    print(f"  Samples:    {len(ds)}")
    print(f"  Params:     {num_params:,}")
    print(f"{'-' * 55}")
    print(f"  mPSNR:      {metrics['masked_psnr']:.4f} dB")
    print(f"  mSAM:       {metrics['masked_sam_deg']:.4f} °")
    print(f"  mMSE:       {metrics['masked_mse']:.6f}")
    print(f"  PSNR:       {metrics['psnr']:.4f} dB")
    print(f"  SAM:        {metrics['sam_deg']:.4f} °")
    print(f"  MSE:        {metrics['mse']:.6f}")
    print(f"  invalidMAE: {metrics['invalid_mae']:.6f}")
    print(f"  bpppc:      {metrics['bpppc']:.6f}")
    print(f"  Latent:     {metrics['latent_shape']}")
    print(f"  CR proxy:   {cr_proxy}")
    print(f"  Infer Time: {metrics['inference_ms_per_batch']:.2f} ms / batch")
    print(f"{'=' * 55}\n")

    result = {
        "checkpoint_path": str(checkpoint_path),
        "split": args.split,
        "difficulty": difficulty,
        "model_name": model_name,
        "num_samples": len(ds),
        "num_input_bands": num_input_bands,
        "quantization_bits": quant_bits,
        "num_params": num_params,
        **metrics,
        "compression_ratio_proxy": cr_proxy,
    }

    if args.save_json:
        out = logs_dir() / f"eval_{model_name}_{difficulty}_{args.split}.json"
        with open(out, "w") as f:
            json.dump(
                {
                    k: str(v)
                    if not isinstance(v, (int, float, str, type(None), list, dict, tuple))
                    else v
                    for k, v in result.items()
                },
                f,
                indent=2,
            )
        print(f"Saved: {out}")

    if not args.disable_wandb:
        with init_wandb(
            project="hsi-compression-paper",
            run_name=args.run_name or f"eval_{model_name}_{args.split}",
            config=result,
        ) as run:
            run.log(
                {
                    "eval/masked_psnr": metrics["masked_psnr"],
                    "eval/masked_sam_deg": metrics["masked_sam_deg"],
                    "eval/masked_mse": metrics["masked_mse"],
                    "eval/psnr": metrics["psnr"],
                    "eval/sam_deg": metrics["sam_deg"],
                    "eval/mse": metrics["mse"],
                    "eval/invalid_mae": metrics["invalid_mae"],
                    "eval/bpppc": metrics["bpppc"],
                    "eval/inference_ms_per_batch": metrics["inference_ms_per_batch"],
                }
            )


if __name__ == "__main__":
    main()
