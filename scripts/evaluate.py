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
    masked_mse,
    masked_psnr,
    masked_rmse,
    masked_sam_deg,
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
    parser.add_argument(
        "--quantization-bits",
        type=int,
        default=8,
        help="Bit depth of the latent representation for bpppc calculation",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def evaluate_model(
    model, loader, device, num_input_bands, quantization_bits, show_progress=True, split_name="eval"
):
    model.eval()

    total_loss = total_rmse = total_psnr = total_sam = total_bpppc = 0.0
    num_batches = 0
    latent_shape = None

    progress = tqdm(loader, desc=f"Evaluate [{split_name}]") if show_progress else loader

    for batch in progress:
        x = batch["x"].to(device)
        mask = batch["valid_mask"].to(device)

        outputs = model(x)
        x_hat = outputs["x_hat"]
        z = outputs.get("z")

        loss = masked_mse(x_hat, x, mask)
        rmse = masked_rmse(x_hat, x, mask)
        psnr = masked_psnr(x_hat, x, mask, data_range=1.0)
        sam = masked_sam_deg(x_hat, x, mask)

        total_loss += loss.item()
        total_rmse += rmse.item()
        total_psnr += psnr.item()
        total_sam += sam.item()
        num_batches += 1

        if z is not None:
            if latent_shape is None:
                latent_shape = tuple(z.shape[1:])
            total_bpppc += estimate_bpppc(
                z, num_bands=num_input_bands, quantization_bits=quantization_bits
            )

        if show_progress:
            progress.set_postfix(
                {
                    "psnr": f"{psnr.item():.2f}dB",
                    "sam": f"{sam.item():.2f}°",
                }
            )

    n = max(num_batches, 1)
    return {
        "loss": total_loss / n,
        "rmse": total_rmse / n,
        "psnr": total_psnr / n,
        "sam_deg": total_sam / n,
        "bpppc": total_bpppc / n,
        "latent_shape": latent_shape,
        "num_batches": num_batches,
    }


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

    model_name = model_section.get("model_name")
    model_kwargs = model_section.get("model_kwargs", {})
    difficulty = args.difficulty or data_section.get("difficulty", "easy")
    quant_bits = args.quantization_bits

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
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    sample_x = (ds[0] if not args.subset_size else ds.dataset[0])["x"]
    num_input_bands = sample_x.shape[0]
    print(f"Input bands: {num_input_bands}")

    model_kwargs["in_channels"] = num_input_bands

    model = build_model(
        model_name=model_name,
        in_channels=num_input_bands,
        **{k: v for k, v in model_kwargs.items() if k != "in_channels"},
    ).to(device)

    load_checkpoint(path=checkpoint_path, model=model, optimizer=None, map_location=device)

    metrics = evaluate_model(
        model=model,
        loader=loader,
        device=device,
        num_input_bands=num_input_bands,
        quantization_bits=quant_bits,
        show_progress=not args.no_progress,
        split_name=args.split,
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
    print(f"  Samples:     {len(ds)}")
    print(f"{'-' * 55}")
    print(f"  PSNR:       {metrics['psnr']:.4f} dB")
    print(f"  SAM:        {metrics['sam_deg']:.4f} °")
    print(f"  RMSE:       {metrics['rmse']:.6f}")
    print(f"  MSE:        {metrics['loss']:.6f}")
    print(f"  bpppc:      {metrics['bpppc']:.6f}")
    print(f"  Latent:     {metrics['latent_shape']}")
    print(f"  CR proxy:   {cr_proxy}")
    print(f"{'=' * 55}\n")

    result = {
        "checkpoint_path": str(checkpoint_path),
        "split": args.split,
        "difficulty": difficulty,
        "model_name": model_name,
        "num_samples": len(ds),
        "num_input_bands": num_input_bands,
        "quantization_bits": quant_bits,
        **metrics,
        "compression_ratio_proxy": cr_proxy,
    }

    if args.save_json:
        out = logs_dir() / f"eval_{model_name}_{difficulty}_{args.split}.json"
        with open(out, "w") as f:
            json.dump(
                {
                    k: str(v) if not isinstance(v, (int, float, str, type(None))) else v
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
                    "eval/psnr": metrics["psnr"],
                    "eval/sam_deg": metrics["sam_deg"],
                    "eval/rmse": metrics["rmse"],
                    "eval/bpppc": metrics["bpppc"],
                }
            )


if __name__ == "__main__":
    main()
