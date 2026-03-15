from pathlib import Path
import os
import sys
import json
import argparse

import torch
import wandb
from torch.utils.data import Subset

from hsi_compression.data import build_dataset, build_dataloader
from hsi_compression.engine.checkpointing import load_checkpoint
from hsi_compression.losses import build_loss
from hsi_compression.metrics import masked_mse, masked_rmse, masked_psnr, masked_sam_deg
from hsi_compression.models.registry import build_model
from hsi_compression.paths import ensure_artifact_dirs, logs_dir
from hsi_compression.utils import load_project_env
from hsi_compression.utils.wandb_utils import init_wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HSI compression model checkpoint")

    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to checkpoint (.pt file)",
    )

    parser.add_argument(
        "dataset_root",
        nargs="?",
        default=None,
        help="Path to dataset root. If omitted, DATASET_ROOT env var will be used.",
    )

    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--difficulty", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--drop-invalid-channels", action="store_true")

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model override. Usually taken from checkpoint config.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default=None,
        help="Optional loss override. Usually taken from checkpoint config.",
    )

    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--save-json", action="store_true")

    return parser.parse_args()


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device: torch.device,
):
    model.eval()

    total_loss = 0.0
    total_rmse = 0.0
    total_psnr = 0.0
    total_sam_deg = 0.0
    num_batches = 0
    latent_shape = None

    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["valid_mask"].to(device)

        outputs = model(x)
        x_hat = outputs["x_hat"]
        z = outputs.get("z")

        loss = masked_mse(x_hat, x, mask)
        rmse = masked_rmse(x_hat, x, mask)
        psnr = masked_psnr(x_hat, x, mask, data_range=1.0)
        sam_deg = masked_sam_deg(x_hat, x, mask)

        total_loss += loss.item()
        total_rmse += rmse.item()
        total_psnr += psnr.item()
        total_sam_deg += sam_deg.item()
        num_batches += 1

        if latent_shape is None and z is not None:
            latent_shape = tuple(z.shape[1:])

    return {
        "loss": total_loss / max(num_batches, 1),
        "rmse": total_rmse / max(num_batches, 1),
        "psnr": total_psnr / max(num_batches, 1),
        "sam_deg": total_sam_deg / max(num_batches, 1),
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

    dataset_root_str = (
        args.dataset_root
        or os.environ.get("DATASET_ROOT")
        or "/home/brwsx/hsi-compression/pipelines/pull-dataset/hyspectnet-11k/hyspecnet-11k-full"
    )

    dataset_root = Path(dataset_root_str)
    if not dataset_root.exists():
        print(f"Error: dataset root does not exist: {dataset_root}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Read checkpoint metadata first
    checkpoint_raw = torch.load(checkpoint_path, map_location="cpu")
    ckpt_config = checkpoint_raw.get("config", {})

    model_name = args.model or ckpt_config.get("model_name")
    if model_name is None:
        print("Error: model name not found in checkpoint and not provided via --model")
        sys.exit(1)

    loss_name = args.loss or ckpt_config.get("loss_name", "masked_mse")
    difficulty = args.difficulty or ckpt_config.get("difficulty", "easy")
    drop_invalid_channels = args.drop_invalid_channels or ckpt_config.get("drop_invalid_channels", False)

    model_kwargs = ckpt_config.get("model_kwargs", {})
    if not model_kwargs:
        # backward-compatible fallback for older checkpoints
        model_kwargs = {
            "latent_channels": ckpt_config.get("latent_channels", 16),
            "hidden_channels": tuple(ckpt_config.get("hidden_channels", (128, 64))),
        }

    # 2. Build dataset
    full_ds = build_dataset(
        dataset_root=dataset_root,
        split_name=args.split,
        difficulty=difficulty,
        normalized=True,
        return_mask=True,
        drop_invalid_channels=drop_invalid_channels,
    )

    if args.subset_size is not None:
        subset_size = min(args.subset_size, len(full_ds))
        ds = Subset(full_ds, list(range(subset_size)))
    else:
        ds = full_ds

    loader = build_dataloader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    sample = full_ds[0]
    in_channels = sample["x"].shape[0]

    # 3. Build model and load weights
    model = build_model(
        model_name=model_name,
        in_channels=in_channels,
        **model_kwargs,
    ).to(device)

    load_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=None,
        map_location=device,
    )

    # 4. Evaluate
    metrics = evaluate_model(
        model=model,
        loader=loader,
        device=device,
    )

    # 5. Compression ratio proxy
    ratio = None
    if hasattr(model, "compression_ratio_proxy") and metrics["latent_shape"] is not None:
        ratio = model.compression_ratio_proxy(
            input_shape=(in_channels, 128, 128),
            latent_shape=metrics["latent_shape"],
        )

    result = {
        "checkpoint_path": str(checkpoint_path),
        "dataset_root": str(dataset_root),
        "split": args.split,
        "difficulty": difficulty,
        "model_name": model_name,
        "loss_name": loss_name,
        "num_samples": len(ds),
        "batch_size": args.batch_size,
        "device": str(device),
        "loss": metrics["loss"],
        "rmse": metrics["rmse"],
        "psnr": metrics["psnr"],
        "sam_deg": metrics["sam_deg"],
        "latent_shape": metrics["latent_shape"],
        "compression_ratio_proxy": ratio,
    }

    print("\nEvaluation results")
    print("------------------")
    print(f"model_name:               {result['model_name']}")
    print(f"split:                    {result['split']}")
    print(f"difficulty:               {result['difficulty']}")
    print(f"num_samples:              {result['num_samples']}")
    print(f"loss:                     {result['loss']:.6f}")
    print(f"rmse:                     {result['rmse']:.6f}")
    print(f"psnr:                     {result['psnr']:.6f}")
    print(f"sam_deg:                  {result['sam_deg']:.6f}")
    print(f"latent_shape:             {result['latent_shape']}")
    print(f"compression_ratio_proxy:  {result['compression_ratio_proxy']}")

    # 6. Save JSON
    if args.save_json:
        out_name = f"eval_{model_name}_{difficulty}_{args.split}.json"
        out_path = logs_dir() / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(f"Saved JSON results to: {out_path}")

    # 7. Optional W&B logging
    if not args.disable_wandb:
        run_name = args.run_name or f"eval_{model_name}_{difficulty}_{args.split}"

        wb_config = {
            "checkpoint_path": str(checkpoint_path),
            "dataset_root": str(dataset_root),
            "split": args.split,
            "difficulty": difficulty,
            "model_name": model_name,
            "loss_name": loss_name,
            "batch_size": args.batch_size,
            "subset_size": args.subset_size,
            "drop_invalid_channels": drop_invalid_channels,
            "model_kwargs": model_kwargs,
        }

        with init_wandb(
            project="hsi-compression",
            run_name=run_name,
            config=wb_config,
        ) as run:
            run.log({
                "eval/loss": result["loss"],
                "eval/rmse": result["rmse"],
                "eval/psnr": result["psnr"],
                "eval/sam_deg": result["sam_deg"],
            }, step=0)

            if result["latent_shape"] is not None:
                run.log({
                    "model/latent_c": result["latent_shape"][0],
                    "model/latent_h": result["latent_shape"][1],
                    "model/latent_w": result["latent_shape"][2],
                }, step=0)

            if ratio is not None:
                run.log({
                    "model/compression_ratio_proxy": ratio,
                }, step=0)

            run.summary["eval_loss"] = result["loss"]
            run.summary["eval_rmse"] = result["rmse"]
            run.summary["eval_psnr"] = result["psnr"]
            run.summary["eval_sam_deg"] = result["sam_deg"]
            run.summary["split"] = result["split"]
            run.summary["difficulty"] = result["difficulty"]
            run.summary["checkpoint_path"] = result["checkpoint_path"]


if __name__ == "__main__":
    main()