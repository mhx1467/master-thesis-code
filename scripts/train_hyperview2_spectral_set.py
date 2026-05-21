#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from hsi_compression.downstream import (
    HYPERVIEW2_TARGET_COLUMNS,
    Hyperview2PixelSetDataset,
    SpectralSetRegressor,
    Standardizer,
    build_hyperview2_samples,
    collate_pixel_set_batch,
    compute_regression_metrics,
    split_samples,
)
from hsi_compression.paths import artifacts_root
from hsi_compression.utils import load_project_env, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a HYPERVIEW2 spectral-set downstream regressor. The model encodes each "
            "pixel spectrum independently and pools the set of pixels with attention."
        )
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--id-column", type=str, default=None)
    parser.add_argument(
        "--modality",
        choices=("prisma", "sentinel2", "airborne", "any"),
        default="prisma",
    )
    parser.add_argument(
        "--normalization",
        choices=("percentile", "minmax", "none"),
        default="percentile",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--pixel-layers", type=int, default=3)
    parser.add_argument("--head-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--run-name",
        type=str,
        default="hyperview2_prisma_spectral_set",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def build_loader(
    dataset: Hyperview2PixelSetDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_pixel_set_batch,
        pin_memory=torch.cuda.is_available(),
    )


def collect_targets(dataset: Hyperview2PixelSetDataset) -> np.ndarray:
    return np.stack([sample.target for sample in dataset.samples]).astype(np.float32)


def run_epoch(
    model: SpectralSetRegressor,
    loader: DataLoader,
    target_standardizer: Standardizer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    loss_fn = torch.nn.MSELoss()
    for batch in loader:
        pixels = batch["pixels"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)
        target = torch.from_numpy(target_standardizer.transform(batch["target"].numpy()))
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(pixels, valid_mask), target)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def predict(
    model: SpectralSetRegressor,
    loader: DataLoader,
    target_standardizer: Standardizer,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    model.eval()
    preds = []
    targets = []
    sample_ids = []
    paths = []
    for batch in loader:
        pixels = batch["pixels"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)
        preds.append(model(pixels, valid_mask).cpu().numpy())
        targets.append(batch["target"].numpy())
        sample_ids.extend(batch["sample_id"])
        paths.extend(batch["path"])
    pred_norm = np.concatenate(preds, axis=0)
    pred = target_standardizer.inverse_transform(pred_norm)
    target = np.concatenate(targets, axis=0).astype(np.float32)
    return pred, target, sample_ids, paths


def write_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target", "mse", "mae", "rmse", "relative_mse", "baseline_mse"])
        for target, values in metrics["targets"].items():
            writer.writerow(
                [
                    target,
                    values["mse"],
                    values["mae"],
                    values["rmse"],
                    values["relative_mse"],
                    values["baseline_mse"],
                ]
            )


def main() -> int:
    load_project_env()
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    output_dir = args.output_dir or (artifacts_root() / "downstream" / args.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = build_hyperview2_samples(
        dataset_root=args.dataset_root,
        modality=args.modality,
        labels_csv=args.labels_csv,
        id_column=args.id_column,
        target_columns=HYPERVIEW2_TARGET_COLUMNS,
        max_samples=args.max_samples,
    )
    train_samples, val_samples = split_samples(
        samples, val_fraction=args.val_fraction, seed=args.seed
    )
    print(
        f"Samples: train={len(train_samples)} | val={len(val_samples)} | modality={args.modality}"
    )

    train_dataset = Hyperview2PixelSetDataset(
        train_samples,
        modality=args.modality,
        normalization=args.normalization,
        max_pixels=args.max_pixels,
    )
    val_dataset = Hyperview2PixelSetDataset(
        val_samples,
        modality=args.modality,
        normalization=args.normalization,
        max_pixels=args.max_pixels,
    )
    train_loader = build_loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_loader = build_loader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    y_train = collect_targets(train_dataset)
    y_val = collect_targets(val_dataset)
    target_standardizer = Standardizer.fit(y_train)
    baseline_mse = ((y_val - y_train.mean(axis=0, keepdims=True)) ** 2).mean(axis=0)
    in_channels = int(train_dataset[0]["pixels"].shape[-1])

    model = SpectralSetRegressor(
        in_channels=in_channels,
        output_dim=len(HYPERVIEW2_TARGET_COLUMNS),
        hidden_dim=args.hidden_dim,
        pixel_layers=args.pixel_layers,
        head_layers=args.head_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(5, args.patience // 4),
    )

    best_score = float("inf")
    best_epoch = 0
    best_metrics: dict[str, Any] | None = None
    best_val_ids: list[str] = []
    best_val_paths: list[str] = []
    checkpoint_path = output_dir / f"{args.run_name}_best.pt"
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            target_standardizer=target_standardizer,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
        )
        pred_val, y_val_eval, val_ids, val_paths = predict(
            model=model,
            loader=val_loader,
            target_standardizer=target_standardizer,
            device=device,
        )
        metrics = compute_regression_metrics(y_val_eval, pred_val, baseline_mse)
        score = float(metrics["hyperview_score"])
        scheduler.step(score)
        lr = float(optimizer.param_groups[0]["lr"])
        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.6f} | "
            f"val_hyperview_score={score:.6f} | val_mae={metrics['mean_mae']:.6f} | lr={lr:.2e}"
        )

        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = metrics
            best_val_ids = val_ids
            best_val_paths = val_paths
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "model_type": "spectral_set",
                        "run_name": args.run_name,
                        "modality": args.modality,
                        "normalization": args.normalization,
                        "target_columns": list(HYPERVIEW2_TARGET_COLUMNS),
                        "in_channels": in_channels,
                        "hidden_dim": args.hidden_dim,
                        "pixel_layers": args.pixel_layers,
                        "head_layers": args.head_layers,
                        "dropout": args.dropout,
                        "seed": args.seed,
                        "val_fraction": args.val_fraction,
                        "max_pixels": args.max_pixels,
                    },
                    "target_standardizer": target_standardizer.to_dict(),
                    "baseline_mse": baseline_mse.astype(np.float32).tolist(),
                    "train_ids": [sample.sample_id for sample in train_samples],
                    "val_ids": val_ids,
                    "train_paths": [str(sample.array_path) for sample in train_samples],
                    "val_paths": val_paths,
                    "best_metrics": metrics,
                    "best_epoch": epoch,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Early stopping after {args.patience} epochs without improvement.")
            break

    if best_metrics is None:
        raise RuntimeError("Training did not produce validation metrics.")

    result = {
        "run_name": args.run_name,
        "checkpoint": str(checkpoint_path),
        "dataset_root": str(args.dataset_root),
        "model_type": "spectral_set",
        "num_samples": len(samples),
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "val_ids": best_val_ids,
        "val_paths": best_val_paths,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_metrics_csv(output_dir / "metrics.csv", best_metrics)
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved metrics:    {output_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
