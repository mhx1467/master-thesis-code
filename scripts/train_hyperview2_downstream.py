#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from hsi_compression.downstream import (
    HYPERVIEW2_TARGET_COLUMNS,
    Hyperview2FeatureDataset,
    SpectralStatsRegressor,
    Standardizer,
    build_hyperview2_samples,
    collate_feature_batch,
    compute_regression_metrics,
    split_samples,
)
from hsi_compression.paths import artifacts_root
from hsi_compression.utils import load_project_env, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight HYPERVIEW2 downstream soil-property regressor."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--id-column", type=str, default=None)
    parser.add_argument(
        "--modality",
        choices=("prisma", "sentinel2", "airborne", "any"),
        default="prisma",
        help="Input modality. PRISMA is the safest main downstream modality.",
    )
    parser.add_argument(
        "--normalization",
        choices=("percentile", "minmax", "none"),
        default="percentile",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--run-name",
        type=str,
        default="hyperview2_spectral_stats_regressor_prisma",
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


def materialize_features(
    dataset: Hyperview2FeatureDataset,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_feature_batch,
    )
    features = []
    targets = []
    sample_ids = []
    paths = []
    for batch in tqdm(loader, desc="Extract features"):
        features.append(batch["features"].numpy())
        targets.append(batch["target"].numpy())
        sample_ids.extend(batch["sample_id"])
        paths.extend(batch["path"])
    return (
        np.concatenate(features, axis=0).astype(np.float32),
        np.concatenate(targets, axis=0).astype(np.float32),
        sample_ids,
        paths,
    )


def evaluate(
    model: torch.nn.Module,
    features: torch.Tensor,
    targets_original: np.ndarray,
    target_standardizer: Standardizer,
    baseline_mse: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, features.shape[0], 4096):
            batch = features[start : start + 4096].to(device)
            preds.append(model(batch).cpu().numpy())
    pred_norm = np.concatenate(preds, axis=0)
    pred = target_standardizer.inverse_transform(pred_norm)
    return compute_regression_metrics(targets_original, pred, baseline_mse)


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

    train_dataset = Hyperview2FeatureDataset(
        train_samples,
        modality=args.modality,
        normalization=args.normalization,
    )
    val_dataset = Hyperview2FeatureDataset(
        val_samples,
        modality=args.modality,
        normalization=args.normalization,
    )
    x_train, y_train, train_ids, train_paths = materialize_features(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    x_val, y_val, val_ids, val_paths = materialize_features(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    feature_standardizer = Standardizer.fit(x_train)
    target_standardizer = Standardizer.fit(y_train)
    baseline_mse = ((y_val - y_train.mean(axis=0, keepdims=True)) ** 2).mean(axis=0)

    x_train_t = torch.from_numpy(feature_standardizer.transform(x_train))
    y_train_t = torch.from_numpy(target_standardizer.transform(y_train))
    x_val_t = torch.from_numpy(feature_standardizer.transform(x_val))

    train_loader = DataLoader(
        TensorDataset(x_train_t, y_train_t),
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = SpectralStatsRegressor(
        input_dim=x_train_t.shape[1],
        output_dim=len(HYPERVIEW2_TARGET_COLUMNS),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()

    best_score = float("inf")
    best_metrics: dict[str, Any] | None = None
    checkpoint_path = output_dir / f"{args.run_name}_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        metrics = evaluate(
            model=model,
            features=x_val_t,
            targets_original=y_val,
            target_standardizer=target_standardizer,
            baseline_mse=baseline_mse,
            device=device,
        )
        train_loss = total_loss / max(len(train_loader), 1)
        score = float(metrics["hyperview_score"])
        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.6f} | "
            f"val_hyperview_score={score:.6f} | val_mae={metrics['mean_mae']:.6f}"
        )
        if score < best_score:
            best_score = score
            best_metrics = metrics
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "run_name": args.run_name,
                        "modality": args.modality,
                        "normalization": args.normalization,
                        "target_columns": list(HYPERVIEW2_TARGET_COLUMNS),
                        "input_dim": int(x_train_t.shape[1]),
                        "hidden_dim": args.hidden_dim,
                        "num_layers": args.num_layers,
                        "dropout": args.dropout,
                        "seed": args.seed,
                        "val_fraction": args.val_fraction,
                    },
                    "feature_standardizer": feature_standardizer.to_dict(),
                    "target_standardizer": target_standardizer.to_dict(),
                    "baseline_mse": baseline_mse.astype(np.float32).tolist(),
                    "train_ids": train_ids,
                    "val_ids": val_ids,
                    "train_paths": train_paths,
                    "val_paths": val_paths,
                    "best_metrics": metrics,
                    "epoch": epoch,
                },
                checkpoint_path,
            )

    if best_metrics is None:
        raise RuntimeError("Training did not produce validation metrics.")

    result = {
        "checkpoint": str(checkpoint_path),
        "dataset_root": str(args.dataset_root),
        "num_samples": len(samples),
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "best_metrics": best_metrics,
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
