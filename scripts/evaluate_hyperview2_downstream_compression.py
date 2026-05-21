#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from hsi_compression.downstream import (
    HYPERVIEW2_TARGET_COLUMNS,
    SpectralSetRegressor,
    SpectralStatsRegressor,
    Standardizer,
    build_hyperview2_samples,
    collate_pixel_set_batch,
    compute_regression_metrics,
    extract_spectral_stats,
)
from hsi_compression.downstream.hyperview2 import load_array, load_mask, normalize_cube
from hsi_compression.engine.checkpointing import load_checkpoint
from hsi_compression.models.registry import build_model
from hsi_compression.paths import artifacts_root
from hsi_compression.utils import load_project_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained HYPERVIEW2 downstream regressor on original inputs and "
            "compressor reconstructions."
        )
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--downstream-checkpoint", type=Path, required=True)
    parser.add_argument("--compressor-checkpoint", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--id-column", type=str, default=None)
    parser.add_argument("--split", choices=("val", "train", "all-labeled"), default="val")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--pad-multiple", type=int, default=4)
    parser.add_argument(
        "--reconstruction-mode",
        choices=("actual", "forward"),
        default="actual",
        help="'actual' uses compress/decompress; 'forward' uses model(x) reconstruction only.",
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


def _load_downstream_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    model_type = cfg.get("model_type", "spectral_stats")
    if model_type == "spectral_set":
        model = SpectralSetRegressor(
            in_channels=int(cfg["in_channels"]),
            output_dim=len(cfg.get("target_columns", HYPERVIEW2_TARGET_COLUMNS)),
            hidden_dim=int(cfg["hidden_dim"]),
            pixel_layers=int(cfg["pixel_layers"]),
            head_layers=int(cfg["head_layers"]),
            dropout=float(cfg["dropout"]),
        )
        feature_standardizer = None
    elif model_type == "spectral_stats":
        model = SpectralStatsRegressor(
            input_dim=int(cfg["input_dim"]),
            output_dim=len(cfg.get("target_columns", HYPERVIEW2_TARGET_COLUMNS)),
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            dropout=float(cfg["dropout"]),
        )
        feature_standardizer = Standardizer.from_dict(checkpoint["feature_standardizer"])
    else:
        raise ValueError(f"Unsupported downstream model_type: {model_type}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return {
        "checkpoint": checkpoint,
        "model": model,
        "config": cfg,
        "model_type": model_type,
        "feature_standardizer": feature_standardizer,
        "target_standardizer": Standardizer.from_dict(checkpoint["target_standardizer"]),
        "baseline_mse": np.asarray(checkpoint["baseline_mse"], dtype=np.float32),
    }


def _filter_samples(samples, checkpoint: dict[str, Any], split: str):
    if split == "all-labeled":
        return samples
    key = "val_ids" if split == "val" else "train_ids"
    wanted = set(checkpoint.get(key, []))
    selected = [sample for sample in samples if sample.sample_id in wanted]
    if not selected:
        raise ValueError(f"No samples matched checkpoint {key}.")
    return selected


def _build_compressor(checkpoint_path: Path, in_channels: int, device: torch.device):
    ckpt_raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt_raw.get("config", {})
    model_section = cfg.get("model", {})
    model_name = model_section.get("model_name")
    model_kwargs = model_section.get("model_kwargs", {})
    if not model_name:
        raise ValueError(f"Could not find model.model_name in checkpoint: {checkpoint_path}")

    model = build_model(model_name, in_channels=in_channels, **model_kwargs).to(device)
    try:
        load_checkpoint(checkpoint_path, model, map_location=device)
    except RuntimeError as exc:
        raise RuntimeError(
            "Compressor checkpoint is incompatible with the selected HYPERVIEW2 modality. "
            f"Sample has {in_channels} bands, but the checkpoint likely expects a different "
            "band count. Use a Hyperview2-specific compressor or a compatible modality."
        ) from exc
    if hasattr(model, "update"):
        model.update(force=True)
    model.eval()
    return model, cfg


def _pad_spatial(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    if multiple <= 1:
        return x, (x.shape[-2], x.shape[-1])
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (h, w)
    return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate"), (h, w)


def _call_forward(model, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    try:
        outputs = model(x, valid_mask=mask)
    except TypeError:
        outputs = model(x)
    if not isinstance(outputs, dict) or "x_hat" not in outputs:
        raise RuntimeError("Compressor forward output must be a dict containing x_hat.")
    return outputs["x_hat"].float()


def _call_actual(model, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    try:
        packed = model.compress(x, valid_mask=mask)
    except TypeError:
        packed = model.compress(x)
    if "latent" in packed:
        outputs = model.decompress(latent=packed["latent"], z_shape=packed.get("z_shape"))
    else:
        kwargs = {"strings": packed["strings"], "shape": packed["shape"]}
        if packed.get("z_shape") is not None:
            kwargs["z_shape"] = packed["z_shape"]
        outputs = model.decompress(**kwargs)
    if not isinstance(outputs, dict) or "x_hat" not in outputs:
        raise RuntimeError("Compressor decompress output must be a dict containing x_hat.")
    return outputs["x_hat"].float()


@torch.no_grad()
def reconstruct_cube(
    model,
    cube: np.ndarray,
    mask_np: np.ndarray | None,
    device: torch.device,
    pad_multiple: int,
    mode: str,
) -> np.ndarray:
    x = torch.from_numpy(cube).unsqueeze(0).to(device=device, dtype=torch.float32)
    mask = None
    if mask_np is not None:
        mask = torch.from_numpy(mask_np.astype(bool)).to(device=device)
        mask = mask.unsqueeze(0).unsqueeze(0).expand(1, x.shape[1], *mask.shape[-2:])
    x_pad, (h, w) = _pad_spatial(x, pad_multiple)
    mask_pad = None
    if mask is not None:
        mask_pad, _ = _pad_spatial(mask.float(), pad_multiple)
        mask_pad = mask_pad.bool()
    if mode == "actual":
        x_hat = _call_actual(model, x_pad, mask_pad)
    else:
        x_hat = _call_forward(model, x_pad, mask_pad)
    x_hat = x_hat[..., :h, :w].clamp(0.0, 1.0)
    return x_hat.squeeze(0).detach().cpu().numpy().astype(np.float32)


def _predict(
    model: torch.nn.Module,
    features: np.ndarray,
    feature_standardizer: Standardizer,
    target_standardizer: Standardizer,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    x = torch.from_numpy(feature_standardizer.transform(features))
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            preds.append(model(x[start : start + batch_size].to(device)).cpu().numpy())
    return target_standardizer.inverse_transform(np.concatenate(preds, axis=0))


def _cube_to_pixel_item(
    cube: np.ndarray,
    mask: np.ndarray | None,
    normalization: str,
    target: np.ndarray,
    sample_id: str,
    path: str,
) -> dict[str, Any]:
    cube = normalize_cube(cube, mask=mask, mode=normalization)
    c, h, w = cube.shape
    pixels = cube.reshape(c, h * w).T.astype(np.float32, copy=False)
    if mask is not None:
        valid = mask.reshape(h * w).astype(bool)
    else:
        valid = np.isfinite(pixels).all(axis=1)
    return {
        "pixels": torch.from_numpy(np.ascontiguousarray(pixels, dtype=np.float32)),
        "valid_mask": torch.from_numpy(np.ascontiguousarray(valid, dtype=bool)),
        "target": torch.from_numpy(target.astype(np.float32)),
        "sample_id": sample_id,
        "path": path,
    }


def _predict_pixel_sets(
    model: torch.nn.Module,
    items: list[dict[str, Any]],
    target_standardizer: Standardizer,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch = collate_pixel_set_batch(items[start : start + batch_size])
            pixels = batch["pixels"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            preds.append(model(pixels, valid_mask).cpu().numpy())
    return target_standardizer.inverse_transform(np.concatenate(preds, axis=0))


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    load_project_env()
    args = parse_args()
    device = select_device(args.device)

    downstream = _load_downstream_checkpoint(args.downstream_checkpoint, device)
    cfg = downstream["config"]
    modality = cfg["modality"]
    normalization = cfg["normalization"]
    output_dir = args.output_dir or (
        artifacts_root() / "downstream" / f"{cfg['run_name']}_compression_eval"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = build_hyperview2_samples(
        dataset_root=args.dataset_root,
        modality=modality,
        labels_csv=args.labels_csv,
        id_column=args.id_column,
        target_columns=cfg.get("target_columns", HYPERVIEW2_TARGET_COLUMNS),
        max_samples=args.max_samples,
    )
    samples = _filter_samples(samples, downstream["checkpoint"], args.split)
    print(f"Evaluating {len(samples)} samples from split={args.split}, modality={modality}")

    first_cube = load_array(samples[0].array_path, modality=modality)
    compressor, compressor_cfg = _build_compressor(
        args.compressor_checkpoint,
        in_channels=first_cube.shape[0],
        device=device,
    )

    original_features = []
    reconstructed_features = []
    original_pixel_items = []
    reconstructed_pixel_items = []
    targets = []
    sample_ids = []
    model_type = downstream["model_type"]
    for sample in tqdm(samples, desc="Reconstruct + extract downstream features"):
        cube_raw = load_array(sample.array_path, modality=modality)
        mask = load_mask(sample.mask_path, shape_hw=tuple(cube_raw.shape[-2:]))
        cube_norm = normalize_cube(cube_raw, mask=mask, mode=normalization)
        reconstructed = reconstruct_cube(
            model=compressor,
            cube=cube_norm,
            mask_np=mask,
            device=device,
            pad_multiple=args.pad_multiple,
            mode=args.reconstruction_mode,
        )
        if model_type == "spectral_set":
            original_pixel_items.append(
                _cube_to_pixel_item(
                    cube_raw,
                    mask=mask,
                    normalization=normalization,
                    target=sample.target,
                    sample_id=sample.sample_id,
                    path=str(sample.array_path),
                )
            )
            reconstructed_pixel_items.append(
                _cube_to_pixel_item(
                    reconstructed,
                    mask=mask,
                    normalization="none",
                    target=sample.target,
                    sample_id=sample.sample_id,
                    path=str(sample.array_path),
                )
            )
        else:
            original_features.append(
                extract_spectral_stats(cube_raw, mask=mask, normalization=normalization)
            )
            reconstructed_features.append(
                extract_spectral_stats(reconstructed, mask=mask, normalization="none")
            )
        targets.append(sample.target)
        sample_ids.append(sample.sample_id)

    targets_np = np.stack(targets).astype(np.float32)

    if model_type == "spectral_set":
        pred_original = _predict_pixel_sets(
            downstream["model"],
            original_pixel_items,
            downstream["target_standardizer"],
            batch_size=args.batch_size,
            device=device,
        )
        pred_reconstructed = _predict_pixel_sets(
            downstream["model"],
            reconstructed_pixel_items,
            downstream["target_standardizer"],
            batch_size=args.batch_size,
            device=device,
        )
    else:
        original_features_np = np.stack(original_features).astype(np.float32)
        reconstructed_features_np = np.stack(reconstructed_features).astype(np.float32)
        pred_original = _predict(
            downstream["model"],
            original_features_np,
            downstream["feature_standardizer"],
            downstream["target_standardizer"],
            batch_size=args.batch_size,
            device=device,
        )
        pred_reconstructed = _predict(
            downstream["model"],
            reconstructed_features_np,
            downstream["feature_standardizer"],
            downstream["target_standardizer"],
            batch_size=args.batch_size,
            device=device,
        )

    original_metrics = compute_regression_metrics(
        targets_np,
        pred_original,
        downstream["baseline_mse"],
        target_columns=cfg.get("target_columns", HYPERVIEW2_TARGET_COLUMNS),
    )
    reconstructed_metrics = compute_regression_metrics(
        targets_np,
        pred_reconstructed,
        downstream["baseline_mse"],
        target_columns=cfg.get("target_columns", HYPERVIEW2_TARGET_COLUMNS),
    )

    result = {
        "dataset_root": str(args.dataset_root),
        "downstream_checkpoint": str(args.downstream_checkpoint),
        "compressor_checkpoint": str(args.compressor_checkpoint),
        "compressor_config": compressor_cfg,
        "split": args.split,
        "modality": modality,
        "normalization": normalization,
        "downstream_model_type": model_type,
        "reconstruction_mode": args.reconstruction_mode,
        "num_samples": len(samples),
        "original_metrics": original_metrics,
        "reconstructed_metrics": reconstructed_metrics,
        "delta_hyperview_score": float(
            reconstructed_metrics["hyperview_score"] - original_metrics["hyperview_score"]
        ),
    }
    json_path = output_dir / "downstream_compression_eval.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _write_summary_csv(
        output_dir / "downstream_compression_eval.csv",
        [
            {
                "condition": "original",
                "hyperview_score": original_metrics["hyperview_score"],
                "mean_mse": original_metrics["mean_mse"],
                "mean_mae": original_metrics["mean_mae"],
            },
            {
                "condition": "reconstructed",
                "hyperview_score": reconstructed_metrics["hyperview_score"],
                "mean_mse": reconstructed_metrics["mean_mse"],
                "mean_mae": reconstructed_metrics["mean_mae"],
            },
        ],
    )
    print(
        "HYPERVIEW score: "
        f"original={original_metrics['hyperview_score']:.6f} | "
        f"reconstructed={reconstructed_metrics['hyperview_score']:.6f} | "
        f"delta={result['delta_hyperview_score']:+.6f}"
    )
    print(f"Saved: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
