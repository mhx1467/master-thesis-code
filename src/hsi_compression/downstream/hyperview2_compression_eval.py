from __future__ import annotations

import json
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from hsi_compression.downstream.hyperview2 import (
    HYPERVIEW2_TARGET_COLUMNS,
    Hyperview2CompressionDataset,
    Hyperview2FeatureDataset,
    build_hyperview2_samples,
    collate_compression_batch,
    compute_regression_metrics,
    load_array,
    load_mask,
    normalize_cube,
    split_samples,
)
from hsi_compression.downstream.hyperview2_regressors import build_hyperview2_regressor
from hsi_compression.engine.checkpointing import load_checkpoint
from hsi_compression.metrics import (
    compute_compression_ratio_from_bpppc,
    masked_psnr,
    masked_sam_deg,
)
from hsi_compression.models.registry import build_model


@dataclass(frozen=True)
class CompressionCheckpoint:
    name: str
    path: str | Path
    variant_name: str | None = None
    modality: str = "prisma"
    compression_normalization: str = "percentile"
    recon_feature_normalization: str = "none"
    batch_size: int = 1
    num_workers: int = 2
    use_bitstream: bool = True
    use_amp: bool | None = None
    pad_multiple: int = 4
    min_spatial_size: int = 4
    allow_in_channel_adapter: bool = True

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["path"] = str(self.path)
        return record


def safe_sample_stem(sample_id: str) -> str:
    return f"{int(sample_id):04d}" if str(sample_id).isdigit() else str(sample_id)


def safe_variant_component(value: str) -> str:
    return str(value).replace("/", "_").replace(" ", "_")


def discover_recon_roots(parent: str | Path) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    parent = Path(parent)
    if parent.exists():
        for path in sorted(parent.glob("*/HYPERVIEW2")):
            if (path / "train/hsi_satellite").is_dir():
                roots[path.parent.name] = path
    return roots


def prepare_recon_root(recon_root: str | Path, source_root: str | Path) -> Path:
    recon_root = Path(recon_root)
    source_root = Path(source_root)
    recon_root.mkdir(parents=True, exist_ok=True)
    for filename in ["train_gt.csv", "submission.csv", "wavelengths.json"]:
        src = source_root / filename
        if src.exists():
            shutil.copy2(src, recon_root / filename)
    for rel in ["train", "test"]:
        (recon_root / rel / "hsi_satellite").mkdir(parents=True, exist_ok=True)
        (recon_root / rel / "hsi_airborne").mkdir(parents=True, exist_ok=True)
        (recon_root / rel / "msi_satellite").mkdir(parents=True, exist_ok=True)
    return recon_root


def infer_recon_input_normalization(name: str, payload: Mapping[str, Any] | None = None) -> str:
    if payload:
        value = payload.get("input_normalization")
        if value:
            return str(value)
    if name.endswith("_input_percentile"):
        return "percentile"
    if name.endswith("_input_minmax"):
        return "minmax"
    return "none"


def read_reconstruction_summary(recon_root: str | Path) -> dict[str, Any] | None:
    path = Path(recon_root) / "reconstruction_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def call_model_forward(model: torch.nn.Module, x: torch.Tensor, mask: torch.Tensor) -> Any:
    try:
        return model(x, valid_mask=mask)
    except TypeError:
        return model(x)


def call_model_compress(model: torch.nn.Module, x: torch.Tensor, mask: torch.Tensor) -> Any:
    try:
        return model.compress(x, valid_mask=mask)
    except TypeError:
        return model.compress(x)


def call_model_decompress(model: torch.nn.Module, packed: Mapping[str, Any]) -> Any:
    if "latent" in packed:
        return model.decompress(latent=packed["latent"], z_shape=packed.get("z_shape"))
    kwargs = {"strings": packed["strings"], "shape": packed["shape"]}
    if packed.get("z_shape") is not None:
        kwargs["z_shape"] = packed["z_shape"]
    return model.decompress(**kwargs)


def sum_string_bytes(obj: Any) -> int:
    if isinstance(obj, bytes):
        return len(obj)
    if isinstance(obj, bytearray):
        return len(obj)
    if isinstance(obj, str):
        return len(obj.encode("utf-8"))
    if isinstance(obj, (list, tuple)):
        return sum(sum_string_bytes(item) for item in obj)
    raise TypeError(f"Unsupported strings container type: {type(obj)!r}")


def read_checkpoint_config(checkpoint_path: str | Path) -> dict[str, Any]:
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return raw.get("config", {})


def checkpoint_data_value(cfg: Mapping[str, Any], key: str, fallback: str) -> str:
    value = cfg.get("data", {}).get(key)
    if value in (None, "", "from_downstream"):
        return fallback
    return str(value)


def resolve_checkpoint_setting(
    setting: str,
    cfg: Mapping[str, Any],
    key: str,
    fallback: str,
) -> str:
    if setting == "checkpoint":
        return checkpoint_data_value(cfg, key=key, fallback=fallback)
    return str(setting)


def resize_tensor_along_dim(tensor: torch.Tensor, target_size: int, dim: int) -> torch.Tensor:
    if tensor.shape[dim] == target_size:
        return tensor
    old_size = tensor.shape[dim]
    if old_size == 1:
        return tensor.repeat_interleave(target_size, dim=dim)

    dim = dim % tensor.ndim
    perm = [dim] + [idx for idx in range(tensor.ndim) if idx != dim]
    inv_perm = [0] * len(perm)
    for idx, value in enumerate(perm):
        inv_perm[value] = idx

    moved = tensor.permute(perm).contiguous()
    trailing_shape = moved.shape[1:]
    flat = moved.reshape(old_size, -1).transpose(0, 1).unsqueeze(0)
    resized = F.interpolate(flat, size=target_size, mode="linear", align_corners=True)
    resized = resized.squeeze(0).transpose(0, 1).reshape((target_size, *trailing_shape))
    return resized.permute(inv_perm).contiguous().to(dtype=tensor.dtype)


def adapt_hyspecnet_202_state_dict_to_hyperview2(
    state_dict: Mapping[str, torch.Tensor],
    model: torch.nn.Module,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Adapt explicit 202-band HySpecNet parameters to 230-band HYPERVIEW2."""
    target_state = model.state_dict()
    adapted: dict[str, torch.Tensor] = {}
    notes: list[str] = []
    resize_rules = {
        "pos_embed": 1,
        "spatial_condition.conv1.weight": 1,
        "decoder.6.weight": 0,
        "decoder.6.bias": 0,
    }

    for key, value in state_dict.items():
        target = target_state.get(key)
        if target is None:
            continue
        if tuple(value.shape) == tuple(target.shape):
            adapted[key] = value
            continue
        if key not in resize_rules:
            raise RuntimeError(
                f"Unexpected checkpoint shape mismatch for {key}: "
                f"checkpoint={tuple(value.shape)} model={tuple(target.shape)}"
            )
        dim = resize_rules[key]
        compatible = all(
            value.shape[idx] == target.shape[idx] for idx in range(value.ndim) if idx != dim
        )
        if value.ndim != target.ndim or not compatible:
            raise RuntimeError(
                f"Cannot adapt {key}: checkpoint={tuple(value.shape)} model={tuple(target.shape)}"
            )
        adapted[key] = resize_tensor_along_dim(value, target.shape[dim], dim=dim)
        notes.append(f"{key}: {tuple(value.shape)} -> {tuple(target.shape)}")

    return adapted, notes


def build_model_from_checkpoint(
    checkpoint_path: str | Path,
    in_channels: int,
    device: torch.device,
    allow_in_channel_adapter: bool = True,
) -> tuple[torch.nn.Module, dict[str, Any], list[str]]:
    checkpoint_path = Path(checkpoint_path)
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = raw.get("config", {})
    model_section = cfg.get("model", {})
    model_name = model_section.get("model_name")
    if not model_name:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain config.model.model_name")
    model_kwargs = dict(model_section.get("model_kwargs", {}))
    checkpoint_in_channels = int(model_kwargs.get("in_channels", in_channels))
    model_kwargs.pop("in_channels", None)
    model = build_model(model_name, in_channels=in_channels, **model_kwargs).to(device)
    adapter_notes: list[str] = []

    if checkpoint_in_channels != in_channels:
        if not allow_in_channel_adapter:
            raise RuntimeError(
                f"Checkpoint has {checkpoint_in_channels} channels, input has {in_channels}. "
                "Enable allow_in_channel_adapter only for documented diagnostic transfer."
            )
        adapted_state, adapter_notes = adapt_hyspecnet_202_state_dict_to_hyperview2(
            raw["model_state_dict"],
            model,
        )
        missing, unexpected = model.load_state_dict(adapted_state, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys while loading adapted checkpoint: {unexpected}")
        if missing:
            raise RuntimeError(f"Missing keys while loading adapted checkpoint: {missing[:20]}")
    else:
        load_checkpoint(checkpoint_path, model=model, optimizer=None, map_location=device)

    if hasattr(model, "update"):
        model.update(force=True)
    model.eval()
    return model, cfg, adapter_notes


def reconstruct_checkpoint(
    checkpoint: CompressionCheckpoint,
    source_root: str | Path,
    recon_parent: str | Path,
    device: torch.device,
    checkpoint_normalization_fallback: str = "percentile",
    split: str = "train",
) -> tuple[Path, dict[str, Any]]:
    checkpoint_path = Path(checkpoint.path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint_cfg = read_checkpoint_config(checkpoint_path)
    resolved_normalization = resolve_checkpoint_setting(
        checkpoint.compression_normalization,
        checkpoint_cfg,
        key="normalization",
        fallback=checkpoint_normalization_fallback,
    )
    variant_name = checkpoint.variant_name or (
        f"{checkpoint.name}_input_{safe_variant_component(resolved_normalization)}"
    )
    use_amp = checkpoint.use_amp if checkpoint.use_amp is not None else device.type == "cuda"

    print(
        f"Reconstructing {checkpoint.name}: variant={variant_name}, "
        f"input_norm={resolved_normalization}, feature_norm={checkpoint.recon_feature_normalization}"
    )
    samples = build_hyperview2_samples(source_root, modality=checkpoint.modality, split=split)
    dataset = Hyperview2CompressionDataset(
        samples,
        modality=checkpoint.modality,
        normalization=resolved_normalization,
    )
    loader = DataLoader(
        dataset,
        batch_size=checkpoint.batch_size,
        shuffle=False,
        num_workers=checkpoint.num_workers,
        collate_fn=partial(
            collate_compression_batch,
            pad_multiple=checkpoint.pad_multiple,
            min_spatial_size=checkpoint.min_spatial_size,
        ),
    )
    first = dataset[0]
    in_channels = int(first["x"].shape[0])
    model, cfg, adapter_notes = build_model_from_checkpoint(
        checkpoint_path,
        in_channels=in_channels,
        device=device,
        allow_in_channel_adapter=checkpoint.allow_in_channel_adapter,
    )
    for note in adapter_notes:
        print("  adapted", note)

    recon_root = prepare_recon_root(Path(recon_parent) / variant_name / "HYPERVIEW2", source_root)
    out_dir = recon_root / split / "hsi_satellite"

    totals = {
        "mse_sum": 0.0,
        "mae_sum": 0.0,
        "values": 0.0,
        "psnr_sum": 0.0,
        "sam_sum": 0.0,
        "samples": 0,
        "metric_samples": 0,
        "actual_bits": 0.0,
        "coded_values": 0.0,
        "encode_time_sec": 0.0,
        "decode_time_sec": 0.0,
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"reconstruct:{variant_name}:{split}"):
            x = batch["x"].to(device, non_blocking=True)
            mask = batch["valid_mask"].to(device, non_blocking=True)
            start_encode = time.perf_counter()
            if (
                checkpoint.use_bitstream
                and hasattr(model, "compress")
                and hasattr(model, "decompress")
            ):
                packed = call_model_compress(model, x, mask)
                encode_time = time.perf_counter() - start_encode
                start_decode = time.perf_counter()
                decoded = call_model_decompress(model, packed)
                decode_time = time.perf_counter() - start_decode
                x_hat = decoded["x_hat"] if isinstance(decoded, dict) else decoded
                if isinstance(packed, Mapping) and packed.get("strings") is not None:
                    try:
                        totals["actual_bits"] += float(sum_string_bytes(packed["strings"]) * 8)
                        totals["coded_values"] += float(
                            sum(c * h * w for c, h, w in batch["original_shape"])
                        )
                    except Exception as exc:  # pragma: no cover - diagnostic path
                        print("actual_bpppc skipped:", exc)
            else:
                with torch.autocast(
                    device_type=device.type,
                    enabled=use_amp and device.type == "cuda",
                ):
                    outputs = call_model_forward(model, x, mask)
                encode_time = time.perf_counter() - start_encode
                decode_time = 0.0
                x_hat = outputs["x_hat"] if isinstance(outputs, dict) else outputs

            x_hat = x_hat.float().clamp(0.0, 1.0)
            totals["encode_time_sec"] += encode_time
            totals["decode_time_sec"] += decode_time

            mask_f = mask.float()
            valid_values = float(mask_f.sum().item())
            if valid_values > 0:
                psnr_value = masked_psnr(x_hat, x, mask)
                sam_value = masked_sam_deg(x_hat, x, mask)
                if torch.isfinite(psnr_value):
                    totals["psnr_sum"] += float(psnr_value.item()) * x.shape[0]
                if torch.isfinite(sam_value):
                    totals["sam_sum"] += float(sam_value.item()) * x.shape[0]
                totals["metric_samples"] += int(x.shape[0])
            totals["samples"] += int(x.shape[0])
            totals["mse_sum"] += float(((x_hat - x) ** 2 * mask_f).sum().item())
            totals["mae_sum"] += float(((x_hat - x).abs() * mask_f).sum().item())
            totals["values"] += valid_values

            for idx, sample_id in enumerate(batch["sample_id"]):
                c, h, w = batch["original_shape"][idx]
                arr = x_hat[idx, :c, :h, :w].detach().cpu().numpy().astype(np.float32)
                valid = mask[idx, :c, :h, :w].detach().cpu().numpy().astype(bool)
                np.savez_compressed(
                    out_dir / f"{safe_sample_stem(sample_id)}.npz", data=arr, mask=valid
                )

    values = max(totals["values"], 1.0)
    actual_bpppc = None
    if totals["coded_values"] > 0:
        actual_bpppc = totals["actual_bits"] / totals["coded_values"]
    summary = {
        "name": checkpoint.name,
        "variant": variant_name,
        "checkpoint_path": str(checkpoint_path),
        "recon_root": str(recon_root),
        "input_modality": checkpoint.modality,
        "input_normalization": resolved_normalization,
        "recon_feature_normalization": checkpoint.recon_feature_normalization,
        "saved_reconstruction_normalization": "none",
        "reconstruction_value_space": f"model_output_for_{resolved_normalization}_input",
        "samples": totals["samples"],
        "masked_mse": totals["mse_sum"] / values,
        "masked_mae": totals["mae_sum"] / values,
        "masked_psnr": totals["psnr_sum"] / max(totals["metric_samples"], 1),
        "masked_sam_deg": totals["sam_sum"] / max(totals["metric_samples"], 1),
        "metric_samples": totals["metric_samples"],
        "actual_bpppc": actual_bpppc,
        "actual_cr_16bit": compute_compression_ratio_from_bpppc(actual_bpppc),
        "encode_time_sec": totals["encode_time_sec"],
        "decode_time_sec": totals["decode_time_sec"],
        "checkpoint_config": cfg,
        "adapter_notes": adapter_notes,
    }
    (recon_root / "reconstruction_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    return recon_root, summary


def make_feature_matrix(
    samples: Sequence[Any],
    modality: str,
    normalization: str,
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    dataset = Hyperview2FeatureDataset(
        samples,
        modality=modality,
        normalization=normalization,
        feature_set=feature_set,
    )
    xs, ys, ids = [], [], []
    for idx in tqdm(range(len(dataset)), desc=f"features:{modality}:{normalization}:{feature_set}"):
        item = dataset[idx]
        xs.append(item["features"].numpy())
        ys.append(item["target"].numpy())
        ids.append(str(item["sample_id"]))
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32), ids


def samples_by_ids(root: str | Path, sample_ids: Sequence[str], modality: str) -> list[Any]:
    samples = build_hyperview2_samples(root, modality=modality, split="train")
    by_id = {sample.sample_id: sample for sample in samples}
    missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
    if missing:
        raise KeyError(f"Missing samples in {root}: {missing[:8]}")
    return [by_id[sample_id] for sample_id in sample_ids]


def prediction_rows_from_array(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_ids: Sequence[str],
    baseline_mse: np.ndarray,
    variant: str,
    mode: str,
    model: str,
    original_feature_normalization: str,
    recon_feature_normalization: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for sample_idx, sample_id in enumerate(sample_ids):
        for target_idx, target in enumerate(HYPERVIEW2_TARGET_COLUMNS):
            true_value = float(y_true[sample_idx, target_idx])
            pred_value = float(y_pred[sample_idx, target_idx])
            error = pred_value - true_value
            squared_error = error * error
            denom = float(baseline_mse[target_idx])
            rows.append(
                {
                    "variant": variant,
                    "source": variant,
                    "mode": mode,
                    "model": model,
                    "sample_id": str(sample_id),
                    "target": str(target),
                    "target_index": target_idx,
                    "original_feature_normalization": original_feature_normalization,
                    "recon_feature_normalization": recon_feature_normalization,
                    "y_true": true_value,
                    "y_pred": pred_value,
                    "error": error,
                    "abs_error": abs(error),
                    "squared_error": squared_error,
                    "relative_squared_error": squared_error / denom if denom > 0 else np.nan,
                }
            )
    return rows


def run_regressors(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    baseline_mse: np.ndarray,
    model_names: Sequence[str],
    variant: str,
    mode: str,
    original_feature_normalization: str,
    sample_ids: Sequence[str],
    n_jobs: int | None = -1,
    seed: int = 42,
    recon_feature_normalization: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    rows = []
    details: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for name in model_names:
        start = time.perf_counter()
        row = {
            "variant": variant,
            "source": variant,
            "mode": mode,
            "model": name,
            "original_feature_normalization": original_feature_normalization,
            "recon_feature_normalization": recon_feature_normalization,
        }
        try:
            regressor = build_hyperview2_regressor(
                name,
                random_state=seed,
                n_jobs=n_jobs,
                n_features=x_train.shape[1],
                n_samples=x_train.shape[0],
                n_targets=y_train.shape[1],
            )
            regressor.fit(x_train, y_train)
            fit_time = time.perf_counter() - start
            pred_start = time.perf_counter()
            y_pred = np.asarray(regressor.predict(x_val), dtype=np.float32)
            predict_time = time.perf_counter() - pred_start
            metrics = compute_regression_metrics(y_val, y_pred, baseline_mse)
            row.update(
                {
                    "status": "ok",
                    "hyperview_score": metrics["hyperview_score"],
                    "mean_mse": metrics["mean_mse"],
                    "mean_mae": metrics["mean_mae"],
                    "fit_time_sec": fit_time,
                    "predict_time_sec": predict_time,
                }
            )
            for target, target_metrics in metrics["targets"].items():
                row[f"{target}_rmse"] = target_metrics["rmse"]
                row[f"{target}_relative_mse"] = target_metrics["relative_mse"]
            details[name] = {
                "status": "ok",
                "metrics": metrics,
                "fit_time_sec": fit_time,
                "predict_time_sec": predict_time,
            }
            prediction_rows.extend(
                prediction_rows_from_array(
                    y_true=y_val,
                    y_pred=y_pred,
                    sample_ids=sample_ids,
                    baseline_mse=baseline_mse,
                    variant=variant,
                    mode=mode,
                    model=name,
                    original_feature_normalization=original_feature_normalization,
                    recon_feature_normalization=recon_feature_normalization,
                )
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            row.update(
                {"status": "failed", "error": str(exc), "fit_time_sec": time.perf_counter() - start}
            )
            details[name] = {"status": "failed", "error": str(exc)}
        rows.append(row)
    return pd.DataFrame(rows), details, prediction_rows


def evaluate_downstream_regressors(
    hv2_root: str | Path,
    recon_roots: Mapping[str, str | Path],
    recon_feature_normalizations: Mapping[str, str],
    model_names: Sequence[str],
    modality: str = "prisma",
    feature_set: str = "mean_std_derivatives",
    original_feature_normalization: str = "percentile",
    val_fraction: float = 0.2,
    seed: int = 42,
    n_jobs: int | None = -1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    original_samples = build_hyperview2_samples(hv2_root, modality=modality, split="train")
    train_samples, val_samples = split_samples(
        original_samples, val_fraction=val_fraction, seed=seed
    )
    x_train_orig, y_train, train_ids = make_feature_matrix(
        train_samples,
        modality,
        original_feature_normalization,
        feature_set,
    )
    x_val_orig, y_val, val_ids = make_feature_matrix(
        val_samples,
        modality,
        original_feature_normalization,
        feature_set,
    )
    baseline_mse = (
        ((y_val - y_train.mean(axis=0, keepdims=True)) ** 2).mean(axis=0).astype(np.float32)
    )

    all_tables = []
    all_details: dict[str, Any] = {}
    all_prediction_rows: list[dict[str, Any]] = []
    df, details, rows = run_regressors(
        x_train_orig,
        y_train,
        x_val_orig,
        y_val,
        baseline_mse,
        model_names,
        variant="original",
        mode="original_train_to_original_val",
        original_feature_normalization=original_feature_normalization,
        sample_ids=val_ids,
        n_jobs=n_jobs,
        seed=seed,
    )
    all_tables.append(df)
    all_details["original/original_train_to_original_val"] = details
    all_prediction_rows.extend(rows)

    for variant_name, recon_root in recon_roots.items():
        recon_feature_normalization = recon_feature_normalizations.get(variant_name, "none")
        recon_train_samples = samples_by_ids(recon_root, train_ids, modality)
        recon_val_samples = samples_by_ids(recon_root, val_ids, modality)
        x_train_recon, y_train_recon, _ = make_feature_matrix(
            recon_train_samples,
            modality,
            recon_feature_normalization,
            feature_set,
        )
        x_val_recon, y_val_recon, _ = make_feature_matrix(
            recon_val_samples,
            modality,
            recon_feature_normalization,
            feature_set,
        )

        df, details, rows = run_regressors(
            x_train_orig,
            y_train,
            x_val_recon,
            y_val_recon,
            baseline_mse,
            model_names,
            variant=variant_name,
            mode="original_train_to_recon_val",
            original_feature_normalization=original_feature_normalization,
            recon_feature_normalization=recon_feature_normalization,
            sample_ids=val_ids,
            n_jobs=n_jobs,
            seed=seed,
        )
        all_tables.append(df)
        all_details[f"{variant_name}/original_train_to_recon_val"] = details
        all_prediction_rows.extend(rows)

        df, details, rows = run_regressors(
            x_train_recon,
            y_train_recon,
            x_val_recon,
            y_val_recon,
            baseline_mse,
            model_names,
            variant=variant_name,
            mode="recon_train_to_recon_val",
            original_feature_normalization=original_feature_normalization,
            recon_feature_normalization=recon_feature_normalization,
            sample_ids=val_ids,
            n_jobs=n_jobs,
            seed=seed,
        )
        all_tables.append(df)
        all_details[f"{variant_name}/recon_train_to_recon_val"] = details
        all_prediction_rows.extend(rows)

    results_df = pd.concat(all_tables, ignore_index=True)
    results_df = results_df.sort_values(
        ["variant", "mode", "status", "hyperview_score"],
        na_position="last",
    ).reset_index(drop=True)
    predictions_df = pd.DataFrame(all_prediction_rows)
    if not predictions_df.empty:
        predictions_df = predictions_df.sort_values(
            ["variant", "mode", "model", "sample_id", "target_index"]
        ).reset_index(drop=True)
    protocol = {
        "dataset": "HYPERVIEW2",
        "dataset_root": str(hv2_root),
        "split_source": "train_gt.csv fixed internal train/validation split",
        "modality": modality,
        "feature_set": feature_set,
        "original_feature_normalization": original_feature_normalization,
        "recon_feature_normalizations": dict(recon_feature_normalizations),
        "val_fraction": val_fraction,
        "seed": seed,
        "target_columns": list(HYPERVIEW2_TARGET_COLUMNS),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "train_sample_ids": train_ids,
        "val_sample_ids": val_ids,
        "model_names": list(model_names),
        "recon_roots": {name: str(root) for name, root in recon_roots.items()},
    }
    payload = {
        "protocol": protocol,
        "baseline_mse": baseline_mse.tolist(),
        "downstream_details": all_details,
        "rows": results_df.to_dict(orient="records"),
    }
    return results_df, predictions_df, payload


def save_downstream_artifacts(
    output_dir: str | Path,
    results_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    payload: Mapping[str, Any],
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_csv": output_dir / "compression_downstream_summary.csv",
        "predictions_csv": output_dir / "compression_downstream_predictions.csv",
        "metrics_json": output_dir / "compression_downstream_metrics.json",
    }
    results_df.to_csv(paths["summary_csv"], index=False)
    predictions_df.to_csv(paths["predictions_csv"], index=False)
    paths["metrics_json"].write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return paths


def best_by_variant_mode(results_df: pd.DataFrame) -> pd.DataFrame:
    ok = results_df[results_df["status"].eq("ok") & results_df["hyperview_score"].notna()].copy()
    idx = ok.groupby(["variant", "mode"])["hyperview_score"].idxmin()
    return ok.loc[idx].sort_values(["variant", "mode"]).reset_index(drop=True)


def prediction_metrics_by_target(predictions_df: pd.DataFrame) -> pd.DataFrame:
    if predictions_df.empty:
        return pd.DataFrame()
    return (
        predictions_df.groupby(["variant", "mode", "model", "target"], dropna=False)
        .agg(
            samples=("sample_id", "nunique"),
            bias=("error", "mean"),
            mae=("abs_error", "mean"),
            mse=("squared_error", "mean"),
            relative_mse=("relative_squared_error", "mean"),
        )
        .reset_index()
        .assign(rmse=lambda df: np.sqrt(df["mse"]))
    )


def prediction_shift_decomposition(predictions_df: pd.DataFrame) -> pd.DataFrame:
    if predictions_df.empty:
        return pd.DataFrame()
    original = predictions_df[
        predictions_df["variant"].eq("original")
        & predictions_df["mode"].eq("original_train_to_original_val")
    ]
    recon = predictions_df[
        ~predictions_df["variant"].eq("original")
        & predictions_df["mode"].eq("original_train_to_recon_val")
    ]
    joined = recon.merge(
        original[["model", "sample_id", "target", "y_pred", "error", "abs_error", "squared_error"]],
        on=["model", "sample_id", "target"],
        how="inner",
        suffixes=("_recon", "_original"),
    )
    if joined.empty:
        return pd.DataFrame()
    joined["prediction_shift"] = joined["y_pred_recon"] - joined["y_pred_original"]
    joined["abs_prediction_shift"] = joined["prediction_shift"].abs()
    joined["extra_abs_error"] = joined["abs_error_recon"] - joined["abs_error_original"]
    joined["extra_squared_error"] = joined["squared_error_recon"] - joined["squared_error_original"]
    joined["cross_term"] = 2.0 * joined["error_original"] * joined["prediction_shift"]
    decomp = (
        joined.groupby(["variant", "model", "target"], dropna=False)
        .agg(
            samples=("sample_id", "nunique"),
            original_mse=("squared_error_original", "mean"),
            recon_mse=("squared_error_recon", "mean"),
            extra_mse=("extra_squared_error", "mean"),
            shift_mse=("prediction_shift", lambda s: float(np.mean(np.asarray(s) ** 2))),
            cross_term=("cross_term", "mean"),
            mean_shift=("prediction_shift", "mean"),
            mean_abs_shift=("abs_prediction_shift", "mean"),
            mean_extra_abs_error=("extra_abs_error", "mean"),
        )
        .reset_index()
    )
    decomp["mse_identity_residual"] = decomp["recon_mse"] - (
        decomp["original_mse"] + decomp["shift_mse"] + decomp["cross_term"]
    )
    return decomp


def load_cube_and_value_mask(
    path: str | Path, modality: str = "prisma"
) -> tuple[np.ndarray, np.ndarray]:
    cube = load_array(path, modality=modality)
    with np.load(path) as archive:
        mask_arr = np.asarray(archive["mask"]) if "mask" in archive.files else None
    if mask_arr is None:
        value_mask = np.isfinite(cube)
    elif mask_arr.shape == cube.shape:
        value_mask = mask_arr.astype(bool)
    else:
        spatial = load_mask(path, shape_hw=tuple(cube.shape[-2:]))
        value_mask = np.broadcast_to(spatial[None], cube.shape).copy()
    return cube.astype(np.float32, copy=False), value_mask.astype(bool, copy=False)


def normalize_original_cube(
    cube: np.ndarray, value_mask: np.ndarray, normalization: str
) -> np.ndarray:
    if normalization == "none":
        return np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    spatial_mask = value_mask.any(axis=0)
    return normalize_cube(cube, mask=spatial_mask, mode=normalization)


def sample_path(root: str | Path, sample_id: str, split: str = "train") -> Path:
    return Path(root) / split / "hsi_satellite" / f"{safe_sample_stem(sample_id)}.npz"


def compute_per_band_diagnostics(
    original_root: str | Path,
    recon_root: str | Path,
    sample_ids: Sequence[str],
    original_normalization: str,
    max_samples: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if max_samples is not None:
        sample_ids = list(sample_ids)[:max_samples]
    n_bands = 230
    abs_sum = np.zeros(n_bands, dtype=np.float64)
    sq_sum = np.zeros(n_bands, dtype=np.float64)
    bias_sum = np.zeros(n_bands, dtype=np.float64)
    orig_sum = np.zeros(n_bands, dtype=np.float64)
    recon_sum = np.zeros(n_bands, dtype=np.float64)
    counts = np.zeros(n_bands, dtype=np.float64)
    sample_rows = []

    for sample_id in tqdm(sample_ids, desc=f"per-band:{Path(recon_root).parent.name}"):
        orig_path = sample_path(original_root, sample_id)
        recon_path = sample_path(recon_root, sample_id)
        if not orig_path.exists() or not recon_path.exists():
            continue
        orig, orig_mask = load_cube_and_value_mask(orig_path)
        recon, recon_mask = load_cube_and_value_mask(recon_path)
        orig = normalize_original_cube(orig, orig_mask, original_normalization)
        c = min(orig.shape[0], recon.shape[0], n_bands)
        h = min(orig.shape[-2], recon.shape[-2])
        w = min(orig.shape[-1], recon.shape[-1])
        orig = orig[:c, :h, :w]
        recon = recon[:c, :h, :w]
        valid = orig_mask[:c, :h, :w] & recon_mask[:c, :h, :w]
        if not valid.any():
            sample_rows.append(
                {"sample_id": sample_id, "valid_values": 0, "mae": np.nan, "rmse": np.nan}
            )
            continue
        diff = recon - orig
        valid_f = valid.astype(np.float32)
        abs_sum[:c] += (np.abs(diff) * valid_f).sum(axis=(1, 2))
        sq_sum[:c] += ((diff**2) * valid_f).sum(axis=(1, 2))
        bias_sum[:c] += (diff * valid_f).sum(axis=(1, 2))
        orig_sum[:c] += (orig * valid_f).sum(axis=(1, 2))
        recon_sum[:c] += (recon * valid_f).sum(axis=(1, 2))
        counts[:c] += valid_f.sum(axis=(1, 2))
        sample_rows.append(
            {
                "sample_id": str(sample_id),
                "valid_values": int(valid.sum()),
                "original_normalization": original_normalization,
                "mae": float(np.abs(diff[valid]).mean()),
                "rmse": float(np.sqrt((diff[valid] ** 2).mean())),
            }
        )

    per_band = pd.DataFrame(
        {
            "band": np.arange(n_bands),
            "valid_count": counts,
            "original_normalization": original_normalization,
            "orig_mean": np.divide(
                orig_sum, counts, out=np.full(n_bands, np.nan), where=counts > 0
            ),
            "recon_mean": np.divide(
                recon_sum, counts, out=np.full(n_bands, np.nan), where=counts > 0
            ),
            "bias": np.divide(bias_sum, counts, out=np.full(n_bands, np.nan), where=counts > 0),
            "mae": np.divide(abs_sum, counts, out=np.full(n_bands, np.nan), where=counts > 0),
            "rmse": np.sqrt(
                np.divide(sq_sum, counts, out=np.full(n_bands, np.nan), where=counts > 0)
            ),
        }
    )
    return per_band, pd.DataFrame(sample_rows)
