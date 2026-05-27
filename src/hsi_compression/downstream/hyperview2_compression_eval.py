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

from hsi_compression.constants import CLEAN_BAND_COUNT, FULL_BAND_COUNT, WATER_VAPOR_BANDS
from hsi_compression.downstream.hyperview2 import (
    HYPERVIEW2_MODALITY_DIRS,
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
    spectral_mapping: str | None = None

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
    if name.endswith("_input_reflectance_0_1") or name.endswith("_input_hyspecnet"):
        return "reflectance_0_1"
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


def _hyperview2_wavelength_key(modality: str) -> str:
    modality_dir = HYPERVIEW2_MODALITY_DIRS.get(modality, modality)
    if modality_dir == "hsi_satellite":
        return "hsi_satellite_wavelengths"
    if modality_dir == "hsi_airborne":
        return "hsi_aerial_wavelengths"
    if modality_dir == "msi_satellite":
        return "msi_satellite_wavelengths"
    return f"{modality_dir}_wavelengths"


def load_hyperview2_wavelengths(
    source_root: str | Path,
    modality: str,
    channels: int,
) -> np.ndarray:
    path = Path(source_root) / "wavelengths.json"
    if not path.exists():
        return np.linspace(0.0, 1.0, channels, dtype=np.float32)

    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get(_hyperview2_wavelength_key(modality))
    if values is None:
        return np.linspace(0.0, 1.0, channels, dtype=np.float32)
    if isinstance(values, Mapping):
        wavelengths = np.asarray(
            [values[f"Band {idx}"] for idx in range(len(values))],
            dtype=np.float32,
        )
    else:
        wavelengths = np.asarray(values, dtype=np.float32)
    if wavelengths.size != channels:
        raise ValueError(f"Expected {channels} wavelengths for {modality}, got {wavelengths.size}")
    return wavelengths


def build_spectral_mapping(
    name: str | None,
    source_root: str | Path,
    modality: str,
    input_channels: int,
) -> dict[str, Any] | None:
    if name in (None, "", "none"):
        return None
    if name != "hyspecnet_202_approx":
        raise ValueError("spectral_mapping must be one of: none, hyspecnet_202_approx")

    source_wavelengths = load_hyperview2_wavelengths(
        source_root=source_root,
        modality=modality,
        channels=input_channels,
    )
    # This diagnostic approximates the HySpecNet/EnMAP 224-band pre-clean grid by spanning the
    # HYPERVIEW2 PRISMA wavelength range, then applies the repository's 22 invalid-band mask.
    full_wavelengths = np.linspace(
        float(source_wavelengths[0]),
        float(source_wavelengths[-1]),
        FULL_BAND_COUNT,
        dtype=np.float32,
    )
    clean_indices = np.asarray(
        [idx for idx in range(FULL_BAND_COUNT) if idx not in set(WATER_VAPOR_BANDS)],
        dtype=np.int64,
    )
    if clean_indices.size != CLEAN_BAND_COUNT:
        raise RuntimeError(f"Expected {CLEAN_BAND_COUNT} clean bands, got {clean_indices.size}")
    return {
        "name": name,
        "description": (
            "Approximate HYPERVIEW2 PRISMA 230 -> HySpecNet clean 202 -> PRISMA 230 "
            "spectral transfer using linear wavelength resampling and repository water-vapor "
            "band removal indices."
        ),
        "input_channels": int(input_channels),
        "model_input_channels": int(CLEAN_BAND_COUNT),
        "output_channels": int(input_channels),
        "source_wavelength_min": float(source_wavelengths[0]),
        "source_wavelength_max": float(source_wavelengths[-1]),
        "source_wavelengths": source_wavelengths.astype(float).tolist(),
        "hyspecnet_full_wavelengths_approx": full_wavelengths.astype(float).tolist(),
        "clean_indices": clean_indices.astype(int).tolist(),
        "removed_indices": list(WATER_VAPOR_BANDS),
    }


def _interp_indices(
    source_positions: Sequence[float],
    target_positions: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray(source_positions, dtype=np.float64)
    target = np.asarray(target_positions, dtype=np.float64)
    if source.ndim != 1 or target.ndim != 1:
        raise ValueError("source_positions and target_positions must be one-dimensional")
    if source.size < 2:
        raise ValueError("At least two source positions are required for interpolation")
    if np.any(np.diff(source) <= 0):
        raise ValueError("source_positions must be strictly increasing")

    hi = np.searchsorted(source, target, side="left")
    hi = np.clip(hi, 0, source.size - 1)
    lo = np.maximum(hi - 1, 0)
    below = target <= source[0]
    above = target >= source[-1]
    lo[below] = 0
    hi[below] = 0
    lo[above] = source.size - 1
    hi[above] = source.size - 1

    denom = source[hi] - source[lo]
    weights = np.divide(
        target - source[lo],
        denom,
        out=np.zeros_like(target, dtype=np.float64),
        where=denom != 0,
    )
    weights = np.clip(weights, 0.0, 1.0).astype(np.float32)
    return lo.astype(np.int64), hi.astype(np.int64), weights


def resample_spectral_tensor(
    x: torch.Tensor,
    source_positions: Sequence[float],
    target_positions: Sequence[float],
) -> torch.Tensor:
    lo, hi, weights = _interp_indices(source_positions, target_positions)
    lo_t = torch.as_tensor(lo, device=x.device, dtype=torch.long)
    hi_t = torch.as_tensor(hi, device=x.device, dtype=torch.long)
    weights_t = torch.as_tensor(weights, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    x_lo = x.index_select(1, lo_t)
    x_hi = x.index_select(1, hi_t)
    return x_lo.mul(1.0 - weights_t).add(x_hi.mul(weights_t))


def resample_spectral_mask(
    mask: torch.Tensor,
    source_positions: Sequence[float],
    target_positions: Sequence[float],
) -> torch.Tensor:
    lo, hi, _ = _interp_indices(source_positions, target_positions)
    lo_t = torch.as_tensor(lo, device=mask.device, dtype=torch.long)
    hi_t = torch.as_tensor(hi, device=mask.device, dtype=torch.long)
    return mask.index_select(1, lo_t) & mask.index_select(1, hi_t)


def apply_input_spectral_mapping(
    x: torch.Tensor,
    mask: torch.Tensor,
    mapping: Mapping[str, Any] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mapping is None:
        return x, mask
    source_wavelengths = mapping["source_wavelengths"]
    full_wavelengths = mapping["hyspecnet_full_wavelengths_approx"]
    clean_indices = torch.as_tensor(mapping["clean_indices"], device=x.device, dtype=torch.long)
    x_full = resample_spectral_tensor(x, source_wavelengths, full_wavelengths)
    mask_full = resample_spectral_mask(mask, source_wavelengths, full_wavelengths)
    return x_full.index_select(1, clean_indices), mask_full.index_select(1, clean_indices)


def invert_output_spectral_mapping(
    x: torch.Tensor,
    mapping: Mapping[str, Any] | None,
) -> torch.Tensor:
    if mapping is None:
        return x
    full_wavelengths = mapping["hyspecnet_full_wavelengths_approx"]
    clean_indices = np.asarray(mapping["clean_indices"], dtype=np.int64)
    clean_wavelengths = np.asarray(full_wavelengths, dtype=np.float32)[clean_indices]
    source_wavelengths = mapping["source_wavelengths"]
    x_full = resample_spectral_tensor(x, clean_wavelengths, full_wavelengths)
    return resample_spectral_tensor(x_full, full_wavelengths, source_wavelengths)


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
        pin_memory=device.type == "cuda",
        collate_fn=partial(
            collate_compression_batch,
            pad_multiple=checkpoint.pad_multiple,
            min_spatial_size=checkpoint.min_spatial_size,
        ),
    )
    first = dataset[0]
    input_channels = int(first["x"].shape[0])
    spectral_mapping = build_spectral_mapping(
        checkpoint.spectral_mapping,
        source_root=source_root,
        modality=checkpoint.modality,
        input_channels=input_channels,
    )
    in_channels = (
        int(spectral_mapping["model_input_channels"])
        if spectral_mapping is not None
        else input_channels
    )
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
        "model_input_coded_values": 0.0,
        "encode_time_sec": 0.0,
        "decode_time_sec": 0.0,
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"reconstruct:{variant_name}:{split}"):
            x = batch["x"].to(device, non_blocking=True)
            mask = batch["valid_mask"].to(device, non_blocking=True)
            x_model, mask_model = apply_input_spectral_mapping(x, mask, spectral_mapping)
            start_encode = time.perf_counter()
            if (
                checkpoint.use_bitstream
                and hasattr(model, "compress")
                and hasattr(model, "decompress")
            ):
                packed = call_model_compress(model, x_model, mask_model)
                encode_time = time.perf_counter() - start_encode
                start_decode = time.perf_counter()
                decoded = call_model_decompress(model, packed)
                decode_time = time.perf_counter() - start_decode
                x_hat_model = decoded["x_hat"] if isinstance(decoded, dict) else decoded
                if isinstance(packed, Mapping) and packed.get("strings") is not None:
                    try:
                        totals["actual_bits"] += float(sum_string_bytes(packed["strings"]) * 8)
                        totals["coded_values"] += float(
                            sum(c * h * w for c, h, w in batch["original_shape"])
                        )
                        totals["model_input_coded_values"] += float(
                            sum(x_model.shape[1] * h * w for _, h, w in batch["original_shape"])
                        )
                    except Exception as exc:  # pragma: no cover - diagnostic path
                        print("actual_bpppc skipped:", exc)
            else:
                with torch.autocast(
                    device_type=device.type,
                    enabled=use_amp and device.type == "cuda",
                ):
                    outputs = call_model_forward(model, x_model, mask_model)
                encode_time = time.perf_counter() - start_encode
                decode_time = 0.0
                x_hat_model = outputs["x_hat"] if isinstance(outputs, dict) else outputs

            x_hat = invert_output_spectral_mapping(x_hat_model.float(), spectral_mapping)
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
    actual_bpppc_model_input = None
    if totals["coded_values"] > 0:
        actual_bpppc = totals["actual_bits"] / totals["coded_values"]
    if totals["model_input_coded_values"] > 0:
        actual_bpppc_model_input = totals["actual_bits"] / totals["model_input_coded_values"]
    summary = {
        "name": checkpoint.name,
        "variant": variant_name,
        "checkpoint_path": str(checkpoint_path),
        "recon_root": str(recon_root),
        "input_modality": checkpoint.modality,
        "input_normalization": resolved_normalization,
        "input_channels": input_channels,
        "model_input_channels": in_channels,
        "output_channels": input_channels,
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
        "actual_bpppc_model_input": actual_bpppc_model_input,
        "actual_cr_16bit": compute_compression_ratio_from_bpppc(actual_bpppc),
        "actual_cr_16bit_model_input": compute_compression_ratio_from_bpppc(
            actual_bpppc_model_input
        ),
        "encode_time_sec": totals["encode_time_sec"],
        "decode_time_sec": totals["decode_time_sec"],
        "checkpoint_config": cfg,
        "adapter_notes": adapter_notes,
        "spectral_mapping": spectral_mapping,
    }
    (recon_root / "reconstruction_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    return recon_root, summary


def reconstruct_spectral_resample_passthrough(
    source_root: str | Path,
    recon_parent: str | Path,
    device: torch.device,
    variant_name: str = "hyperview2_spectral_resample_passthrough_hyspecnet202_to_230",
    modality: str = "prisma",
    normalization: str = "reflectance_0_1",
    spectral_mapping_name: str = "hyspecnet_202_approx",
    batch_size: int = 16,
    num_workers: int = 2,
    pad_multiple: int = 4,
    min_spatial_size: int = 4,
    split: str = "train",
) -> tuple[Path, dict[str, Any]]:
    """Write a no-codec spectral resampling baseline as HYPERVIEW2 reconstruction files."""
    print(
        f"Reconstructing spectral resample passthrough: variant={variant_name}, "
        f"input_norm={normalization}, mapping={spectral_mapping_name}"
    )
    samples = build_hyperview2_samples(source_root, modality=modality, split=split)
    dataset = Hyperview2CompressionDataset(
        samples,
        modality=modality,
        normalization=normalization,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=partial(
            collate_compression_batch,
            pad_multiple=pad_multiple,
            min_spatial_size=min_spatial_size,
        ),
    )
    first = dataset[0]
    input_channels = int(first["x"].shape[0])
    spectral_mapping = build_spectral_mapping(
        spectral_mapping_name,
        source_root=source_root,
        modality=modality,
        input_channels=input_channels,
    )
    if spectral_mapping is None:
        raise ValueError("spectral_mapping_name must resolve to a mapping for passthrough")
    model_input_channels = int(spectral_mapping["model_input_channels"])

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
        "encode_time_sec": 0.0,
        "decode_time_sec": 0.0,
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"resample-passthrough:{variant_name}:{split}"):
            x = batch["x"].to(device, non_blocking=True)
            mask = batch["valid_mask"].to(device, non_blocking=True)
            start = time.perf_counter()
            x_model, _ = apply_input_spectral_mapping(x, mask, spectral_mapping)
            x_hat = invert_output_spectral_mapping(x_model.float(), spectral_mapping)
            totals["encode_time_sec"] += time.perf_counter() - start
            x_hat = x_hat.float().clamp(0.0, 1.0)

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
    summary = {
        "name": "spectral_resample_passthrough",
        "variant": variant_name,
        "recon_root": str(recon_root),
        "baseline_type": "no_codec_spectral_resample_passthrough",
        "input_modality": modality,
        "input_normalization": normalization,
        "input_channels": input_channels,
        "model_input_channels": model_input_channels,
        "output_channels": input_channels,
        "recon_feature_normalization": "none",
        "saved_reconstruction_normalization": "none",
        "reconstruction_value_space": f"resample_inverse_for_{normalization}_input",
        "samples": totals["samples"],
        "masked_mse": totals["mse_sum"] / values,
        "masked_mae": totals["mae_sum"] / values,
        "masked_psnr": totals["psnr_sum"] / max(totals["metric_samples"], 1),
        "masked_sam_deg": totals["sam_sum"] / max(totals["metric_samples"], 1),
        "metric_samples": totals["metric_samples"],
        "actual_bpppc": None,
        "actual_bpppc_model_input": None,
        "actual_cr_16bit": None,
        "actual_cr_16bit_model_input": None,
        "encode_time_sec": totals["encode_time_sec"],
        "decode_time_sec": totals["decode_time_sec"],
        "adapter_notes": [],
        "spectral_mapping": spectral_mapping,
    }
    (recon_root / "reconstruction_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    return recon_root, summary


def _torch_spectral_gradient(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 1:
        return torch.zeros_like(values)
    gradient = torch.empty_like(values)
    gradient[0] = values[1] - values[0]
    gradient[-1] = values[-1] - values[-2]
    if values.numel() > 2:
        gradient[1:-1] = (values[2:] - values[:-2]) * 0.5
    return gradient


def _torch_spectral_features_from_normalized(
    cube: torch.Tensor,
    value_mask: torch.Tensor,
    feature_set: str,
) -> torch.Tensor:
    c, h, w = cube.shape
    flat = cube.reshape(c, h * w)
    mask_flat = value_mask.reshape(c, h * w).bool()
    valid = mask_flat.all(dim=0)
    valid_fraction = valid.float().mean() if valid.numel() else torch.zeros((), device=cube.device)
    valid_values = flat[:, valid]
    if valid_values.numel() == 0:
        mean = torch.zeros(c, dtype=torch.float32, device=cube.device)
        std = torch.zeros_like(mean)
    else:
        mean = valid_values.mean(dim=1)
        std = valid_values.std(dim=1, unbiased=False)
    features = [mean, std]

    if feature_set in {"mean_std_derivatives", "full_stats"}:
        features.extend([_torch_spectral_gradient(mean), _torch_spectral_gradient(std)])

    if feature_set == "full_stats":
        if valid_values.numel() == 0:
            zeros = torch.zeros(c, dtype=torch.float32, device=cube.device)
            features.extend([zeros, zeros, zeros, zeros, zeros])
        else:
            features.extend(
                [
                    valid_values.min(dim=1).values,
                    valid_values.max(dim=1).values,
                    torch.quantile(valid_values, 0.5, dim=1),
                    torch.quantile(valid_values, 0.25, dim=1),
                    torch.quantile(valid_values, 0.75, dim=1),
                ]
            )

    features.append(valid_fraction.reshape(1))
    return torch.cat(features).float()


def _make_feature_matrix_torch(
    samples: Sequence[Any],
    modality: str,
    normalization: str,
    feature_set: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    dataset = Hyperview2CompressionDataset(
        samples,
        modality=modality,
        normalization=normalization,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=partial(
            collate_compression_batch,
            pad_multiple=1,
            min_spatial_size=1,
        ),
    )
    targets_by_id = {str(sample.sample_id): sample.target.astype(np.float32) for sample in samples}
    xs, ys, ids = [], [], []
    desc = f"features:{modality}:{normalization}:{feature_set}:{device.type}"
    for batch in tqdm(loader, desc=desc):
        x = batch["x"].to(device, non_blocking=True)
        mask = batch["valid_mask"].to(device, non_blocking=True)
        batch_features = []
        for idx in range(x.shape[0]):
            c, h, w = batch["original_shape"][idx]
            batch_features.append(
                _torch_spectral_features_from_normalized(
                    x[idx, :c, :h, :w],
                    mask[idx, :c, :h, :w],
                    feature_set,
                )
            )
        xs.append(torch.stack(batch_features, dim=0).detach().cpu().numpy())
        batch_ids = [str(sample_id) for sample_id in batch["sample_id"]]
        ys.append(np.stack([targets_by_id[sample_id] for sample_id in batch_ids], axis=0))
        ids.extend(batch_ids)
    return np.concatenate(xs, axis=0).astype(np.float32), np.concatenate(ys, axis=0), ids


def make_feature_matrix(
    samples: Sequence[Any],
    modality: str,
    normalization: str,
    feature_set: str,
    feature_device: torch.device | None = None,
    batch_size: int = 64,
    num_workers: int = 2,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if feature_device is not None:
        return _make_feature_matrix_torch(
            samples,
            modality,
            normalization,
            feature_set,
            feature_device,
            batch_size=batch_size,
            num_workers=num_workers,
        )

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
    feature_device: torch.device | None = None,
    feature_batch_size: int = 64,
    feature_num_workers: int = 2,
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
        feature_device=feature_device,
        batch_size=feature_batch_size,
        num_workers=feature_num_workers,
    )
    x_val_orig, y_val, val_ids = make_feature_matrix(
        val_samples,
        modality,
        original_feature_normalization,
        feature_set,
        feature_device=feature_device,
        batch_size=feature_batch_size,
        num_workers=feature_num_workers,
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
            feature_device=feature_device,
            batch_size=feature_batch_size,
            num_workers=feature_num_workers,
        )
        x_val_recon, y_val_recon, _ = make_feature_matrix(
            recon_val_samples,
            modality,
            recon_feature_normalization,
            feature_set,
            feature_device=feature_device,
            batch_size=feature_batch_size,
            num_workers=feature_num_workers,
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
        "feature_device": str(feature_device) if feature_device is not None else "cpu_numpy",
        "feature_batch_size": feature_batch_size,
        "feature_num_workers": feature_num_workers,
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
    n_bands: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if max_samples is not None:
        sample_ids = list(sample_ids)[:max_samples]
    abs_sum: np.ndarray | None = None
    sq_sum: np.ndarray | None = None
    bias_sum: np.ndarray | None = None
    orig_sum: np.ndarray | None = None
    recon_sum: np.ndarray | None = None
    counts: np.ndarray | None = None
    sample_rows = []

    for sample_id in tqdm(sample_ids, desc=f"per-band:{Path(recon_root).parent.name}"):
        orig_path = sample_path(original_root, sample_id)
        recon_path = sample_path(recon_root, sample_id)
        if not orig_path.exists() or not recon_path.exists():
            continue
        orig, orig_mask = load_cube_and_value_mask(orig_path)
        recon, recon_mask = load_cube_and_value_mask(recon_path)
        orig = normalize_original_cube(orig, orig_mask, original_normalization)
        if n_bands is None and abs_sum is None:
            n_bands = min(orig.shape[0], recon.shape[0])
        elif n_bands is None:
            n_bands = int(abs_sum.shape[0])
        if abs_sum is None:
            abs_sum = np.zeros(n_bands, dtype=np.float64)
            sq_sum = np.zeros(n_bands, dtype=np.float64)
            bias_sum = np.zeros(n_bands, dtype=np.float64)
            orig_sum = np.zeros(n_bands, dtype=np.float64)
            recon_sum = np.zeros(n_bands, dtype=np.float64)
            counts = np.zeros(n_bands, dtype=np.float64)
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
        assert abs_sum is not None
        assert sq_sum is not None
        assert bias_sum is not None
        assert orig_sum is not None
        assert recon_sum is not None
        assert counts is not None
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

    if n_bands is None:
        n_bands = 0
    if abs_sum is None:
        abs_sum = np.zeros(n_bands, dtype=np.float64)
        sq_sum = np.zeros(n_bands, dtype=np.float64)
        bias_sum = np.zeros(n_bands, dtype=np.float64)
        orig_sum = np.zeros(n_bands, dtype=np.float64)
        recon_sum = np.zeros(n_bands, dtype=np.float64)
        counts = np.zeros(n_bands, dtype=np.float64)
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
