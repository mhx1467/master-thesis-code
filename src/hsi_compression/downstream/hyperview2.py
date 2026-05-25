from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

HYPERVIEW2_TARGET_COLUMNS = ("B", "Cu", "Zn", "Fe", "S", "Mn")
HYPERVIEW2_FEATURE_SETS = ("mean_std", "mean_std_derivatives", "full_stats")
HYPERVIEW2_MODALITY_DIRS = {
    "airborne": "hsi_airborne",
    "prisma": "hsi_satellite",
    "sentinel2": "msi_satellite",
}
EXPECTED_BANDS_BY_MODALITY = {
    "airborne": 430,
    "prisma": 230,
    "sentinel2": 13,
}
ID_COLUMN_CANDIDATES = (
    "id",
    "sample_index",
    "sampleindex",
    "field_id",
    "sample_id",
    "parcel_id",
    "name",
    "filename",
    "file",
    "path",
)


@dataclass(frozen=True)
class Hyperview2Sample:
    sample_id: str
    array_path: Path
    target: np.ndarray
    mask_path: Path | None = None


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-6

    @classmethod
    def fit(cls, values: np.ndarray, eps: float = 1e-6) -> Standardizer:
        # targets are standardized using train statistics so regression losses are balanced.
        mean = values.mean(axis=0).astype(np.float32)
        std = values.std(axis=0).astype(np.float32)
        std = np.maximum(std, eps).astype(np.float32)
        return cls(mean=mean, std=std, eps=eps)

    def transform(self, values: np.ndarray) -> np.ndarray:
        # standardization maps each target to roughly zero mean and unit variance.
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        # predictions are transformed back before reporting real target metrics.
        return (values * self.std + self.mean).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Standardizer:
        return cls(
            mean=np.asarray(data["mean"], dtype=np.float32),
            std=np.asarray(data["std"], dtype=np.float32),
            eps=float(data.get("eps", 1e-6)),
        )


class SpectralStatsRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = len(HYPERVIEW2_TARGET_COLUMNS),
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for _ in range(num_layers - 1):
            # hidden layers learn nonlinear relations between spectral statistics and soil values.
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpectralSetRegressor(nn.Module):
    """Regress soil parameters from a set of pixel spectra."""

    def __init__(
        self,
        in_channels: int,
        output_dim: int = len(HYPERVIEW2_TARGET_COLUMNS),
        hidden_dim: int = 256,
        pixel_layers: int = 3,
        head_layers: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if pixel_layers < 1:
            raise ValueError("pixel_layers must be >= 1")
        if head_layers < 1:
            raise ValueError("head_layers must be >= 1")

        encoder_layers: list[nn.Module] = []
        prev_dim = in_channels
        for _ in range(pixel_layers):
            # each pixel spectrum is encoded independently before set pooling.
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                ]
            )
            prev_dim = hidden_dim
        self.pixel_encoder = nn.Sequential(*encoder_layers)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # final representation contains attention pool, mean pool, and valid pixel fraction.
        head_input_dim = hidden_dim * 2 + 1
        head: list[nn.Module] = []
        prev_dim = head_input_dim
        for _ in range(head_layers - 1):
            head.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                ]
            )
            prev_dim = hidden_dim
        head.append(nn.Linear(prev_dim, output_dim))
        self.head = nn.Sequential(*head)

    def encode_set(
        self, pixels: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the pooled set representation before the final regression head."""
        if pixels.ndim != 3:
            raise ValueError(f"Expected pixels with shape (B, N, C), got {tuple(pixels.shape)}")
        valid_mask = torch.isfinite(pixels).all(dim=-1) if valid_mask is None else valid_mask.bool()
        if valid_mask.shape != pixels.shape[:2]:
            raise ValueError(
                f"valid_mask shape {tuple(valid_mask.shape)} does not match pixels {tuple(pixels.shape)}"
            )

        pixels = torch.nan_to_num(pixels, nan=0.0, posinf=0.0, neginf=0.0)
        empty_rows = ~valid_mask.any(dim=1)
        if empty_rows.any():
            # avoid all-masked rows because softmax over only invalid pixels would be undefined.
            valid_mask = valid_mask.clone()
            valid_mask[empty_rows, 0] = True

        encoded = self.pixel_encoder(pixels)
        mask_f = valid_mask.unsqueeze(-1).to(encoded.dtype)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        # mean pooling gives a stable global summary of all valid pixels.
        mean_pool = (encoded * mask_f).sum(dim=1) / denom

        # attention pooling lets the model emphasize informative pixels in the set
        scores = self.attention(encoded).squeeze(-1)
        scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        attn_pool = (encoded * weights).sum(dim=1)
        valid_fraction = valid_mask.to(encoded.dtype).mean(dim=1, keepdim=True)

        pooled = torch.cat([attn_pool, mean_pool, valid_fraction], dim=1)
        return pooled

    def forward(self, pixels: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.encode_set(pixels, valid_mask))


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _parse_float(value: Any) -> float:
    if value is None:
        raise ValueError("missing value")
    text = str(value).strip()
    if not text:
        raise ValueError("empty value")
    return float(text.replace(",", "."))


def _resolve_columns(
    headers: Sequence[str],
    target_columns: Sequence[str],
    id_column: str | None,
) -> tuple[str, list[str]]:
    by_key = {normalize_key(header): header for header in headers}
    target_resolved = []
    for column in target_columns:
        # resolve target names case-insensitively and ignoring punctuation.
        key = normalize_key(column)
        if key not in by_key:
            raise ValueError(f"Missing target column {column!r}. Available columns: {headers}")
        target_resolved.append(by_key[key])

    if id_column is not None:
        id_key = normalize_key(id_column)
        if id_key not in by_key:
            raise ValueError(f"Missing id column {id_column!r}. Available columns: {headers}")
        return by_key[id_key], target_resolved

    target_keys = {normalize_key(column) for column in target_resolved}
    for candidate in ID_COLUMN_CANDIDATES:
        key = normalize_key(candidate)
        if key in by_key and key not in target_keys:
            # choose the first known id-like column that is not a target value.
            return by_key[key], target_resolved

    for header in headers:
        if normalize_key(header) not in target_keys:
            # fallback for custom csv files: use the first non-target column as id.
            return header, target_resolved
    raise ValueError("Could not infer an id column from label CSV.")


def _read_label_rows(
    labels_csv: Path,
    target_columns: Sequence[str],
    id_column: str | None,
) -> list[tuple[str, np.ndarray]]:
    with labels_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        resolved_id, resolved_targets = _resolve_columns(headers, target_columns, id_column)
        rows = []
        for row in reader:
            raw_id = str(row[resolved_id]).strip()
            if not raw_id:
                continue
            target = np.asarray([_parse_float(row[column]) for column in resolved_targets])
            rows.append((raw_id, target.astype(np.float32)))
    if not rows:
        raise ValueError(f"No usable labeled rows found in {labels_csv}")
    return rows


def resolve_hyperview2_root(dataset_root: str | Path) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    if (root / "train_gt.csv").is_file() and (root / "train").is_dir():
        return root
    nested = root / "HYPERVIEW2"
    if (nested / "train_gt.csv").is_file() and (nested / "train").is_dir():
        return nested
    raise FileNotFoundError(
        "Expected HYPERVIEW2 root with train_gt.csv and train/. "
        f"Checked: {root} and {nested}"
    )


def _canonical_sample_id(raw_id: str) -> str:
    text = str(raw_id).strip()
    if not text:
        raise ValueError("Empty HYPERVIEW2 sample id")
    if text.isdigit():
        return str(int(text))
    return Path(text).stem


def _array_stem_candidates(raw_id: str) -> list[str]:
    sample_id = _canonical_sample_id(raw_id)
    stems = [sample_id]
    if sample_id.isdigit():
        stems.insert(0, f"{int(sample_id):04d}")
    return list(dict.fromkeys(stems))


def _resolve_hyperview2_array_path(array_dir: Path, raw_id: str) -> Path:
    for stem in _array_stem_candidates(raw_id):
        for suffix in (".npz", ".npy", ".tif", ".tiff"):
            path = array_dir / f"{stem}{suffix}"
            if path.is_file():
                return path
    checked = ", ".join(str(array_dir / f"{stem}.npz") for stem in _array_stem_candidates(raw_id))
    raise FileNotFoundError(f"Could not find HYPERVIEW2 array for id {raw_id!r}. Checked: {checked}")


def build_hyperview2_samples(
    dataset_root: str | Path,
    modality: str = "prisma",
    split: str = "train",
    labels_csv: str | Path | None = None,
    id_column: str | None = None,
    target_columns: Sequence[str] = HYPERVIEW2_TARGET_COLUMNS,
    max_samples: int | None = None,
) -> list[Hyperview2Sample]:
    root = resolve_hyperview2_root(dataset_root)
    if modality not in HYPERVIEW2_MODALITY_DIRS:
        raise ValueError(f"modality must be one of: {', '.join(HYPERVIEW2_MODALITY_DIRS)}")
    if split not in {"train", "test"}:
        raise ValueError("split must be one of: train, test")

    label_path = (
        Path(labels_csv).expanduser().resolve()
        if labels_csv
        else root / "train_gt.csv"
    )
    if not label_path.is_file():
        raise FileNotFoundError(f"HYPERVIEW2 labels CSV does not exist: {label_path}")
    array_dir = root / split / HYPERVIEW2_MODALITY_DIRS[modality]
    if not array_dir.is_dir():
        raise FileNotFoundError(f"HYPERVIEW2 array directory does not exist: {array_dir}")

    label_rows = _read_label_rows(label_path, target_columns, id_column)

    samples = []
    missing = []
    for raw_id, target in label_rows:
        key = _canonical_sample_id(raw_id)
        try:
            path = _resolve_hyperview2_array_path(array_dir, raw_id)
        except FileNotFoundError:
            missing.append(raw_id)
            continue
        samples.append(
            Hyperview2Sample(
                sample_id=key,
                array_path=path,
                mask_path=path if path.suffix.lower() == ".npz" else None,
                target=target,
            )
        )
        if max_samples is not None and len(samples) >= max_samples:
            break

    if not samples:
        preview = ", ".join(missing[:8])
        raise ValueError(
            f"No labeled {modality} samples could be paired with {split} image arrays. "
            f"Missing examples: {preview}"
        )
    return sorted(samples, key=lambda sample: sample.sample_id)


def load_array(path: str | Path, modality: str = "any") -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as archive:
            # npz files can contain several arrays; choose the largest image-like one.
            candidates = [archive[key] for key in archive.files if archive[key].ndim >= 3]
            if not candidates:
                raise ValueError(f"No 3D array found in {path}")
            array = max(candidates, key=lambda item: item.size)
    elif suffix in {".tif", ".tiff"}:
        array = tifffile.imread(path)
    else:
        raise ValueError(f"Unsupported array suffix for {path}")
    expected = EXPECTED_BANDS_BY_MODALITY.get(modality)
    return to_chw(np.asarray(array), expected_bands=expected)


def to_chw(array: np.ndarray, expected_bands: int | None = None) -> np.ndarray:
    array = np.asarray(array)
    while array.ndim > 3:
        # remove singleton dimensions that often come from exported remote-sensing files.
        singleton_axes = [axis for axis, dim in enumerate(array.shape) if dim == 1]
        if not singleton_axes:
            break
        array = np.squeeze(array, axis=singleton_axes[0])
    if array.ndim == 2:
        return array[None].astype(np.float32, copy=False)
    if array.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D array, got shape {array.shape}")

    shape = tuple(int(dim) for dim in array.shape)
    if expected_bands is not None:
        # if modality is known, use its band count to identify the spectral axis.
        axes = [axis for axis, dim in enumerate(shape) if dim == expected_bands]
    else:
        known = set(EXPECTED_BANDS_BY_MODALITY.values())
        axes = [axis for axis, dim in enumerate(shape) if dim in known]
    if not axes:
        # fallback heuristic: spectral axis is usually the smaller outer dimension.
        axes = [0] if shape[0] <= shape[-1] else [2]

    band_axis = axes[0]
    if band_axis == 0:
        chw = array
    elif band_axis == 2:
        chw = np.moveaxis(array, 2, 0)
    else:
        chw = np.moveaxis(array, band_axis, 0)
    return np.ascontiguousarray(chw, dtype=np.float32)


def load_mask(path: str | Path | None, shape_hw: tuple[int, int]) -> np.ndarray | None:
    if path is None:
        return None
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        mask = np.asarray(tifffile.imread(path))
    elif path.suffix.lower() == ".npz":
        with np.load(path) as archive:
            if "mask" not in archive.files:
                return None
            mask = np.asarray(archive["mask"])
    else:
        mask = np.asarray(np.load(path))
    mask = np.asarray(mask)
    while mask.ndim > 3:
        singleton_axes = [axis for axis, dim in enumerate(mask.shape) if dim == 1]
        if not singleton_axes:
            break
        mask = np.squeeze(mask, axis=singleton_axes[0])
    if mask.ndim == 3:
        # collapse mask cubes to one spatial mask when needed.
        if tuple(mask.shape[-2:]) == tuple(shape_hw):
            mask = mask.max(axis=0)
        elif tuple(mask.shape[:2]) == tuple(shape_hw):
            mask = mask.max(axis=-1)
        elif mask.shape[0] == 1:
            mask = np.squeeze(mask, axis=0)
        elif mask.shape[-1] == 1:
            mask = np.squeeze(mask, axis=-1)
        elif tuple(np.squeeze(mask).shape) == tuple(shape_hw):
            mask = np.squeeze(mask)
        else:
            mask = mask.max(axis=0) if mask.shape[0] <= mask.shape[-1] else mask.max(axis=-1)
    elif mask.ndim == 1:
        if tuple(shape_hw) == (1, 1):
            mask = np.asarray([[mask.any()]])
        elif mask.size == int(shape_hw[0]) * int(shape_hw[1]):
            mask = mask.reshape(shape_hw)
        else:
            raise ValueError(f"Mask shape {mask.shape} does not match image shape {shape_hw}")
    if tuple(mask.shape) != tuple(shape_hw):
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {shape_hw}")
    return mask > 0


def normalize_cube(
    cube: np.ndarray,
    mask: np.ndarray | None = None,
    mode: str = "percentile",
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> np.ndarray:
    cube = np.asarray(cube, dtype=np.float32)
    finite = np.isfinite(cube)
    if mask is not None:
        # normalization statistics should ignore pixels outside the valid mask.
        finite &= mask[None]
    values = cube[finite]
    if values.size == 0:
        return np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "none":
        return np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "minmax":
        low = float(values.min())
        high = float(values.max())
    elif mode == "percentile":
        # percentile scaling is more robust to outliers than raw minmax.
        low, high = np.percentile(values, [percentile_low, percentile_high]).astype(np.float32)
        low = float(low)
        high = float(high)
    else:
        raise ValueError("normalization must be one of: none, minmax, percentile")
    if not math.isfinite(high - low) or high <= low:
        return np.zeros_like(cube, dtype=np.float32)
    return np.clip((cube - low) / (high - low), 0.0, 1.0).astype(np.float32)


def extract_spectral_stats(
    cube: np.ndarray,
    mask: np.ndarray | None = None,
    normalization: str = "percentile",
) -> np.ndarray:
    return extract_spectral_features(
        cube,
        mask=mask,
        normalization=normalization,
        feature_set="mean_std",
    )


def _valid_flat_pixels(cube: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    c, h, w = cube.shape
    flat = cube.reshape(c, h * w)
    valid = mask.reshape(h * w).astype(bool) if mask is not None else np.isfinite(flat).all(axis=0)
    return flat, valid


def _band_stat(values: np.ndarray, channels: int, statistic: str) -> np.ndarray:
    if values.size == 0:
        return np.zeros(channels, dtype=np.float32)
    if statistic == "mean":
        return values.mean(axis=1).astype(np.float32)
    if statistic == "std":
        return values.std(axis=1).astype(np.float32)
    if statistic == "min":
        return values.min(axis=1).astype(np.float32)
    if statistic == "max":
        return values.max(axis=1).astype(np.float32)
    if statistic == "median":
        return np.median(values, axis=1).astype(np.float32)
    if statistic == "q25":
        return np.percentile(values, 25.0, axis=1).astype(np.float32)
    if statistic == "q75":
        return np.percentile(values, 75.0, axis=1).astype(np.float32)
    raise ValueError(f"Unsupported band statistic: {statistic}")


def _spectral_gradient(values: np.ndarray) -> np.ndarray:
    if values.shape[0] <= 1:
        return np.zeros_like(values, dtype=np.float32)
    return np.gradient(values.astype(np.float32)).astype(np.float32)


def extract_spectral_features(
    cube: np.ndarray,
    mask: np.ndarray | None = None,
    normalization: str = "percentile",
    feature_set: str = "mean_std",
) -> np.ndarray:
    if feature_set not in HYPERVIEW2_FEATURE_SETS:
        raise ValueError(f"feature_set must be one of: {', '.join(HYPERVIEW2_FEATURE_SETS)}")
    cube = normalize_cube(cube, mask=mask, mode=normalization)
    c, _, _ = cube.shape
    flat, valid = _valid_flat_pixels(cube, mask)
    # valid pixels are the only ones used for spectral statistics.
    valid_fraction = float(valid.mean()) if valid.size else 0.0
    valid_values = flat[:, valid] if valid.any() else np.empty((c, 0), dtype=np.float32)
    mean = _band_stat(valid_values, channels=c, statistic="mean")
    std = _band_stat(valid_values, channels=c, statistic="std")
    features = [mean, std]

    if feature_set in {"mean_std_derivatives", "full_stats"}:
        features.extend([_spectral_gradient(mean), _spectral_gradient(std)])

    if feature_set == "full_stats":
        features.extend(
            [
                _band_stat(valid_values, channels=c, statistic="min"),
                _band_stat(valid_values, channels=c, statistic="max"),
                _band_stat(valid_values, channels=c, statistic="median"),
                _band_stat(valid_values, channels=c, statistic="q25"),
                _band_stat(valid_values, channels=c, statistic="q75"),
            ]
        )

    features.append(np.asarray([valid_fraction], dtype=np.float32))
    return np.concatenate(features).astype(np.float32)


class Hyperview2FeatureDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Hyperview2Sample],
        modality: str = "prisma",
        normalization: str = "percentile",
        feature_set: str = "mean_std",
    ):
        if not samples:
            raise ValueError("Empty Hyperview2FeatureDataset")
        if feature_set not in HYPERVIEW2_FEATURE_SETS:
            raise ValueError(f"feature_set must be one of: {', '.join(HYPERVIEW2_FEATURE_SETS)}")
        self.samples = list(samples)
        self.modality = modality
        self.normalization = normalization
        self.feature_set = feature_set

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        cube = load_array(sample.array_path, modality=self.modality)
        mask = load_mask(sample.mask_path, shape_hw=tuple(cube.shape[-2:]))
        # feature baseline uses simple mean and std per band instead of all pixels.
        features = extract_spectral_features(
            cube,
            mask=mask,
            normalization=self.normalization,
            feature_set=self.feature_set,
        )
        return {
            "features": torch.from_numpy(features),
            "target": torch.from_numpy(sample.target.astype(np.float32)),
            "sample_id": sample.sample_id,
            "path": str(sample.array_path),
        }


class Hyperview2PixelSetDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Hyperview2Sample],
        modality: str = "prisma",
        normalization: str = "percentile",
        max_pixels: int | None = None,
    ):
        if not samples:
            raise ValueError("Empty Hyperview2PixelSetDataset")
        if max_pixels is not None and max_pixels <= 0:
            raise ValueError("max_pixels must be positive when provided")
        self.samples = list(samples)
        self.modality = modality
        self.normalization = normalization
        self.max_pixels = max_pixels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        cube = load_array(sample.array_path, modality=self.modality)
        mask = load_mask(sample.mask_path, shape_hw=tuple(cube.shape[-2:]))
        cube = normalize_cube(cube, mask=mask, mode=self.normalization)
        c, h, w = cube.shape
        # every spatial pixel becomes one row with c spectral values.
        pixels = cube.reshape(c, h * w).T.astype(np.float32, copy=False)
        if mask is not None:
            valid = mask.reshape(h * w).astype(bool)
        else:
            valid = np.isfinite(pixels).all(axis=1)
        if self.max_pixels is not None and pixels.shape[0] > self.max_pixels:
            valid_indices = np.flatnonzero(valid)
            if valid_indices.size > self.max_pixels:
                # deterministic subsampling keeps runs reproducible.
                take = np.linspace(0, valid_indices.size - 1, self.max_pixels, dtype=np.int64)
                selected = valid_indices[take]
            else:
                selected = np.linspace(0, pixels.shape[0] - 1, self.max_pixels, dtype=np.int64)
            pixels = pixels[selected]
            valid = valid[selected]
        return {
            "pixels": torch.from_numpy(np.ascontiguousarray(pixels, dtype=np.float32)),
            "valid_mask": torch.from_numpy(np.ascontiguousarray(valid, dtype=bool)),
            "target": torch.from_numpy(sample.target.astype(np.float32)),
            "sample_id": sample.sample_id,
            "path": str(sample.array_path),
        }


class Hyperview2CompressionDataset(Dataset):
    """HYPERVIEW2 cubes for unsupervised compressor training.

    The dataset returns normalized CHW cubes and per-value validity masks. Spatial dimensions
    are intentionally left untouched here; use ``collate_compression_batch`` to pad batches.
    """

    def __init__(
        self,
        samples: Sequence[Hyperview2Sample],
        modality: str = "prisma",
        normalization: str = "percentile",
    ):
        if not samples:
            raise ValueError("Empty Hyperview2CompressionDataset")
        self.samples = list(samples)
        self.modality = modality
        self.normalization = normalization

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        cube = load_array(sample.array_path, modality=self.modality)
        mask = load_mask(sample.mask_path, shape_hw=tuple(cube.shape[-2:]))
        cube = normalize_cube(cube, mask=mask, mode=self.normalization)
        c, h, w = cube.shape
        if mask is None:
            valid_mask = np.isfinite(cube)
        else:
            # compressor loss needs a per-value mask, so broadcast the spatial mask over bands.
            valid_mask = np.broadcast_to(mask[None], (c, h, w)).copy()
        return {
            "x": torch.from_numpy(np.ascontiguousarray(cube, dtype=np.float32)),
            "valid_mask": torch.from_numpy(np.ascontiguousarray(valid_mask, dtype=bool)),
            "sample_id": sample.sample_id,
            "path": str(sample.array_path),
        }


def collate_feature_batch(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "features": torch.stack([item["features"] for item in batch], dim=0),
        "target": torch.stack([item["target"] for item in batch], dim=0),
        "sample_id": [item["sample_id"] for item in batch],
        "path": [item["path"] for item in batch],
    }


def collate_pixel_set_batch(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    channels = {int(item["pixels"].shape[1]) for item in batch}
    if len(channels) != 1:
        raise ValueError(f"All samples in a batch must have the same band count, got {channels}")
    batch_size = len(batch)
    max_pixels = max(int(item["pixels"].shape[0]) for item in batch)
    in_channels = channels.pop()
    pixels = torch.zeros(batch_size, max_pixels, in_channels, dtype=torch.float32)
    valid_mask = torch.zeros(batch_size, max_pixels, dtype=torch.bool)
    for idx, item in enumerate(batch):
        # pad shorter pixel sets so they can share one tensor batch.
        n_pixels = int(item["pixels"].shape[0])
        pixels[idx, :n_pixels] = item["pixels"]
        valid_mask[idx, :n_pixels] = item["valid_mask"]
    return {
        "pixels": pixels,
        "valid_mask": valid_mask,
        "target": torch.stack([item["target"] for item in batch], dim=0),
        "sample_id": [item["sample_id"] for item in batch],
        "path": [item["path"] for item in batch],
    }


def _padded_size(size: int, pad_multiple: int, min_size: int) -> int:
    target = max(int(size), int(min_size))
    if pad_multiple <= 1:
        return target
    remainder = target % pad_multiple
    return target if remainder == 0 else target + pad_multiple - remainder


def collate_compression_batch(
    batch: Sequence[dict[str, Any]],
    pad_multiple: int = 4,
    min_spatial_size: int = 4,
) -> dict[str, Any]:
    channels = {int(item["x"].shape[0]) for item in batch}
    if len(channels) != 1:
        raise ValueError(f"All samples in a batch must have the same band count, got {channels}")
    in_channels = channels.pop()
    max_h = max(int(item["x"].shape[-2]) for item in batch)
    max_w = max(int(item["x"].shape[-1]) for item in batch)
    # pad to model-friendly multiples because encoders downsample spatial dimensions.
    padded_h = _padded_size(max_h, pad_multiple=pad_multiple, min_size=min_spatial_size)
    padded_w = _padded_size(max_w, pad_multiple=pad_multiple, min_size=min_spatial_size)

    xs = torch.zeros(len(batch), in_channels, padded_h, padded_w, dtype=torch.float32)
    masks = torch.zeros(len(batch), in_channels, padded_h, padded_w, dtype=torch.bool)
    original_shapes = []
    for idx, item in enumerate(batch):
        x = item["x"]
        valid_mask = item["valid_mask"].bool()
        h, w = int(x.shape[-2]), int(x.shape[-1])
        pad_h = padded_h - h
        pad_w = padded_w - w
        # replicate padding avoids creating sharp zero borders in the input tensor.
        xs[idx] = F.pad(x.unsqueeze(0), (0, pad_w, 0, pad_h), mode="replicate").squeeze(0)
        masks[idx, :, :h, :w] = valid_mask
        original_shapes.append(tuple(int(dim) for dim in x.shape))
    return {
        "x": xs,
        "valid_mask": masks,
        "sample_id": [item["sample_id"] for item in batch],
        "path": [item["path"] for item in batch],
        "original_shape": original_shapes,
    }


def split_samples(
    samples: Sequence[Hyperview2Sample],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[Hyperview2Sample], list[Hyperview2Sample]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(samples))
    # fixed seed gives reproducible train and validation partitions.
    rng.shuffle(indices)
    val_count = max(1, int(round(len(samples) * val_fraction)))
    val_indices = set(indices[:val_count].tolist())
    train = [sample for idx, sample in enumerate(samples) if idx not in val_indices]
    val = [sample for idx, sample in enumerate(samples) if idx in val_indices]
    if not train or not val:
        raise ValueError("Split produced an empty train or validation set")
    return train, val


def hyperview_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_mse: np.ndarray,
    eps: float = 1e-12,
) -> tuple[float, np.ndarray]:
    mse = ((y_pred - y_true) ** 2).mean(axis=0)
    # official-style score normalizes each target mse by a baseline mse.
    per_target = mse / np.maximum(baseline_mse, eps)
    return float(per_target.mean()), per_target.astype(np.float32)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_mse: np.ndarray,
    target_columns: Sequence[str] = HYPERVIEW2_TARGET_COLUMNS,
) -> dict[str, Any]:
    mse = ((y_pred - y_true) ** 2).mean(axis=0)
    mae = np.abs(y_pred - y_true).mean(axis=0)
    score, per_target_score = hyperview_score(y_true, y_pred, baseline_mse)
    return {
        "hyperview_score": score,
        "mean_mse": float(mse.mean()),
        "mean_mae": float(mae.mean()),
        "targets": {
            target: {
                "mse": float(mse[idx]),
                "mae": float(mae[idx]),
                "rmse": float(np.sqrt(mse[idx])),
                "relative_mse": float(per_target_score[idx]),
                "baseline_mse": float(baseline_mse[idx]),
            }
            for idx, target in enumerate(target_columns)
        },
    }


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
