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
EXPECTED_BANDS_BY_MODALITY = {
    "airborne": 430,
    "prisma": 230,
    "sentinel2": 13,
}
ARRAY_SUFFIXES = {".npy", ".npz", ".tif", ".tiff"}
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
        mean = values.mean(axis=0).astype(np.float32)
        std = values.std(axis=0).astype(np.float32)
        std = np.maximum(std, eps).astype(np.float32)
        return cls(mean=mean, std=std, eps=eps)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
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
            valid_mask = valid_mask.clone()
            valid_mask[empty_rows, 0] = True

        encoded = self.pixel_encoder(pixels)
        mask_f = valid_mask.unsqueeze(-1).to(encoded.dtype)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        mean_pool = (encoded * mask_f).sum(dim=1) / denom

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


def infer_split(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    stem = path.stem.lower()
    tokens = parts | {name, stem}
    if tokens & {"train", "training", "t"} or stem.startswith("train"):
        return "train"
    if tokens & {"val", "valid", "validation", "v"} or stem.startswith(("val", "valid")):
        return "validation"
    if tokens & {"test", "testing", "psi", "ψ"} or stem.startswith("test"):
        return "test"
    return "unknown"


def infer_modality(path: Path) -> str:
    text = path.as_posix().lower()
    name = path.name.lower()
    parts = {part.lower() for part in path.parts}
    if "mask" in name:
        return "mask"
    if "prisma" in text or "hsi_satellite" in parts:
        return "prisma"
    if "sentinel" in text or "sentinel-2" in text or "s2" in parts or "msi_satellite" in parts:
        return "sentinel2"
    if "airborne" in text or "hsi_airborne" in parts or "hyspex" in text or "vs-725" in text:
        return "airborne"
    return "unknown"


def find_label_csv(dataset_root: Path, target_columns: Sequence[str]) -> Path:
    target_keys = {normalize_key(column) for column in target_columns}
    candidates: list[tuple[int, Path]] = []
    for path in dataset_root.rglob("*.csv"):
        try:
            with path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                headers = {normalize_key(header) for header in (reader.fieldnames or [])}
        except UnicodeDecodeError:
            continue
        overlap = len(target_keys & headers)
        if overlap == len(target_keys):
            candidates.append((0 if "train" in path.as_posix().lower() else 1, path))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a CSV containing all target columns: {tuple(target_columns)}"
        )
    return sorted(candidates)[0][1]


def _resolve_columns(
    headers: Sequence[str],
    target_columns: Sequence[str],
    id_column: str | None,
) -> tuple[str, list[str]]:
    by_key = {normalize_key(header): header for header in headers}
    target_resolved = []
    for column in target_columns:
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
            return by_key[key], target_resolved

    for header in headers:
        if normalize_key(header) not in target_keys:
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


def _clean_identifier(value: str) -> str:
    text = Path(value).stem if any(sep in value for sep in ("/", "\\")) else value
    text = normalize_key(text)
    for token in (
        "spectralimage",
        "spectral",
        "image",
        "prisma",
        "sentinel2",
        "sentinel",
        "airborne",
        "hyspex",
        "hsi",
        "msi",
        "satellite",
        "mask",
        "data",
        "gt",
        "label",
        "labels",
    ):
        text = text.replace(token, "")
    return text


def _canonical_identifier(value: str) -> str:
    cleaned = _clean_identifier(value)
    if cleaned.isdigit():
        return str(int(cleaned))
    return cleaned


def _identifier_aliases(value: str) -> set[str]:
    cleaned = _clean_identifier(value)
    aliases = {cleaned, _canonical_identifier(value)}
    return {alias for alias in aliases if alias}


def _path_id_candidates(path: Path) -> set[str]:
    candidates = set()
    for value in (path.stem, path.parent.name, f"{path.parent.name}_{path.stem}"):
        candidates.update(_identifier_aliases(value))
    return {candidate for candidate in candidates if candidate}


def _index_paths(dataset_root: Path, modality: str, split: str | None = None) -> dict[str, Path]:
    buckets: dict[str, list[Path]] = {}
    for path in dataset_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in ARRAY_SUFFIXES:
            continue
        if split is not None and infer_split(path) != split:
            continue
        inferred = infer_modality(path)
        if modality != "any" and inferred != modality:
            continue
        for candidate in _path_id_candidates(path):
            buckets.setdefault(candidate, []).append(path)

    index = {}
    for key, paths in buckets.items():
        unique = sorted(set(paths), key=lambda item: (len(item.as_posix()), item.as_posix()))
        if unique:
            index[key] = unique[0]
    return index


def build_hyperview2_samples(
    dataset_root: str | Path,
    modality: str = "prisma",
    labels_csv: str | Path | None = None,
    id_column: str | None = None,
    target_columns: Sequence[str] = HYPERVIEW2_TARGET_COLUMNS,
    max_samples: int | None = None,
) -> list[Hyperview2Sample]:
    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"HYPERVIEW2 dataset root does not exist: {root}")
    if modality not in {"prisma", "sentinel2", "airborne", "any"}:
        raise ValueError("modality must be one of: prisma, sentinel2, airborne, any")

    label_path = (
        Path(labels_csv).expanduser().resolve()
        if labels_csv
        else find_label_csv(root, target_columns)
    )
    label_rows = _read_label_rows(label_path, target_columns, id_column)
    label_split = infer_split(label_path)
    array_split = None if label_split == "unknown" else label_split
    array_index = _index_paths(root, modality, split=array_split)
    mask_index = _index_paths(root, "mask", split=array_split)

    samples = []
    missing = []
    for raw_id, target in label_rows:
        key = _canonical_identifier(raw_id)
        path = next(
            (array_index[alias] for alias in _identifier_aliases(raw_id) if alias in array_index),
            None,
        )
        if path is None:
            missing.append(raw_id)
            continue
        samples.append(
            Hyperview2Sample(
                sample_id=key,
                array_path=path,
                mask_path=mask_index.get(key),
                target=target,
            )
        )
        if max_samples is not None and len(samples) >= max_samples:
            break

    if not samples:
        preview = ", ".join(missing[:8])
        raise ValueError(
            f"No labeled {modality} samples could be paired with image arrays. "
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
        axes = [axis for axis, dim in enumerate(shape) if dim == expected_bands]
    else:
        known = set(EXPECTED_BANDS_BY_MODALITY.values())
        axes = [axis for axis, dim in enumerate(shape) if dim in known]
    if not axes:
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
    mask = np.asarray(
        tifffile.imread(path) if Path(path).suffix.lower() in {".tif", ".tiff"} else np.load(path)
    )
    mask = np.squeeze(mask)
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[-1] == 1:
            mask = mask[..., 0]
        else:
            mask = mask.max(axis=0) if mask.shape[0] <= mask.shape[-1] else mask.max(axis=-1)
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
    cube = normalize_cube(cube, mask=mask, mode=normalization)
    c, h, w = cube.shape
    flat = cube.reshape(c, h * w)
    valid = mask.reshape(h * w).astype(bool) if mask is not None else np.isfinite(flat).all(axis=0)
    valid_fraction = float(valid.mean()) if valid.size else 0.0
    if not valid.any():
        mean = np.zeros(c, dtype=np.float32)
        std = np.zeros(c, dtype=np.float32)
    else:
        valid_values = flat[:, valid]
        mean = valid_values.mean(axis=1).astype(np.float32)
        std = valid_values.std(axis=1).astype(np.float32)
    return np.concatenate([mean, std, np.asarray([valid_fraction], dtype=np.float32)]).astype(
        np.float32
    )


class Hyperview2FeatureDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Hyperview2Sample],
        modality: str = "prisma",
        normalization: str = "percentile",
    ):
        if not samples:
            raise ValueError("Empty Hyperview2FeatureDataset")
        self.samples = list(samples)
        self.modality = modality
        self.normalization = normalization

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        cube = load_array(sample.array_path, modality=self.modality)
        mask = load_mask(sample.mask_path, shape_hw=tuple(cube.shape[-2:]))
        features = extract_spectral_stats(cube, mask=mask, normalization=self.normalization)
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
        pixels = cube.reshape(c, h * w).T.astype(np.float32, copy=False)
        if mask is not None:
            valid = mask.reshape(h * w).astype(bool)
        else:
            valid = np.isfinite(pixels).all(axis=1)
        if self.max_pixels is not None and pixels.shape[0] > self.max_pixels:
            valid_indices = np.flatnonzero(valid)
            if valid_indices.size > self.max_pixels:
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
