from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hsi_compression.constants import (
    DEFAULT_DIFFICULTY,
    DEFAULT_NUM_WORKERS,
    NODATA_VALUE,
    WATER_VAPOR_BANDS,
)
from hsi_compression.datasets import HSITiffDataset
from hsi_compression.paths import default_stats_path
from hsi_compression.splits import resolve_split_paths, split_csv_path
from hsi_compression.transforms import GlobalMinMaxNormalize


def build_dataset(
    dataset_root: str | Path,
    split_name: str,
    difficulty: str = DEFAULT_DIFFICULTY,
    normalized: bool = True,
    stats_path: str | Path | None = None,
    return_mask: bool = True,
    drop_invalid_channels: bool = True,
    prefer_npy: bool = True,
):
    dataset_root = Path(dataset_root)
    csv_path = split_csv_path(dataset_root, split_name, difficulty)
    paths = resolve_split_paths(dataset_root, csv_path)

    npy_available = False
    if prefer_npy and paths:
        stem = paths[0].stem.replace("-SPECTRAL_IMAGE", "")
        npy_available = (paths[0].parent / f"{stem}-DATA.npy").exists()

    transform = None
    if normalized and not npy_available:
        stats_path = (
            Path(stats_path) if stats_path is not None
            else default_stats_path(difficulty)
        )
        if not stats_path.exists():
            raise FileNotFoundError(
                f"No stats found: {stats_path}\n"
                f"Run: python scripts/build_stats.py {dataset_root}"
            )
        stats = torch.load(stats_path, map_location="cpu", weights_only=True)
        if "global_min" not in stats or "global_max" not in stats:
            raise KeyError(
                "Stats file missing 'global_min'/'global_max'.\n"
                "Regenerate: python scripts/build_stats.py <dataset_root>"
            )
        transform = GlobalMinMaxNormalize(
            global_min=float(stats["global_min"]),
            global_max=float(stats["global_max"]),
        )

    return HSITiffDataset(
        paths=paths,
        nodata_value=NODATA_VALUE,
        transform=transform,
        return_mask=return_mask,
        invalid_channels=WATER_VAPOR_BANDS,
        drop_invalid_channels=drop_invalid_channels,
        prefer_npy=prefer_npy,
    )


class _FastCollator:
    def __init__(self, batch_size: int, num_bands: int = 202,
                 patch_size: int = 128, use_mask: bool = True):
        self.batch_size = batch_size
        self.use_mask   = use_mask
        self._x_buf    = torch.empty(batch_size, num_bands, patch_size, patch_size)
        self._mask_buf = torch.ones(batch_size, num_bands, patch_size, patch_size,
                                    dtype=torch.bool)

    def __call__(self, batch: list[dict]) -> dict:
        n = len(batch)
        x_out = self._x_buf[:n]
        for i, item in enumerate(batch):
            x_out[i].copy_(item["x"])

        result = {
            "x":          x_out,
            "patch_id":   [item["patch_id"] for item in batch],
        }

        if self.use_mask:
            result["valid_mask"] = self._mask_buf[:n]

        return result


def build_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = DEFAULT_NUM_WORKERS,
    sampler=None,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> DataLoader:
    use_persistent = persistent_workers and num_workers > 0
    prefetch = 2 if num_workers > 0 else None

    using_npy = getattr(dataset, '_use_npy', False)
    if not using_npy and hasattr(dataset, 'dataset'):
        using_npy = getattr(dataset.dataset, '_use_npy', False)

    try:
        sample = dataset[0] if not hasattr(dataset, 'dataset') else dataset.dataset[0]
        num_bands  = sample["x"].shape[0]
        patch_size = sample["x"].shape[1]
        has_mask   = "valid_mask" in sample
    except Exception:
        num_bands  = 202
        patch_size = 128
        has_mask   = True

    collate_fn = _FastCollator(
        batch_size=batch_size,
        num_bands=num_bands,
        patch_size=patch_size,
        use_mask=has_mask,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=use_persistent,
        prefetch_factor=prefetch,
        drop_last=False,
        collate_fn=collate_fn,
    )
