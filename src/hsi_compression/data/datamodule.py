from pathlib import Path
import torch
from torch.utils.data import DataLoader

from hsi_compression.constants import (
    DEFAULT_DIFFICULTY,
    DEFAULT_NUM_WORKERS,
    WATER_VAPOR_BANDS,
    NODATA_VALUE,
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
):
    dataset_root = Path(dataset_root)
    csv_path = split_csv_path(dataset_root, split_name, difficulty)
    paths = resolve_split_paths(dataset_root, csv_path)

    transform = None
    if normalized:
        stats_path = (
            Path(stats_path)
            if stats_path is not None
            else default_stats_path(difficulty)
        )
        if not stats_path.exists():
            raise FileNotFoundError(
                f"No stats file found: {stats_path}\n"
                f"Run: python scripts/build_stats.py <dataset_root>"
            )
        stats = torch.load(stats_path, map_location="cpu", weights_only=True)

        if "global_min" in stats and "global_max" in stats:
            transform = GlobalMinMaxNormalize(
                global_min=float(stats["global_min"]),
                global_max=float(stats["global_max"]),
            )
        else:
            raise KeyError(
                f"Stats file {stats_path} does not contain 'global_min'/'global_max'.\n"
                f"Generate a new file: python scripts/build_stats.py <dataset_root>"
            )

    invalid_channels = WATER_VAPOR_BANDS if not drop_invalid_channels else None

    return HSITiffDataset(
        paths=paths,
        nodata_value=NODATA_VALUE,
        transform=transform,
        return_mask=return_mask,
        invalid_channels=invalid_channels,
        drop_invalid_channels=drop_invalid_channels,
    )


def build_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = DEFAULT_NUM_WORKERS,
    sampler=None,
    pin_memory: bool = True,
    persistent_workers: bool = True,
):
    use_persistent = persistent_workers and num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        drop_last=False,
    )
