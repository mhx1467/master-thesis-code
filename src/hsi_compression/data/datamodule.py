from pathlib import Path
import torch
from torch.utils.data import DataLoader

from hsi_compression.constants import (
    DEFAULT_DIFFICULTY,
    DEFAULT_NUM_WORKERS,
    INVALID_CHANNELS,
    NODATA_VALUE,
)
from hsi_compression.datasets import HSITiffDataset
from hsi_compression.paths import default_stats_path
from hsi_compression.splits import resolve_split_paths, split_csv_path
from hsi_compression.transforms import BandStandardize


def build_dataset(
    dataset_root: str | Path,
    split_name: str,
    difficulty: str = DEFAULT_DIFFICULTY,
    normalized: bool = False,
    stats_path: str | Path | None = None,
    return_mask: bool = True,
    drop_invalid_channels: bool = False,
):
    dataset_root = Path(dataset_root)
    csv_path = split_csv_path(dataset_root, split_name, difficulty)
    paths = resolve_split_paths(dataset_root, csv_path)

    transform = None
    if normalized:
        stats_path = Path(stats_path) if stats_path is not None else default_stats_path(difficulty)
        stats = torch.load(stats_path, map_location="cpu")
        transform = BandStandardize(stats["mean"], stats["std"])

    return HSITiffDataset(
        paths=paths,
        nodata_value=NODATA_VALUE,
        transform=transform,
        return_mask=return_mask,
        invalid_channels=INVALID_CHANNELS,
        drop_invalid_channels=drop_invalid_channels,
    )


def build_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = DEFAULT_NUM_WORKERS,
    sampler=None,
    pin_memory: bool = False,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )