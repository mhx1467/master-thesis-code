from pathlib import Path
import torch
from torch.utils.data import DataLoader

from hsi_compression.constants import INVALID_CHANNELS, DEFAULT_DIFFICULTY
from hsi_compression.splits import resolve_split_paths
from hsi_compression.datasets import HSITiffDataset
from hsi_compression.transforms import BandStandardize
from hsi_compression.paths import default_stats_path


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
    split_csv = dataset_root / "splits" / difficulty / f"{split_name}.csv"
    paths = resolve_split_paths(dataset_root, split_csv)

    transform = None
    if normalized:
        stats_path = Path(stats_path) if stats_path is not None else default_stats_path(difficulty)
        stats = torch.load(stats_path, map_location="cpu")
        transform = BandStandardize(stats["mean"], stats["std"])

    ds = HSITiffDataset(
        paths=paths,
        transform=transform,
        return_mask=return_mask,
        invalid_channels=INVALID_CHANNELS,
        drop_invalid_channels=drop_invalid_channels,
    )
    return ds


def build_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )