import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from hsi_compression.constants import (
    DEFAULT_DIFFICULTY,
    DEFAULT_NUM_WORKERS,
    NODATA_VALUE,
    WATER_VAPOR_BANDS,
)
from hsi_compression.datasets import HSITiffDataset
from hsi_compression.splits import load_split_csv, resolve_split_paths, split_csv_path


def _split_entry_to_tif_path(dataset_root: Path, rel_entry: str) -> Path:
    rel = Path(rel_entry)
    patch_id = rel.stem.removesuffix("-DATA")
    return dataset_root / "patches" / rel.parent / f"{patch_id}-SPECTRAL_IMAGE.TIF"


def _resolve_tif_split_paths(dataset_root: Path, csv_path: Path) -> list[Path]:
    paths = [_split_entry_to_tif_path(dataset_root, entry) for entry in load_split_csv(csv_path)]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} resolved TIF patch files do not exist. First missing: {missing[0]}"
        )
    return paths


def build_dataset(
    dataset_root: str | Path,
    split_name: str,
    difficulty: str = DEFAULT_DIFFICULTY,
    normalized: bool = True,
    stats_path: str | Path | None = None,
    return_mask: bool = True,
    drop_invalid_channels: bool = True,
    prefer_npy: bool = True,
    npy_mmap: bool = False,
):
    _ = normalized, stats_path
    dataset_root = Path(dataset_root)
    csv_path = split_csv_path(dataset_root, split_name, difficulty)
    paths = (
        resolve_split_paths(dataset_root, csv_path)
        if prefer_npy
        else _resolve_tif_split_paths(dataset_root, csv_path)
    )

    return HSITiffDataset(
        paths=paths,
        nodata_value=NODATA_VALUE,
        transform=None,
        return_mask=return_mask,
        invalid_channels=WATER_VAPOR_BANDS,
        drop_invalid_channels=drop_invalid_channels,
        prefer_npy=prefer_npy,
        npy_mmap=npy_mmap,
    )


def build_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = DEFAULT_NUM_WORKERS,
    sampler=None,
    pin_memory: bool = True,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = 2,
    seed: int | None = None,
) -> DataLoader:
    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

        def _seed_worker(worker_id: int) -> None:
            # every worker gets a deterministic but different seed.
            worker_seed = (seed + worker_id) % (2**32)
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        worker_init_fn = _seed_worker

    kwargs = {
        "batch_size": batch_size,
        "shuffle": (shuffle if sampler is None else False),
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "generator": generator,
        "worker_init_fn": worker_init_fn,
    }

    if num_workers > 0:
        # persistent workers reduce dataloader startup overhead on large hsi datasets
        kwargs["persistent_workers"] = (
            persistent_workers if persistent_workers is not None else True
        )
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(dataset, **kwargs)
