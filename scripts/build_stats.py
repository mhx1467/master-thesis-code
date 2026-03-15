from pathlib import Path
import sys
import torch

from hsi_compression.data.datamodule import build_dataset
from hsi_compression.stats import compute_band_stats
from hsi_compression.paths import ensure_artifact_dirs, default_stats_path
from hsi_compression.utils import load_project_env


def main():
    load_project_env()

    if len(sys.argv) < 2:
        print("Usage: python build_stats.py <dataset_root_path>")
        sys.exit(1)

    dataset_root = Path(sys.argv[1])
    if not dataset_root.exists():
        print(f"Error: Provided dataset root path does not exist: {dataset_root}")
        print("Example usage: python build_stats.py /path/to/dataset")
        sys.exit(1)

    difficulty = "easy"

    ensure_artifact_dirs()

    ds_raw = build_dataset(
        dataset_root=dataset_root,
        split_name="train",
        difficulty=difficulty,
        normalized=False,
        return_mask=True,
        drop_invalid_channels=False,
    )

    mean, std, count = compute_band_stats(ds_raw)

    print("mean shape:", mean.shape)
    print("std shape:", std.shape)
    print("first 10 mean:", mean[:10])
    print("first 10 std:", std[:10])
    print("bands with zero valid count:", int((count == 0).sum().item()))

    out_path = default_stats_path(difficulty)
    torch.save(
        {
            "mean": mean,
            "std": std,
            "count": count,
            "valid_band_mask": count > 0,
            "split": "train",
            "difficulty": difficulty,
            "normalization": "per_band_standardize_valid_only",
            "nodata_value": -32768,
        },
        out_path,
    )
    print(f"Saved stats to: {out_path}")


if __name__ == "__main__":
    main()