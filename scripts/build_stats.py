from pathlib import Path
import sys
import argparse
import torch

from hsi_compression.data.datamodule import build_dataset
from hsi_compression.stats import compute_global_minmax
from hsi_compression.paths import ensure_artifact_dirs, default_stats_path
from hsi_compression.utils import load_project_env


def parse_args():
    parser = argparse.ArgumentParser(
        description="Global min/max normalization"
    )
    parser.add_argument("dataset_root", type=str, help="Path to the dataset directory")
    parser.add_argument(
        "--difficulty", type=str, default="easy", choices=["easy", "hard"],
        help="Split type"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit to N samples to testing"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Workers numbers"
    )
    return parser.parse_args()


def main():
    load_project_env()
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"Error: path does not exist: {dataset_root}")
        sys.exit(1)

    ensure_artifact_dirs()

    print(f"Loading training set [{args.difficulty}]...")

    ds_raw = build_dataset(
        dataset_root=dataset_root,
        split_name="train",
        difficulty=args.difficulty,
        normalized=False,
        return_mask=True,
        drop_invalid_channels=True,
    )

    print(f"Number of training patches: {len(ds_raw)}")
    if args.max_samples:
        print(f"Using {args.max_samples} samples (test mode)")

    stats = compute_global_minmax(
        dataset=ds_raw,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
    )

    print(f"\n{'='*50}")
    print(f"global_min:        {stats['global_min']:.4f}")
    print(f"global_max:        {stats['global_max']:.4f}")
    print(f"num_valid_pixels:  {stats['num_valid_pixels']:,}")
    print(f"{'='*50}\n")

    out_path = default_stats_path(args.difficulty)
    torch.save(
        {
            "global_min":       stats["global_min"],
            "global_max":       stats["global_max"],
            "num_valid_pixels": stats["num_valid_pixels"],
            "split":            "train",
            "difficulty":       args.difficulty,
            "normalization":    "global_minmax",
            "num_bands":        202,
            "nodata_value":     -32768,
        },
        out_path,
    )
    print(f"Saved statistics: {out_path}")


if __name__ == "__main__":
    main()
