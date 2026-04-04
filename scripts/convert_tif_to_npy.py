"""
Converts raw HySpecNet-11k patches from .TIF format to .npy.

Preprocessing (compatible with Fuchs & Demir 2023):
  1. Replace nodata (-32768) with zeros
  2. Remove water bands: [126-140] and [160-166] (0-indexed) → 202 bands
  3. Clip to [0, 10000] (physical reflectance range EnMAP L2A)
  4. Min-max normalization → [0.0, 1.0]  (global: min=0, max=10000)
  5. Save as float32 in (C, H, W) format = (202, 128, 128)

Output format (C, H, W):
  Directly compatible with PyTorch Conv2d — no transpose in DataLoader.
  datasets.py automatically detects (202, 128, 128) shape as CHW.

Disk requirements:
  11,483 patches × 202 × 128 × 128 × 4B ≈ 144 GB
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tifffile as tiff
from tqdm import tqdm

WATER_VAPOR_BANDS = list(range(126, 141)) + list(range(160, 167))  # 22 bands
NODATA_VALUE = -32768
CLIP_MIN = 0.0
CLIP_MAX = 10000.0
EXPECTED_BANDS = 202
PATCH_SIZE = 128
EXPECTED_SHAPE_CHW = (EXPECTED_BANDS, PATCH_SIZE, PATCH_SIZE)


def _npy_path(tif_path: Path) -> Path:
    stem = tif_path.stem.replace("-SPECTRAL_IMAGE", "")
    return tif_path.parent / f"{stem}-DATA.npy"


def _preprocess_tif(tif_path: Path) -> np.ndarray:
    raw = tiff.imread(str(tif_path))  # (224, H, W) int16

    if raw.ndim != 3:
        raise ValueError(f"unexpected ndim={raw.ndim}, shape={raw.shape}")

    x = raw.astype(np.float32)
    x[x == NODATA_VALUE] = 0.0

    keep = [i for i in range(x.shape[0]) if i not in WATER_VAPOR_BANDS]
    x = x[keep]  # (202, H, W)

    if x.shape[0] != EXPECTED_BANDS:
        raise ValueError(f"unexpected band count after drop: {x.shape[0]}")

    x = np.clip(x, CLIP_MIN, CLIP_MAX)
    x = (x - CLIP_MIN) / (CLIP_MAX - CLIP_MIN)

    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.shape != EXPECTED_SHAPE_CHW:
        raise ValueError(f"shape mismatch: {x.shape}")

    return x


def convert_one(args: tuple[Path, bool]) -> tuple[Path, str, str]:
    """
    Converts one TIF → NPY file.
    Returns (npy_path, status, error_msg).
    Status: "converted" | "skipped" | "error"
    """
    tif_path, force = args
    npy_path = _npy_path(tif_path)

    try:
        x = _preprocess_tif(tif_path)

        if not force and npy_path.exists():
            try:
                existing = np.load(str(npy_path), mmap_mode="r")
                if (
                    existing.shape == EXPECTED_SHAPE_CHW
                    and existing.dtype == np.float32
                    and np.array_equal(existing, x)
                ):
                    return npy_path, "skipped", ""
            except Exception:
                pass

        np.save(str(npy_path), x)
        return npy_path, "converted", ""

    except Exception as e:
        return npy_path, "error", str(e)


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "dataset_root", type=str, help="Main dataset directory (contains patches/ subdirectory)"
    )
    p.add_argument("--workers", type=int, default=8, help="Number of parallel workers (default: 8)")
    p.add_argument("--dry-run", action="store_true", help="Only count files, do not convert")
    p.add_argument(
        "--force", action="store_true", help="Overwrite existing .npy files (even correct ones)"
    )
    p.add_argument(
        "--verify", action="store_true", help="After conversion, verify random sample of files"
    )
    return p.parse_args()


def verify_sample(npy_files: list[Path], n: int = 20) -> None:
    """Verifies random sample of converted files."""
    import random

    sample = random.sample(npy_files, min(n, len(npy_files)))
    errors = []
    for f in sample:
        try:
            arr = np.load(str(f), mmap_mode="r")
            assert arr.shape == EXPECTED_SHAPE_CHW, f"shape {arr.shape}"
            assert arr.dtype == np.float32, f"dtype {arr.dtype}"
            assert arr.min() >= 0.0, f"min={arr.min():.4f} < 0"
            assert arr.max() <= 1.0, f"max={arr.max():.4f} > 1"
            assert np.isfinite(arr).all(), "contains NaN/Inf"
        except Exception as e:
            errors.append((f.name, str(e)))

    print(f"\nVerification of {len(sample)} random files:")
    if errors:
        print(f"  ERRORS: {len(errors)}")
        for name, msg in errors:
            print(f"    {name}: {msg}")
    else:
        print(f"  All OK — shape {EXPECTED_SHAPE_CHW}, float32, range [0,1]")


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)

    if not dataset_root.exists():
        print(f"Error: {dataset_root} does not exist")
        sys.exit(1)

    patches_dir = dataset_root / "patches"
    if not patches_dir.exists():
        print(f"Error: directory {patches_dir} does not exist")
        sys.exit(1)

    # Find all TIF files
    tif_files = sorted(patches_dir.rglob("*-SPECTRAL_IMAGE.TIF"))
    if not tif_files:
        tif_files = sorted(patches_dir.rglob("*-SPECTRAL_IMAGE.tif"))
    if not tif_files:
        print(f"Error: no *-SPECTRAL_IMAGE.TIF files in {patches_dir}")
        sys.exit(1)

    existing_npy = [f for f in tif_files if _npy_path(f).exists()]
    correct_shape = 0
    wrong_shape = 0
    for f in existing_npy:
        try:
            arr = np.load(str(_npy_path(f)), mmap_mode="r")
            if arr.shape == EXPECTED_SHAPE_CHW and arr.dtype == np.float32:
                correct_shape += 1
            else:
                wrong_shape += 1
        except Exception:
            wrong_shape += 1

    to_convert = len(tif_files)
    size_per_file_mb = EXPECTED_BANDS * PATCH_SIZE * PATCH_SIZE * 4 / 1024 / 1024
    total_gb = len(tif_files) * size_per_file_mb / 1024

    print("=" * 55)
    print("  TIF → NPY CONVERSION  |  HySpecNet-11k")
    print("=" * 55)
    print(f"  Directory:           {patches_dir}")
    print(f"  TIF files:           {len(tif_files):,}")
    print(f"  Existing candidates: {correct_shape:,}  (shape {EXPECTED_SHAPE_CHW}, dtype float32)")
    if wrong_shape:
        print(f"  Wrong shape/error:   {wrong_shape:,}  (will be recalculated)")
    print(f"  To verify/convert:   {to_convert:,}")
    print(f"  Required space:      ~{total_gb:.1f} GB")
    print(f"  Output format:       {EXPECTED_SHAPE_CHW} float32 [0.0, 1.0]")
    print(f"  Workers:             {args.workers}")
    print()

    if args.dry_run:
        print(
            "Dry-run mode — no changes. Existing DATA.npy files are not trusted without TIF parity check."
        )
        return

    job_args = [(f, args.force) for f in tif_files]
    counts = {"converted": 0, "skipped": 0, "error": 0}
    error_list = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(convert_one, a): a[0] for a in job_args}
        with tqdm(total=len(tif_files), unit="file", desc="Conversion") as bar:
            for future in as_completed(futures):
                npy_path, status, msg = future.result()
                counts[status] += 1
                if status == "error":
                    error_list.append((npy_path.name, msg))
                bar.update(1)
                bar.set_postfix(
                    conv=counts["converted"], skip=counts["skipped"], err=counts["error"]
                )

    print()
    print("Done:")
    print(f"  Converted:   {counts['converted']:,}")
    print(f"  Skipped:     {counts['skipped']:,}  (already OK)")
    print(f"  Errors:      {counts['error']:,}")

    if error_list:
        print("\nFirst errors:")
        for name, msg in error_list[:10]:
            print(f"  {name}: {msg}")

    if args.verify or counts["error"] == 0:
        all_npy = [_npy_path(f) for f in tif_files if _npy_path(f).exists()]
        verify_sample(all_npy)


if __name__ == "__main__":
    main()
