import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tifffile as tiff
from tqdm import tqdm

WATER_VAPOR_BANDS = list(range(126, 141)) + list(range(160, 167))
NODATA_VALUE = -32768
CLIP_MIN = 0.0
CLIP_MAX = 10000.0


def convert_one(tif_path: Path) -> tuple[Path, bool, str]:
    stem = tif_path.stem.replace("-SPECTRAL_IMAGE", "")
    npy_path = tif_path.parent / f"{stem}-DATA.npy"

    if npy_path.exists():
        return npy_path, True, "exists"

    try:
        x = tiff.imread(str(tif_path)).astype(np.float32)  # (C, H, W)

        x[x == NODATA_VALUE] = 0.0

        keep = [i for i in range(x.shape[0]) if i not in WATER_VAPOR_BANDS]
        x = x[keep]   # (202, H, W)

        x = np.clip(x, CLIP_MIN, CLIP_MAX)
        x = (x - CLIP_MIN) / (CLIP_MAX - CLIP_MIN)

        x = x.transpose(1, 2, 0).astype(np.float32)  # (H, W, C)
        np.save(str(npy_path), x)

        return npy_path, True, ""
    except Exception as e:
        return npy_path, False, str(e)


def parse_args():
    p = argparse.ArgumentParser(description="Convert TIF spectral images to normalized NumPy arrays")
    p.add_argument("dataset_root", type=str)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dry-run", action="store_true", help="Only count files, don't convert")
    return p.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)

    if not dataset_root.exists():
        print(f"Error: {dataset_root} does not exist")
        sys.exit(1)

    tif_files = sorted((dataset_root / "patches").rglob("*-SPECTRAL_IMAGE.TIF"))
    if not tif_files:
        tif_files = sorted((dataset_root / "patches").rglob("*-SPECTRAL_IMAGE.tif"))

    if not tif_files:
        print(f"Error: No *-SPECTRAL_IMAGE.TIF files found in {dataset_root}/patches/")
        sys.exit(1)

    already_done = sum(
        1 for f in tif_files
        if (f.parent / f.stem.replace("-SPECTRAL_IMAGE", "-DATA.npy")
            .replace(".npy", "")).with_suffix(".npy").exists()
    )

    sample_size_mb = 128 * 128 * 202 * 4 / 1024 / 1024
    total_gb = len(tif_files) * sample_size_mb / 1024

    print(f"Found TIF files:     {len(tif_files):,}")
    print(f"Already converted .npy:  {already_done:,}")
    print(f"Pending conversion:              {len(tif_files) - already_done:,}")
    print(f"Required disk space: ~{total_gb:.1f} GB")
    print()

    if args.dry_run:
        return

    to_convert = [
        f for f in tif_files
        if not (f.parent / f"{f.stem.replace('-SPECTRAL_IMAGE', '')}-DATA.npy").exists()
    ]

    if not to_convert:
        print("All files are already converted.")
        return

    print(f"Converting {len(to_convert):,} files ({args.workers} workers)...")

    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(convert_one, f): f for f in to_convert}
        with tqdm(total=len(to_convert), unit="file") as pbar:
            for future in as_completed(futures):
                npy_path, success, msg = future.result()
                if not success:
                    errors.append((npy_path, msg))
                pbar.update(1)

    print(f"\nDone. Errors: {len(errors)}")
    if errors:
        for path, msg in errors:
            print(f"  {path}: {msg}")


if __name__ == "__main__":
    main()
