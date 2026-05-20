#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from hsi_compression.paths import logs_dir
from hsi_compression.utils import load_project_env

EXPECTED_BANDS = {
    "airborne": 430,
    "prisma": 230,
    "sentinel2": 13,
    "hyspecnet_mamba_checkpoint": 202,
}
SOIL_TARGETS = {"b", "cu", "zn", "fe", "s", "mn"}
ARRAY_SUFFIXES = {".npy", ".npz", ".tif", ".tiff"}
TABLE_SUFFIXES = {".csv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe a local HYPERVIEW2 dataset tree without importing it into the "
            "HySpecNet benchmark protocol."
        )
    )
    parser.add_argument("dataset_root", type=Path, help="Path to the HYPERVIEW2 dataset root")
    parser.add_argument(
        "--max-files-to-index",
        type=int,
        default=50000,
        help="Maximum number of files to index before stopping the recursive scan.",
    )
    parser.add_argument(
        "--max-array-files",
        type=int,
        default=32,
        help="Maximum number of array/image files to inspect for shape and dtype.",
    )
    parser.add_argument(
        "--max-csv-files",
        type=int,
        default=16,
        help="Maximum number of CSV files to inspect for headers and sample rows.",
    )
    parser.add_argument(
        "--max-csv-rows",
        type=int,
        default=3,
        help="Number of preview rows to read per inspected CSV file.",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults to artifacts/logs/probe_hyperview2.json.",
    )
    return parser.parse_args()


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
    if any(token in name for token in ("gt", "label", "target", "soil")):
        return "labels"
    return "unknown"


def safe_relpath(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def list_files(root: Path, limit: int) -> tuple[list[Path], bool]:
    files: list[Path] = []
    truncated = False
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if len(files) >= limit:
            truncated = True
            break
        files.append(path)
    return files, truncated


def shape_band_candidates(shape: tuple[int, ...]) -> list[int]:
    known_counts = set(EXPECTED_BANDS.values())
    return [dim for dim in shape if dim in known_counts]


def inspect_array(path: Path, root: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    base: dict[str, Any] = {
        "path": safe_relpath(path, root),
        "suffix": suffix,
        "split": infer_split(path),
        "modality": infer_modality(path),
        "size_bytes": path.stat().st_size,
    }

    if suffix == ".npy":
        array = np.load(path, mmap_mode="r")
        shape = tuple(int(dim) for dim in array.shape)
        base.update(
            {
                "kind": "npy",
                "shape": list(shape),
                "dtype": str(array.dtype),
                "band_candidates": shape_band_candidates(shape),
            }
        )
        return base

    if suffix == ".npz":
        with np.load(path, mmap_mode="r") as archive:
            arrays = []
            for key in archive.files[:8]:
                arr = archive[key]
                shape = tuple(int(dim) for dim in arr.shape)
                arrays.append(
                    {
                        "name": key,
                        "shape": list(shape),
                        "dtype": str(arr.dtype),
                        "band_candidates": shape_band_candidates(shape),
                    }
                )
        base.update({"kind": "npz", "arrays": arrays})
        return base

    if suffix in {".tif", ".tiff"}:
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            shape = tuple(int(dim) for dim in series.shape)
            base.update(
                {
                    "kind": "tiff",
                    "shape": list(shape),
                    "dtype": str(series.dtype),
                    "band_candidates": shape_band_candidates(shape),
                }
            )
        return base

    raise ValueError(f"Unsupported array suffix: {suffix}")


def inspect_csv(path: Path, root: Path, max_rows: int) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        rows = []
        for _ in range(max_rows):
            try:
                rows.append(next(reader))
            except StopIteration:
                break

    normalized_headers = {header.strip().lower(): header for header in headers}
    target_columns = [
        original
        for normalized, original in normalized_headers.items()
        if normalized in SOIL_TARGETS
    ]
    return {
        "path": safe_relpath(path, root),
        "split": infer_split(path),
        "modality": infer_modality(path),
        "size_bytes": path.stat().st_size,
        "num_columns": len(headers),
        "columns": headers,
        "soil_target_columns": target_columns,
        "preview_rows": rows,
    }


def summarize_arrays(array_records: list[dict[str, Any]]) -> dict[str, Any]:
    by_modality: dict[str, Counter[str]] = defaultdict(Counter)
    observed_band_counts: Counter[int] = Counter()

    for record in array_records:
        modality = str(record["modality"])
        if "shape" in record:
            shape_key = "x".join(str(dim) for dim in record["shape"])
            by_modality[modality][shape_key] += 1
            for band_count in record.get("band_candidates", []):
                observed_band_counts[int(band_count)] += 1
        for array_info in record.get("arrays", []):
            shape_key = "x".join(str(dim) for dim in array_info["shape"])
            by_modality[modality][shape_key] += 1
            for band_count in array_info.get("band_candidates", []):
                observed_band_counts[int(band_count)] += 1

    return {
        "shapes_by_modality": {
            modality: dict(counter) for modality, counter in sorted(by_modality.items())
        },
        "observed_known_band_counts": dict(sorted(observed_band_counts.items())),
    }


def build_warnings(report: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    observed = {int(key) for key in report["array_summary"]["observed_known_band_counts"]}

    if EXPECTED_BANDS["hyspecnet_mamba_checkpoint"] not in observed:
        warnings.append(
            "No inspected array exposes 202 bands. Existing HySpecNet Mamba checkpoints "
            "cannot be used directly without adaptation or retraining."
        )

    if EXPECTED_BANDS["prisma"] in observed:
        warnings.append(
            "PRISMA samples appear to use 230 bands. Use a Hyperview2-specific compressor "
            "configuration instead of the 202-band HySpecNet checkpoints."
        )

    if EXPECTED_BANDS["airborne"] in observed:
        warnings.append(
            "Airborne samples appear to use 430 bands. Public challenge metadata says this "
            "modality is train-only, so it should not be the main downstream evaluation path."
        )

    csv_records = report["csv_samples"]
    has_targets = any(record["soil_target_columns"] for record in csv_records)
    if csv_records and not has_targets:
        warnings.append(
            "Inspected CSV files do not expose the six expected soil target columns "
            "(B, Cu, Zn, Fe, S, Mn). Verify label file naming and format before training."
        )

    if report["scan_truncated"]:
        warnings.append(
            "File scan was truncated. Increase --max-files-to-index before treating counts "
            "as complete."
        )

    return warnings


def print_report(report: dict[str, Any]) -> None:
    print("HYPERVIEW2 dataset probe")
    print(f"Root: {report['dataset_root']}")
    print(
        f"Files scanned: {report['num_files_scanned']}"
        + (" (truncated)" if report["scan_truncated"] else "")
    )
    print(f"Extensions: {report['file_counts_by_extension']}")
    print(f"Files by inferred split: {report['file_counts_by_split']}")
    print(f"Files by inferred modality: {report['file_counts_by_modality']}")

    print("\nArray shape summary:")
    shapes = report["array_summary"]["shapes_by_modality"]
    if not shapes:
        print("  No array files inspected.")
    for modality, counter in shapes.items():
        print(f"  {modality}: {counter}")

    print("\nCSV samples:")
    if not report["csv_samples"]:
        print("  No CSV files inspected.")
    for record in report["csv_samples"][:8]:
        print(
            f"  {record['path']} | columns={record['num_columns']} | "
            f"soil_targets={record['soil_target_columns']}"
        )

    if report["warnings"]:
        print("\nProtocol warnings:")
        for warning in report["warnings"]:
            print(f"  - {warning}")


def main() -> int:
    load_project_env()
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"HYPERVIEW2 dataset root does not exist: {root}")

    files, scan_truncated = list_files(root, args.max_files_to_index)
    file_counts_by_extension = Counter(path.suffix.lower() or "<none>" for path in files)
    file_counts_by_split = Counter(infer_split(path) for path in files)
    file_counts_by_modality = Counter(infer_modality(path) for path in files)

    array_paths = [path for path in files if path.suffix.lower() in ARRAY_SUFFIXES]
    csv_paths = [path for path in files if path.suffix.lower() in TABLE_SUFFIXES]

    array_records = []
    for path in array_paths[: args.max_array_files]:
        try:
            array_records.append(inspect_array(path, root))
        except Exception as exc:  # noqa: BLE001 - probe should continue after bad files.
            array_records.append(
                {
                    "path": safe_relpath(path, root),
                    "error": f"{type(exc).__name__}: {exc}",
                    "split": infer_split(path),
                    "modality": infer_modality(path),
                }
            )

    csv_records = []
    for path in csv_paths[: args.max_csv_files]:
        try:
            csv_records.append(inspect_csv(path, root, args.max_csv_rows))
        except Exception as exc:  # noqa: BLE001 - probe should continue after bad files.
            csv_records.append(
                {
                    "path": safe_relpath(path, root),
                    "error": f"{type(exc).__name__}: {exc}",
                    "split": infer_split(path),
                    "modality": infer_modality(path),
                    "soil_target_columns": [],
                }
            )

    report: dict[str, Any] = {
        "dataset_root": str(root),
        "expected_public_band_counts": EXPECTED_BANDS,
        "num_files_scanned": len(files),
        "scan_truncated": scan_truncated,
        "file_counts_by_extension": dict(sorted(file_counts_by_extension.items())),
        "file_counts_by_split": dict(sorted(file_counts_by_split.items())),
        "file_counts_by_modality": dict(sorted(file_counts_by_modality.items())),
        "array_samples": array_records,
        "array_summary": summarize_arrays(array_records),
        "csv_samples": csv_records,
    }
    report["warnings"] = build_warnings(report)

    output_path = args.save_json or (logs_dir() / "probe_hyperview2.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print_report(report)
    print(f"\nSaved JSON: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
