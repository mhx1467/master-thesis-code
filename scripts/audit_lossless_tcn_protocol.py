#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
import struct
import time
from collections import Counter
from math import prod
from pathlib import Path
from typing import Any

import torch

from hsi_compression.constants import NODATA_VALUE, WATER_VAPOR_BANDS
from hsi_compression.data import build_dataset
from hsi_compression.datasets import HSITiffDataset
from hsi_compression.engine.checkpointing import load_checkpoint
from hsi_compression.metrics import (
    compute_actual_bpppc_from_strings,
    compute_compression_ratio_from_bpppc,
)
from hsi_compression.models.registry import build_model
from hsi_compression.paths import logs_dir
from hsi_compression.splits import load_split_csv, split_csv_path
from hsi_compression.utils import load_config, load_project_env

DEFAULT_CONFIG = Path("configs/tcn/spectral_tcn_lossless_symbol_grid.yaml")
ORIGINAL_BITS_PER_CHANNEL = 16.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether SpectralTCNLossless really uses residual coding, preserves exact "
            "reconstruction, and reports actual bitrate on a HySpecNet split."
        )
    )
    parser.add_argument("dataset_root", type=Path, help="HySpecNet-11k dataset root")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--difficulty", default=None, choices=["easy", "hard"])
    parser.add_argument(
        "--source",
        default="data_npy",
        choices=("data_npy", "tif"),
        help=(
            "data_npy uses benchmark DATA.npy split entries; tif maps split entries to "
            "sibling *-SPECTRAL_IMAGE.TIF files and applies repository preprocessing."
        ),
    )
    parser.add_argument(
        "--allow-missing-split-entries",
        action="store_true",
        help="Filter missing TIF split paths instead of failing. Use only for local partial copies.",
    )
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--save-json", type=Path, default=None)
    parser.add_argument(
        "--require-residual-backend",
        action="store_true",
        help="Exit with a non-zero status if any sample uses the raw-float fallback backend.",
    )
    parser.add_argument(
        "--original-bits-per-channel",
        type=float,
        default=ORIGINAL_BITS_PER_CHANNEL,
        help="Reference source precision used for compression ratio reporting.",
    )
    return parser.parse_args()


def split_entry_to_tif_path(dataset_root: Path, rel_entry: str) -> Path:
    rel = Path(rel_entry)
    patch_id = rel.stem.removesuffix("-DATA")
    return dataset_root / "patches" / rel.parent / f"{patch_id}-SPECTRAL_IMAGE.TIF"


def resolve_tif_split_paths(
    dataset_root: Path,
    split: str,
    difficulty: str,
    allow_missing: bool,
) -> tuple[list[Path], int]:
    csv_path = split_csv_path(dataset_root, split, difficulty)
    paths = [split_entry_to_tif_path(dataset_root, entry) for entry in load_split_csv(csv_path)]
    missing = [path for path in paths if not path.exists()]
    if missing and not allow_missing:
        raise FileNotFoundError(
            f"{len(missing)} TIF files do not exist. First missing: {missing[0]}"
        )
    if allow_missing:
        paths = [path for path in paths if path.exists()]
    if not paths:
        raise RuntimeError("No usable TIF samples after resolving split paths.")
    return paths, len(missing)


def build_audit_dataset(
    dataset_root: Path,
    source: str,
    split: str,
    difficulty: str,
    allow_missing: bool,
    data_cfg: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    if source == "data_npy":
        dataset = build_dataset(
            dataset_root=dataset_root,
            split_name=split,
            difficulty=difficulty,
            return_mask=True,
            drop_invalid_channels=data_cfg.get("drop_invalid_channels", True),
            prefer_npy=True,
            npy_mmap=data_cfg.get("npy_mmap", False),
        )
        return dataset, {"missing_split_entries": 0}

    paths, missing = resolve_tif_split_paths(
        dataset_root=dataset_root,
        split=split,
        difficulty=difficulty,
        allow_missing=allow_missing,
    )
    dataset = HSITiffDataset(
        paths=paths,
        nodata_value=NODATA_VALUE,
        invalid_channels=WATER_VAPOR_BANDS,
        drop_invalid_channels=data_cfg.get("drop_invalid_channels", True),
        prefer_npy=False,
        return_mask=True,
    )
    return dataset, {"missing_split_entries": missing}


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def unwrap_single_bytes(strings: Any) -> bytes | None:
    if isinstance(strings, (bytes, bytearray)):
        return bytes(strings)
    if isinstance(strings, (list, tuple)) and len(strings) == 1:
        return unwrap_single_bytes(strings[0])
    return None


def sum_string_bytes(strings: Any) -> int:
    if isinstance(strings, (bytes, bytearray)):
        return len(strings)
    if isinstance(strings, (list, tuple)):
        return sum(sum_string_bytes(item) for item in strings)
    raise TypeError(f"Unsupported strings container type: {type(strings)!r}")


def read_tcn_header(strings: Any) -> dict[str, Any] | None:
    raw = unwrap_single_bytes(strings)
    if raw is None or len(raw) < 4:
        return None

    header_len = struct.unpack("<I", raw[:4])[0]
    header_start = 4
    header_end = header_start + header_len
    if header_end > len(raw):
        return None
    return json.loads(raw[header_start:header_end].decode("utf-8"))


def is_exact_symbol_grid(model: torch.nn.Module, x: torch.Tensor) -> bool | None:
    to_symbols = getattr(model, "_to_symbols", None)
    is_exact = getattr(model, "_is_exact_symbol_grid", None)
    if to_symbols is None or is_exact is None:
        return None
    symbols = to_symbols(x)
    return bool(is_exact(x, symbols))


def build_tcn_from_config(config: dict[str, Any], in_channels: int) -> torch.nn.Module:
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("model_name", "spectral_tcn_lossless")
    model_kwargs = model_cfg.get("model_kwargs", {})
    if model_name != "spectral_tcn_lossless":
        raise ValueError(f"Expected spectral_tcn_lossless config, got model_name={model_name!r}")
    return build_model(model_name, in_channels=in_channels, **model_kwargs)


def audit_sample(
    model: torch.nn.Module,
    sample: dict[str, Any],
    index: int,
    device: torch.device,
    original_bits_per_channel: float,
) -> dict[str, Any]:
    x = sample["x"].unsqueeze(0).to(device=device, dtype=torch.float32)
    mask = sample.get("valid_mask")
    if mask is not None:
        mask = mask.unsqueeze(0).to(device=device)

    exact_grid = is_exact_symbol_grid(model, x)
    symbols = model._to_symbols(x)  # noqa: SLF001 - protocol audit checks model internals.
    x_target = model._symbols_to_float(symbols)  # noqa: SLF001

    sync_if_cuda(device)
    encode_start = time.perf_counter()
    try:
        packed = model.compress(x, valid_mask=mask)
    except ValueError as exc:
        if "not exactly representable on the configured symbol grid" in str(exc):
            raise ValueError(
                "The selected input source is not exactly representable on the configured "
                "symbol grid. For the primary lossless-symbol-grid protocol, rerun with "
                "--source tif or generate exact symbol-grid DATA.npy artifacts. Do not enable "
                "raw_fallback for TCN residual claims."
            ) from exc
        raise
    sync_if_cuda(device)
    encode_ms = (time.perf_counter() - encode_start) * 1000.0

    sync_if_cuda(device)
    decode_start = time.perf_counter()
    decoded = model.decompress(packed["strings"], packed["shape"])["x_hat"]
    sync_if_cuda(device)
    decode_ms = (time.perf_counter() - decode_start) * 1000.0

    if decoded.shape != x.shape:
        raise RuntimeError(
            f"Decoded shape mismatch for sample {index}: got {tuple(decoded.shape)}, "
            f"expected {tuple(x.shape)}"
        )

    x_cpu = x.detach().cpu()
    x_target_cpu = x_target.detach().cpu()
    decoded_cpu = decoded.detach().cpu()
    mismatch_count = int((decoded_cpu != x_target_cpu).sum().item())
    max_abs_error = float((decoded_cpu - x_target_cpu).abs().max().item())
    input_canonical_mismatch_count = int((x_cpu != x_target_cpu).sum().item())
    input_max_canonical_abs_error = float((x_cpu - x_target_cpu).abs().max().item())
    encoded_bytes = sum_string_bytes(packed["strings"])
    actual_bpppc = compute_actual_bpppc_from_strings(packed["strings"], tuple(x.shape))

    header = read_tcn_header(packed["strings"])
    codec_backend = header.get("codec_backend") if header else "unknown"
    compression_ratio = compute_compression_ratio_from_bpppc(
        actual_bpppc, original_bits_per_channel
    )

    return {
        "index": index,
        "path": sample.get("path"),
        "patch_id": sample.get("patch_id"),
        "shape": list(x.shape),
        "codec_backend": codec_backend,
        "header": header,
        "exact_symbol_grid": exact_grid,
        "exact_reconstruction": mismatch_count == 0,
        "mismatch_count": mismatch_count,
        "max_abs_error": max_abs_error,
        "input_canonical_mismatch_count": input_canonical_mismatch_count,
        "input_max_canonical_abs_error": input_max_canonical_abs_error,
        "encoded_bytes": encoded_bytes,
        "actual_bpppc": actual_bpppc,
        "compression_ratio": compression_ratio,
        "encode_ms": encode_ms,
        "decode_ms": decode_ms,
    }


def summarize(
    sample_reports: list[dict[str, Any]], original_bits_per_channel: float
) -> dict[str, Any]:
    backend_counts = Counter(str(report["codec_backend"]) for report in sample_reports)
    total_bytes = 0
    total_values = 0
    for report in sample_reports:
        total_values += prod(int(dim) for dim in report["shape"])
        total_bytes += int(report["encoded_bytes"])

    pooled_bpppc = (total_bytes * 8) / total_values if total_values else None
    return {
        "num_samples": len(sample_reports),
        "backend_counts": dict(sorted(backend_counts.items())),
        "all_exact_reconstruction": all(
            report["exact_reconstruction"] for report in sample_reports
        ),
        "num_exact_symbol_grid": sum(
            1 for report in sample_reports if report["exact_symbol_grid"] is True
        ),
        "num_input_bit_exact_to_canonical": sum(
            1 for report in sample_reports if report["input_canonical_mismatch_count"] == 0
        ),
        "mean_sample_actual_bpppc": statistics.fmean(
            report["actual_bpppc"] for report in sample_reports
        ),
        "pooled_actual_bpppc": pooled_bpppc,
        "pooled_compression_ratio": compute_compression_ratio_from_bpppc(
            pooled_bpppc, original_bits_per_channel
        ),
        "mean_encode_ms": statistics.fmean(report["encode_ms"] for report in sample_reports),
        "mean_decode_ms": statistics.fmean(report["decode_ms"] for report in sample_reports),
        "max_abs_error": max(report["max_abs_error"] for report in sample_reports),
        "total_mismatch_count": sum(report["mismatch_count"] for report in sample_reports),
        "max_input_canonical_abs_error": max(
            report["input_max_canonical_abs_error"] for report in sample_reports
        ),
        "total_input_canonical_mismatch_count": sum(
            report["input_canonical_mismatch_count"] for report in sample_reports
        ),
    }


def build_warnings(summary: dict[str, Any], checkpoint: Path | None) -> list[str]:
    warnings: list[str] = []
    backend_counts = summary["backend_counts"]
    if backend_counts.get("zlib_raw_float32", 0) > 0:
        warnings.append(
            "At least one sample used zlib_raw_float32. That is exact, but it is not evidence "
            "that the TCN residual predictor is useful."
        )
    if backend_counts.get("zlib_residual", 0) == 0:
        warnings.append(
            "No inspected sample used zlib_residual. Do not report this as TCN predictive "
            "lossless compression."
        )
    if not summary["all_exact_reconstruction"]:
        warnings.append("Exact reconstruction failed for at least one sample.")
    if checkpoint is None:
        warnings.append("No checkpoint was loaded. Bitrate reflects an untrained/random TCN.")
    return warnings


def print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("Spectral TCN lossless protocol audit")
    print(f"Dataset: {report['dataset_root']}")
    print(f"Source: {report['source']}")
    print(f"Split: {report['difficulty']}/{report['split']}")
    print(f"Checkpoint: {report['checkpoint'] or 'none'}")
    print(f"Samples audited: {summary['num_samples']}")
    print(f"Backend counts: {summary['backend_counts']}")
    print(f"Exact reconstruction: {summary['all_exact_reconstruction']}")
    print(f"Exact symbol-grid samples: {summary['num_exact_symbol_grid']}/{summary['num_samples']}")
    print(
        "Input bit-exact to canonical grid: "
        f"{summary['num_input_bit_exact_to_canonical']}/{summary['num_samples']}"
    )
    print(f"Pooled actual bpppc: {summary['pooled_actual_bpppc']:.6f}")
    print(f"Pooled CR: {summary['pooled_compression_ratio']:.4f}:1")
    print(
        f"Mean encode/decode: {summary['mean_encode_ms']:.2f} / {summary['mean_decode_ms']:.2f} ms"
    )
    print(f"Total mismatches: {summary['total_mismatch_count']}")
    print(f"Max |err|: {summary['max_abs_error']:.10f}")
    print(
        "Input canonicalization mismatches: "
        f"{summary['total_input_canonical_mismatch_count']} | "
        f"max |input-canonical|={summary['max_input_canonical_abs_error']:.10f}"
    )

    if report["warnings"]:
        print("\nProtocol warnings:")
        for warning in report["warnings"]:
            print(f"  - {warning}")


def main() -> int:
    load_project_env()
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    difficulty = args.difficulty or data_cfg.get("difficulty", "easy")

    dataset, dataset_meta = build_audit_dataset(
        dataset_root=dataset_root,
        source=args.source,
        split=args.split,
        difficulty=difficulty,
        allow_missing=args.allow_missing_split_entries,
        data_cfg=data_cfg,
    )

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty.")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")

    first_sample = dataset[0]
    in_channels = int(first_sample["x"].shape[0])
    device = select_device(args.device)
    model = build_tcn_from_config(config, in_channels=in_channels).to(device)
    model.eval()

    checkpoint_path = None
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint.expanduser().resolve()
        load_checkpoint(checkpoint_path, model, map_location=device)

    sample_reports = []
    num_samples = min(args.num_samples, len(dataset))
    with torch.no_grad():
        for index in range(num_samples):
            sample_reports.append(
                audit_sample(
                    model=model,
                    sample=dataset[index],
                    index=index,
                    device=device,
                    original_bits_per_channel=args.original_bits_per_channel,
                )
            )

    summary = summarize(sample_reports, args.original_bits_per_channel)
    report: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "config": str(args.config),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "source": args.source,
        "split": args.split,
        "difficulty": difficulty,
        "device": str(device),
        "in_channels": in_channels,
        "original_bits_per_channel": args.original_bits_per_channel,
        "dataset_meta": dataset_meta,
        "summary": summary,
        "samples": sample_reports,
    }
    report["warnings"] = build_warnings(summary, checkpoint_path)

    output_path = args.save_json or (logs_dir() / "audit_lossless_tcn_protocol.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print_report(report)
    print(f"\nSaved JSON: {output_path}")

    if args.require_residual_backend and summary["backend_counts"].get("zlib_raw_float32", 0) > 0:
        return 2
    if not summary["all_exact_reconstruction"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
