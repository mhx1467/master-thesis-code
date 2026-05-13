#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import lzma
import statistics
import struct
import time
import zlib
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover - optional dependency
    zstd = None

from hsi_compression.constants import NODATA_VALUE, WATER_VAPOR_BANDS
from hsi_compression.data import build_dataset
from hsi_compression.datasets import HSITiffDataset
from hsi_compression.engine.checkpointing import load_checkpoint
from hsi_compression.metrics import compute_compression_ratio_from_bpppc
from hsi_compression.models.registry import build_model
from hsi_compression.paths import logs_dir
from hsi_compression.splits import load_split_csv, split_csv_path
from hsi_compression.utils import load_config, load_project_env

DEFAULT_CONFIG = Path("configs/tcn/spectral_tcn_lossless_symbol_grid.yaml")
DEFAULT_ORIGINAL_BITS_PER_CHANNEL = 16.0
DEFAULT_CODECS = (
    "raw_zlib",
    "raw_lzma",
    "raw_zstd",
    "symbols_zlib",
    "symbols_zstd",
    "bitplane_symbols_zstd",
    "spectral_delta_zlib",
    "spectral_delta_zstd",
    "bitplane_spectral_delta_zstd",
    "tcn_residual_zlib",
    "tcn_residual_zstd",
)


@dataclass(frozen=True)
class CodecResult:
    codec: str
    skipped: bool
    skipped_reason: str | None
    codec_backend: str | None
    encoded_bytes: int | None
    actual_bpppc: float | None
    compression_ratio: float | None
    exact_reconstruction: bool | None
    mismatch_count: int | None
    max_abs_error: float | None
    encode_ms: float | None
    decode_ms: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate simple lossless baselines and SpectralTCNLossless on the same "
            "HySpecNet split/source protocol."
        )
    )
    parser.add_argument("dataset_root", type=Path, help="HySpecNet-11k dataset root")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--difficulty", default=None, choices=("easy", "hard"))
    parser.add_argument(
        "--source",
        default="data_npy",
        choices=("data_npy", "tif"),
        help=(
            "data_npy uses benchmark DATA.npy split entries; tif maps the same split entries "
            "to sibling *-SPECTRAL_IMAGE.TIF files and then applies repository preprocessing."
        ),
    )
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--codecs", default=",".join(DEFAULT_CODECS))
    parser.add_argument("--symbol-scale", type=int, default=None)
    parser.add_argument("--zlib-level", type=int, default=9)
    parser.add_argument(
        "--allow-missing-split-entries",
        action="store_true",
        help="Filter missing split paths instead of failing. Use only for local incomplete copies.",
    )
    parser.add_argument(
        "--original-bits-per-channel",
        type=float,
        default=DEFAULT_ORIGINAL_BITS_PER_CHANNEL,
        help="Reference source precision for compression-ratio reporting.",
    )
    parser.add_argument("--save-json", type=Path, default=None)
    parser.add_argument("--save-csv", type=Path, default=None)
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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
        raise RuntimeError("No usable samples after resolving split paths.")
    return paths, len(missing)


def build_eval_dataset(
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


def build_tcn_model(
    config: dict[str, Any],
    in_channels: int,
    device: torch.device,
    checkpoint: Path | None,
) -> torch.nn.Module:
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("model_name", "spectral_tcn_lossless")
    model_kwargs = model_cfg.get("model_kwargs", {})
    if model_name != "spectral_tcn_lossless":
        raise ValueError(f"Expected spectral_tcn_lossless config, got {model_name!r}")
    model = build_model(model_name, in_channels=in_channels, **model_kwargs).to(device)
    model.eval()
    if checkpoint is not None:
        load_checkpoint(checkpoint, model, map_location=device)
    return model


def pack_payload(header: dict[str, Any], payload: bytes) -> bytes:
    header_bytes = json.dumps(header, sort_keys=True).encode("utf-8")
    return struct.pack("<I", len(header_bytes)) + header_bytes + payload


def unpack_payload(strings: bytes) -> tuple[dict[str, Any], bytes]:
    header_len = struct.unpack("<I", strings[:4])[0]
    header_start = 4
    header_end = header_start + header_len
    return json.loads(strings[header_start:header_end].decode("utf-8")), strings[header_end:]


def zlib_compress(data: bytes, level: int) -> bytes:
    return zlib.compress(data, level=level)


def zlib_decompress(data: bytes) -> bytes:
    return zlib.decompress(data)


def lzma_compress(data: bytes, level: int) -> bytes:
    del level
    return lzma.compress(data, preset=9)


def lzma_decompress(data: bytes) -> bytes:
    return lzma.decompress(data)


def zstd_compress(data: bytes, level: int) -> bytes:
    if zstd is None:
        raise RuntimeError("Optional dependency 'zstandard' is not installed.")
    compressor = zstd.ZstdCompressor(level=level)
    return compressor.compress(data)


def zstd_decompress(data: bytes) -> bytes:
    if zstd is None:
        raise RuntimeError("Optional dependency 'zstandard' is not installed.")
    decompressor = zstd.ZstdDecompressor()
    return decompressor.decompress(data)


def tensor_to_symbols(x: torch.Tensor, symbol_scale: int) -> torch.Tensor:
    return torch.round(x.clamp(0.0, 1.0) * symbol_scale).to(torch.int32)


def symbols_to_tensor(symbols: torch.Tensor, symbol_scale: int) -> torch.Tensor:
    return symbols.to(torch.float32) / float(symbol_scale)


def is_exact_symbol_grid(x: torch.Tensor, symbols: torch.Tensor, symbol_scale: int) -> bool:
    x_float = x.float()
    scaled = x_float * float(symbol_scale)
    finite = torch.isfinite(scaled).all()
    in_range = (x_float >= -1e-7).all() and (x_float <= 1.0 + 1e-7).all()
    close_to_symbols = torch.allclose(
        scaled,
        symbols.to(dtype=torch.float32, device=x.device),
        rtol=0.0,
        atol=1e-3,
    )
    return bool(finite and in_range and close_to_symbols)


def uint16_to_bitplane_bytes(values: np.ndarray) -> bytes:
    flat = np.ascontiguousarray(values.astype(np.uint16, copy=False)).reshape(-1)
    bytes_view = flat.view(np.uint8).reshape(-1, 2)
    bits = np.unpackbits(bytes_view, axis=1, bitorder="little")
    return np.packbits(bits.T.reshape(-1), bitorder="little").tobytes()


def bitplane_bytes_to_uint16(data: bytes, shape: Sequence[int]) -> np.ndarray:
    num_values = prod(int(dim) for dim in shape)
    packed = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(packed, bitorder="little")[: num_values * 16]
    bits = bits.reshape(16, num_values).T
    bytes_view = np.ascontiguousarray(np.packbits(bits, axis=1, bitorder="little"))
    return bytes_view.reshape(num_values, 2).view(np.uint16).reshape(shape).copy()


def zigzag_encode_int32(values: np.ndarray) -> np.ndarray:
    values_i32 = values.astype(np.int32, copy=False)
    mapped = np.where(values_i32 >= 0, values_i32 * 2, (-values_i32 * 2) - 1)
    return mapped.astype(np.uint16)


def zigzag_decode_uint16(values: np.ndarray) -> np.ndarray:
    values_i32 = np.array(values, dtype=np.int32, copy=True)
    half = np.right_shift(values_i32, 1)
    sign = -np.bitwise_and(values_i32, 1)
    return np.bitwise_xor(half, sign).astype(np.int32)


def summarize_roundtrip(
    codec: str,
    x: torch.Tensor,
    encoded: bytes,
    decoded: torch.Tensor,
    encode_ms: float,
    decode_ms: float,
    original_bits_per_channel: float,
    codec_backend: str,
) -> CodecResult:
    decoded = decoded.to(device=x.device, dtype=x.dtype)
    mismatch_count = int((decoded != x).sum().item())
    max_abs_error = float((decoded - x).abs().max().item())
    total_values = prod(int(dim) for dim in x.shape)
    actual_bpppc = (len(encoded) * 8) / total_values
    return CodecResult(
        codec=codec,
        skipped=False,
        skipped_reason=None,
        codec_backend=codec_backend,
        encoded_bytes=len(encoded),
        actual_bpppc=actual_bpppc,
        compression_ratio=compute_compression_ratio_from_bpppc(
            actual_bpppc, original_bits_per_channel
        ),
        exact_reconstruction=mismatch_count == 0,
        mismatch_count=mismatch_count,
        max_abs_error=max_abs_error,
        encode_ms=encode_ms,
        decode_ms=decode_ms,
    )


def skipped(codec: str, reason: str) -> CodecResult:
    return CodecResult(
        codec=codec,
        skipped=True,
        skipped_reason=reason,
        codec_backend=None,
        encoded_bytes=None,
        actual_bpppc=None,
        compression_ratio=None,
        exact_reconstruction=None,
        mismatch_count=None,
        max_abs_error=None,
        encode_ms=None,
        decode_ms=None,
    )


def run_raw_codec(
    codec: str,
    x: torch.Tensor,
    compressor: Callable[[bytes, int], bytes],
    decompressor: Callable[[bytes], bytes],
    level: int,
    original_bits_per_channel: float,
) -> CodecResult:
    x_cpu = x.detach().cpu().contiguous().to(torch.float32)
    array = np.ascontiguousarray(x_cpu.numpy().astype(np.float32, copy=False))

    start = time.perf_counter()
    encoded = pack_payload(
        {"codec_backend": codec, "dtype": "float32", "shape": list(array.shape)},
        compressor(array.tobytes(order="C"), level),
    )
    encode_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    header, payload = unpack_payload(encoded)
    decoded_array = np.frombuffer(decompressor(payload), dtype=np.float32).copy()
    decoded_array = decoded_array.reshape(header["shape"])
    decoded = torch.from_numpy(decoded_array)
    decode_ms = (time.perf_counter() - start) * 1000.0

    return summarize_roundtrip(
        codec=codec,
        x=x_cpu,
        encoded=encoded,
        decoded=decoded,
        encode_ms=encode_ms,
        decode_ms=decode_ms,
        original_bits_per_channel=original_bits_per_channel,
        codec_backend=codec,
    )


def run_symbols_zlib(
    x: torch.Tensor,
    symbol_scale: int,
    level: int,
    original_bits_per_channel: float,
) -> CodecResult:
    x_cpu = x.detach().cpu().contiguous().to(torch.float32)
    symbols = tensor_to_symbols(x_cpu, symbol_scale)
    if not is_exact_symbol_grid(x_cpu, symbols, symbol_scale):
        return skipped("symbols_zlib", "input is not exactly representable on symbol grid")
    x_target = symbols_to_tensor(symbols, symbol_scale)

    array = np.ascontiguousarray(symbols.numpy().astype(np.int16, copy=False))
    start = time.perf_counter()
    encoded = pack_payload(
        {
            "codec_backend": "symbols_zlib",
            "dtype": "int16",
            "shape": list(array.shape),
            "symbol_scale": symbol_scale,
        },
        zlib.compress(array.tobytes(order="C"), level=level),
    )
    encode_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    header, payload = unpack_payload(encoded)
    decoded_array = np.frombuffer(zlib.decompress(payload), dtype=np.int16).copy()
    decoded_symbols = torch.from_numpy(decoded_array.reshape(header["shape"])).to(torch.int32)
    decoded = symbols_to_tensor(decoded_symbols, symbol_scale)
    decode_ms = (time.perf_counter() - start) * 1000.0

    return summarize_roundtrip(
        codec="symbols_zlib",
        x=x_target,
        encoded=encoded,
        decoded=decoded,
        encode_ms=encode_ms,
        decode_ms=decode_ms,
        original_bits_per_channel=original_bits_per_channel,
        codec_backend="symbols_zlib",
    )


def run_spectral_delta_zlib(
    x: torch.Tensor,
    symbol_scale: int,
    level: int,
    original_bits_per_channel: float,
) -> CodecResult:
    x_cpu = x.detach().cpu().contiguous().to(torch.float32)
    symbols = tensor_to_symbols(x_cpu, symbol_scale)
    if not is_exact_symbol_grid(x_cpu, symbols, symbol_scale):
        return skipped("spectral_delta_zlib", "input is not exactly representable on symbol grid")
    x_target = symbols_to_tensor(symbols, symbol_scale)

    symbols_np = np.ascontiguousarray(symbols.numpy().astype(np.int32, copy=False))
    residuals = np.empty_like(symbols_np, dtype=np.int16)
    residuals[:, 0] = symbols_np[:, 0]
    residuals[:, 1:] = symbols_np[:, 1:] - symbols_np[:, :-1]

    start = time.perf_counter()
    encoded = pack_payload(
        {
            "codec_backend": "spectral_delta_zlib",
            "dtype": "int16",
            "shape": list(residuals.shape),
            "symbol_scale": symbol_scale,
        },
        zlib.compress(residuals.tobytes(order="C"), level=level),
    )
    encode_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    header, payload = unpack_payload(encoded)
    decoded_residuals = np.frombuffer(zlib.decompress(payload), dtype=np.int16).copy()
    decoded_residuals = decoded_residuals.reshape(header["shape"]).astype(np.int32)
    decoded_symbols = torch.from_numpy(np.cumsum(decoded_residuals, axis=1)).to(torch.int32)
    decoded = symbols_to_tensor(decoded_symbols, symbol_scale)
    decode_ms = (time.perf_counter() - start) * 1000.0

    return summarize_roundtrip(
        codec="spectral_delta_zlib",
        x=x_target,
        encoded=encoded,
        decoded=decoded,
        encode_ms=encode_ms,
        decode_ms=decode_ms,
        original_bits_per_channel=original_bits_per_channel,
        codec_backend="spectral_delta_zlib",
    )


def run_symbols_codec(
    codec: str,
    x: torch.Tensor,
    symbol_scale: int,
    compressor: Callable[[bytes, int], bytes],
    decompressor: Callable[[bytes], bytes],
    level: int,
    original_bits_per_channel: float,
    bitplane: bool = False,
) -> CodecResult:
    x_cpu = x.detach().cpu().contiguous().to(torch.float32)
    symbols = tensor_to_symbols(x_cpu, symbol_scale)
    if not is_exact_symbol_grid(x_cpu, symbols, symbol_scale):
        return skipped(codec, "input is not exactly representable on symbol grid")
    x_target = symbols_to_tensor(symbols, symbol_scale)

    symbols_np = np.ascontiguousarray(symbols.numpy().astype(np.uint16, copy=False))
    payload = uint16_to_bitplane_bytes(symbols_np) if bitplane else symbols_np.tobytes(order="C")

    start = time.perf_counter()
    encoded = pack_payload(
        {
            "codec_backend": codec,
            "dtype": "uint16",
            "shape": list(symbols_np.shape),
            "symbol_scale": symbol_scale,
            "transform": "bitplane" if bitplane else "none",
        },
        compressor(payload, level),
    )
    encode_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    header, compressed_payload = unpack_payload(encoded)
    decoded_payload = decompressor(compressed_payload)
    if bitplane:
        decoded_symbols_np = bitplane_bytes_to_uint16(decoded_payload, header["shape"]).astype(
            np.int32
        )
    else:
        decoded_symbols_np = (
            np.frombuffer(decoded_payload, dtype=np.uint16)
            .copy()
            .reshape(header["shape"])
            .astype(np.int32)
        )
    decoded = symbols_to_tensor(torch.from_numpy(decoded_symbols_np), symbol_scale)
    decode_ms = (time.perf_counter() - start) * 1000.0

    return summarize_roundtrip(
        codec=codec,
        x=x_target,
        encoded=encoded,
        decoded=decoded,
        encode_ms=encode_ms,
        decode_ms=decode_ms,
        original_bits_per_channel=original_bits_per_channel,
        codec_backend=codec,
    )


def run_spectral_delta_codec(
    codec: str,
    x: torch.Tensor,
    symbol_scale: int,
    compressor: Callable[[bytes, int], bytes],
    decompressor: Callable[[bytes], bytes],
    level: int,
    original_bits_per_channel: float,
    bitplane: bool = False,
) -> CodecResult:
    x_cpu = x.detach().cpu().contiguous().to(torch.float32)
    symbols = tensor_to_symbols(x_cpu, symbol_scale)
    if not is_exact_symbol_grid(x_cpu, symbols, symbol_scale):
        return skipped(codec, "input is not exactly representable on symbol grid")
    x_target = symbols_to_tensor(symbols, symbol_scale)

    symbols_np = np.ascontiguousarray(symbols.numpy().astype(np.int32, copy=False))
    residuals = np.empty_like(symbols_np, dtype=np.int32)
    residuals[:, 0] = symbols_np[:, 0]
    residuals[:, 1:] = symbols_np[:, 1:] - symbols_np[:, :-1]

    if bitplane:
        residual_payload = uint16_to_bitplane_bytes(zigzag_encode_int32(residuals))
        payload_dtype = "uint16"
        transform = "zigzag+bitplane"
    else:
        residual_payload = np.ascontiguousarray(residuals.astype(np.int16)).tobytes(order="C")
        payload_dtype = "int16"
        transform = "none"

    start = time.perf_counter()
    encoded = pack_payload(
        {
            "codec_backend": codec,
            "dtype": payload_dtype,
            "shape": list(residuals.shape),
            "symbol_scale": symbol_scale,
            "transform": transform,
        },
        compressor(residual_payload, level),
    )
    encode_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    header, compressed_payload = unpack_payload(encoded)
    decoded_payload = decompressor(compressed_payload)
    if bitplane:
        decoded_mapped = bitplane_bytes_to_uint16(decoded_payload, header["shape"])
        decoded_residuals = zigzag_decode_uint16(decoded_mapped)
    else:
        decoded_residuals = (
            np.frombuffer(decoded_payload, dtype=np.int16)
            .copy()
            .reshape(header["shape"])
            .astype(np.int32)
        )
    decoded_symbols = torch.from_numpy(np.cumsum(decoded_residuals, axis=1)).to(torch.int32)
    decoded = symbols_to_tensor(decoded_symbols, symbol_scale)
    decode_ms = (time.perf_counter() - start) * 1000.0

    return summarize_roundtrip(
        codec=codec,
        x=x_target,
        encoded=encoded,
        decoded=decoded,
        encode_ms=encode_ms,
        decode_ms=decode_ms,
        original_bits_per_channel=original_bits_per_channel,
        codec_backend=codec,
    )


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


def unwrap_single_bytes(strings: Any) -> bytes | None:
    if isinstance(strings, (bytes, bytearray)):
        return bytes(strings)
    if (
        isinstance(strings, Sequence)
        and not isinstance(strings, (str, bytes, bytearray))
        and len(strings) == 1
    ):
        return unwrap_single_bytes(strings[0])
    return None


def sum_string_bytes(strings: Any) -> int:
    if isinstance(strings, (bytes, bytearray)):
        return len(strings)
    if isinstance(strings, Sequence) and not isinstance(strings, (str, bytes, bytearray)):
        return sum(sum_string_bytes(item) for item in strings)
    raise TypeError(f"Unsupported strings container type: {type(strings)!r}")


def run_tcn_residual_codec(
    codec: str,
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    compressor: Callable[[bytes, int], bytes],
    decompressor: Callable[[bytes], bytes],
    level: int,
    original_bits_per_channel: float,
) -> CodecResult:
    x_device = x.to(device=device, dtype=torch.float32)
    symbols = model._to_symbols(x_device)  # noqa: SLF001 - codec audit script uses model internals.
    if not model._is_exact_symbol_grid(x_device, symbols):  # noqa: SLF001
        return skipped(codec, "input is not exactly representable on symbol grid")
    x_target = model._symbols_to_float(symbols).detach().cpu()  # noqa: SLF001

    sync_if_cuda(device)
    start = time.perf_counter()
    with torch.no_grad():
        predicted_symbols = model._predict_symbols_sequential_from_symbols(symbols)  # noqa: SLF001
        residuals = (symbols - predicted_symbols).to(torch.int32)
    sync_if_cuda(device)

    residual_min = int(residuals.min().item())
    residual_max = int(residuals.max().item())
    residual_dtype = np.int16 if residual_min >= -32768 and residual_max <= 32767 else np.int32
    residual_array = np.ascontiguousarray(residuals.cpu().numpy().astype(residual_dtype))
    encoded = pack_payload(
        {
            "codec_backend": codec,
            "dtype": np.dtype(residual_dtype).name,
            "shape": list(residuals.shape),
            "symbol_scale": int(model.symbol_scale),
            "transform": "none",
        },
        compressor(residual_array.tobytes(order="C"), level),
    )
    encode_ms = (time.perf_counter() - start) * 1000.0

    sync_if_cuda(device)
    start = time.perf_counter()
    header, compressed_payload = unpack_payload(encoded)
    residual_payload = decompressor(compressed_payload)
    decoded_residual_array = (
        np.frombuffer(residual_payload, dtype=np.dtype(header["dtype"]))
        .copy()
        .reshape(header["shape"])
    )
    decoded_residuals = torch.from_numpy(decoded_residual_array).to(
        device=device, dtype=torch.int32
    )
    with torch.no_grad():
        decoded_symbols = model._decode_symbols_from_residuals(decoded_residuals)  # noqa: SLF001
        decoded = model._symbols_to_float(decoded_symbols)  # noqa: SLF001
    sync_if_cuda(device)
    decode_ms = (time.perf_counter() - start) * 1000.0

    return summarize_roundtrip(
        codec=codec,
        x=x_target,
        encoded=encoded,
        decoded=decoded.detach().cpu(),
        encode_ms=encode_ms,
        decode_ms=decode_ms,
        original_bits_per_channel=original_bits_per_channel,
        codec_backend=codec,
    )


def run_tcn_codec(
    model: torch.nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor | None,
    device: torch.device,
    original_bits_per_channel: float,
) -> CodecResult:
    x_device = x.to(device=device, dtype=torch.float32)
    mask_device = mask.to(device=device) if mask is not None else None

    sync_if_cuda(device)
    start = time.perf_counter()
    with torch.no_grad():
        packed = model.compress(x_device, valid_mask=mask_device)
    sync_if_cuda(device)
    encode_ms = (time.perf_counter() - start) * 1000.0

    sync_if_cuda(device)
    start = time.perf_counter()
    with torch.no_grad():
        decoded = model.decompress(packed["strings"], packed["shape"])["x_hat"]
    sync_if_cuda(device)
    decode_ms = (time.perf_counter() - start) * 1000.0

    encoded_bytes = sum_string_bytes(packed["strings"])
    header = read_tcn_header(packed["strings"])
    codec_backend = str(header.get("codec_backend", "unknown")) if header else "unknown"
    fake_encoded = b"0" * encoded_bytes
    return summarize_roundtrip(
        codec="tcn_residual_zlib",
        x=x_device.detach().cpu(),
        encoded=fake_encoded,
        decoded=decoded.detach().cpu(),
        encode_ms=encode_ms,
        decode_ms=decode_ms,
        original_bits_per_channel=original_bits_per_channel,
        codec_backend=codec_backend,
    )


def evaluate_sample(
    sample: dict[str, Any],
    index: int,
    codecs: set[str],
    tcn_model: torch.nn.Module | None,
    device: torch.device,
    symbol_scale: int,
    zlib_level: int,
    original_bits_per_channel: float,
) -> dict[str, Any]:
    x = sample["x"].unsqueeze(0).contiguous()

    reports: list[CodecResult] = []
    if "raw_zlib" in codecs:
        reports.append(
            run_raw_codec(
                codec="raw_zlib",
                x=x,
                compressor=zlib_compress,
                decompressor=zlib_decompress,
                level=zlib_level,
                original_bits_per_channel=original_bits_per_channel,
            )
        )
    if "raw_lzma" in codecs:
        reports.append(
            run_raw_codec(
                codec="raw_lzma",
                x=x,
                compressor=lzma_compress,
                decompressor=lzma_decompress,
                level=zlib_level,
                original_bits_per_channel=original_bits_per_channel,
            )
        )
    if "raw_zstd" in codecs:
        if zstd is None:
            reports.append(skipped("raw_zstd", "optional dependency 'zstandard' is not installed"))
        else:
            reports.append(
                run_raw_codec(
                    codec="raw_zstd",
                    x=x,
                    compressor=zstd_compress,
                    decompressor=zstd_decompress,
                    level=zlib_level,
                    original_bits_per_channel=original_bits_per_channel,
                )
            )
    if "symbols_zlib" in codecs:
        reports.append(run_symbols_zlib(x, symbol_scale, zlib_level, original_bits_per_channel))
    if "symbols_zstd" in codecs:
        if zstd is None:
            reports.append(
                skipped("symbols_zstd", "optional dependency 'zstandard' is not installed")
            )
        else:
            reports.append(
                run_symbols_codec(
                    codec="symbols_zstd",
                    x=x,
                    symbol_scale=symbol_scale,
                    compressor=zstd_compress,
                    decompressor=zstd_decompress,
                    level=zlib_level,
                    original_bits_per_channel=original_bits_per_channel,
                )
            )
    if "bitplane_symbols_zstd" in codecs:
        if zstd is None:
            reports.append(
                skipped(
                    "bitplane_symbols_zstd",
                    "optional dependency 'zstandard' is not installed",
                )
            )
        else:
            reports.append(
                run_symbols_codec(
                    codec="bitplane_symbols_zstd",
                    x=x,
                    symbol_scale=symbol_scale,
                    compressor=zstd_compress,
                    decompressor=zstd_decompress,
                    level=zlib_level,
                    original_bits_per_channel=original_bits_per_channel,
                    bitplane=True,
                )
            )
    if "spectral_delta_zlib" in codecs:
        reports.append(
            run_spectral_delta_zlib(x, symbol_scale, zlib_level, original_bits_per_channel)
        )
    if "spectral_delta_zstd" in codecs:
        if zstd is None:
            reports.append(
                skipped("spectral_delta_zstd", "optional dependency 'zstandard' is not installed")
            )
        else:
            reports.append(
                run_spectral_delta_codec(
                    codec="spectral_delta_zstd",
                    x=x,
                    symbol_scale=symbol_scale,
                    compressor=zstd_compress,
                    decompressor=zstd_decompress,
                    level=zlib_level,
                    original_bits_per_channel=original_bits_per_channel,
                )
            )
    if "bitplane_spectral_delta_zstd" in codecs:
        if zstd is None:
            reports.append(
                skipped(
                    "bitplane_spectral_delta_zstd",
                    "optional dependency 'zstandard' is not installed",
                )
            )
        else:
            reports.append(
                run_spectral_delta_codec(
                    codec="bitplane_spectral_delta_zstd",
                    x=x,
                    symbol_scale=symbol_scale,
                    compressor=zstd_compress,
                    decompressor=zstd_decompress,
                    level=zlib_level,
                    original_bits_per_channel=original_bits_per_channel,
                    bitplane=True,
                )
            )
    if "tcn_residual_zlib" in codecs:
        if tcn_model is None:
            reports.append(skipped("tcn_residual_zlib", "TCN model was not built"))
        else:
            reports.append(
                run_tcn_residual_codec(
                    codec="tcn_residual_zlib",
                    model=tcn_model,
                    x=x,
                    device=device,
                    compressor=zlib_compress,
                    decompressor=zlib_decompress,
                    level=zlib_level,
                    original_bits_per_channel=original_bits_per_channel,
                )
            )
    if "tcn_residual_zstd" in codecs:
        if tcn_model is None:
            reports.append(skipped("tcn_residual_zstd", "TCN model was not built"))
        elif zstd is None:
            reports.append(
                skipped("tcn_residual_zstd", "optional dependency 'zstandard' is not installed")
            )
        else:
            reports.append(
                run_tcn_residual_codec(
                    codec="tcn_residual_zstd",
                    model=tcn_model,
                    x=x,
                    device=device,
                    compressor=zstd_compress,
                    decompressor=zstd_decompress,
                    level=zlib_level,
                    original_bits_per_channel=original_bits_per_channel,
                )
            )

    return {
        "index": index,
        "path": sample.get("path"),
        "patch_id": sample.get("patch_id"),
        "shape": list(x.shape),
        "results": [result.__dict__ for result in reports],
    }


def summarize_reports(sample_reports: list[dict[str, Any]], original_bits: float) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_values_by_codec: dict[str, int] = defaultdict(int)

    for sample in sample_reports:
        sample_values = prod(int(dim) for dim in sample["shape"])
        for result in sample["results"]:
            grouped[result["codec"]].append(result)
            if not result["skipped"]:
                total_values_by_codec[result["codec"]] += sample_values

    summary = {}
    for codec, results in sorted(grouped.items()):
        usable = [result for result in results if not result["skipped"]]
        skipped_count = len(results) - len(usable)
        if not usable:
            summary[codec] = {
                "num_samples": len(results),
                "num_usable": 0,
                "num_skipped": skipped_count,
                "skip_reasons": sorted(
                    {str(result["skipped_reason"]) for result in results if result["skipped"]}
                ),
            }
            continue

        total_bytes = sum(int(result["encoded_bytes"]) for result in usable)
        pooled_bpppc = (total_bytes * 8) / total_values_by_codec[codec]
        summary[codec] = {
            "num_samples": len(results),
            "num_usable": len(usable),
            "num_skipped": skipped_count,
            "backend_counts": dict(
                sorted(
                    {
                        backend: sum(1 for result in usable if result["codec_backend"] == backend)
                        for backend in {result["codec_backend"] for result in usable}
                    }.items()
                )
            ),
            "all_exact_reconstruction": all(result["exact_reconstruction"] for result in usable),
            "total_mismatch_count": sum(int(result["mismatch_count"]) for result in usable),
            "max_abs_error": max(float(result["max_abs_error"]) for result in usable),
            "pooled_actual_bpppc": pooled_bpppc,
            "pooled_compression_ratio": compute_compression_ratio_from_bpppc(
                pooled_bpppc, original_bits
            ),
            "mean_sample_actual_bpppc": statistics.fmean(
                float(result["actual_bpppc"]) for result in usable
            ),
            "mean_encode_ms": statistics.fmean(float(result["encode_ms"]) for result in usable),
            "mean_decode_ms": statistics.fmean(float(result["decode_ms"]) for result in usable),
        }
    return summary


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "codec",
        "num_usable",
        "num_skipped",
        "pooled_actual_bpppc",
        "pooled_compression_ratio",
        "mean_sample_actual_bpppc",
        "all_exact_reconstruction",
        "total_mismatch_count",
        "max_abs_error",
        "mean_encode_ms",
        "mean_decode_ms",
        "backend_counts",
        "skip_reasons",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for codec, values in sorted(summary.items()):
            row = {"codec": codec, **values}
            row["backend_counts"] = json.dumps(row.get("backend_counts", {}), sort_keys=True)
            row["skip_reasons"] = json.dumps(row.get("skip_reasons", []), sort_keys=True)
            writer.writerow({field: row.get(field) for field in fields})


def print_summary(summary: dict[str, Any]) -> None:
    print("\nLossless codec summary")
    for codec, values in sorted(summary.items()):
        if values.get("num_usable", 0) == 0:
            print(
                f"  {codec}: no usable samples "
                f"(skipped={values.get('num_skipped')}, reasons={values.get('skip_reasons')})"
            )
            continue
        print(
            f"  {codec}: bpppc={values['pooled_actual_bpppc']:.6f}, "
            f"CR={values['pooled_compression_ratio']:.4f}:1, "
            f"exact={values['all_exact_reconstruction']}, "
            f"backend={values.get('backend_counts', {})}"
        )


def main() -> int:
    load_project_env()
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {}).get("model_kwargs", {})
    difficulty = args.difficulty or data_cfg.get("difficulty", "easy")
    symbol_scale = int(args.symbol_scale or model_cfg.get("symbol_scale", 10000))
    codecs = {codec.strip() for codec in args.codecs.split(",") if codec.strip()}
    unknown = codecs - set(DEFAULT_CODECS)
    if unknown:
        raise ValueError(f"Unknown codecs: {sorted(unknown)}. Available: {DEFAULT_CODECS}")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")

    dataset, dataset_meta = build_eval_dataset(
        dataset_root=dataset_root,
        source=args.source,
        split=args.split,
        difficulty=difficulty,
        allow_missing=args.allow_missing_split_entries,
        data_cfg=data_cfg,
    )
    first = dataset[0]
    in_channels = int(first["x"].shape[0])
    device = select_device(args.device)

    tcn_model = None
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else None
    if {"tcn_residual_zlib", "tcn_residual_zstd"} & codecs:
        tcn_model = build_tcn_model(config, in_channels, device, checkpoint)

    num_samples = min(args.num_samples, len(dataset))
    print(
        f"Dataset: {dataset_root} | source={args.source} | split={difficulty}/{args.split} | "
        f"samples={num_samples}/{len(dataset)} | symbol_scale={symbol_scale}"
    )
    if checkpoint is None and {"tcn_residual_zlib", "tcn_residual_zstd"} & codecs:
        print("Warning: no TCN checkpoint loaded; TCN residual bitrate uses random weights.")

    sample_reports = []
    for index in range(num_samples):
        sample_reports.append(
            evaluate_sample(
                sample=dataset[index],
                index=index,
                codecs=codecs,
                tcn_model=tcn_model,
                device=device,
                symbol_scale=symbol_scale,
                zlib_level=args.zlib_level,
                original_bits_per_channel=args.original_bits_per_channel,
            )
        )

    summary = summarize_reports(sample_reports, args.original_bits_per_channel)
    report = {
        "dataset_root": str(dataset_root),
        "source": args.source,
        "split": args.split,
        "difficulty": difficulty,
        "num_samples": num_samples,
        "in_channels": in_channels,
        "symbol_scale": symbol_scale,
        "original_bits_per_channel": args.original_bits_per_channel,
        "config": str(args.config),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "device": str(device),
        "dataset_meta": dataset_meta,
        "summary": summary,
        "samples": sample_reports,
    }

    out_json = args.save_json or (logs_dir() / "eval_lossless_codecs.json")
    out_csv = args.save_csv or (logs_dir() / "eval_lossless_codecs_summary.csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_summary_csv(out_csv, summary)

    print_summary(summary)
    print(f"\nSaved JSON: {out_json}")
    print(f"Saved CSV:  {out_csv}")

    failed = [
        codec
        for codec, values in summary.items()
        if values.get("num_usable", 0) > 0 and not values.get("all_exact_reconstruction", False)
    ]
    return 3 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
