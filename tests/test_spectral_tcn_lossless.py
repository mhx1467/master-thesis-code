import json
import struct

import pytest
import torch

from hsi_compression.models.registry import build_model


def _read_payload_header(strings: bytes) -> dict:
    header_len = struct.unpack("<I", strings[:4])[0]
    header_start = 4
    header_end = header_start + header_len
    return json.loads(strings[header_start:header_end].decode("utf-8"))


def test_registry_builds_spectral_tcn_lossless():
    model = build_model("spectral_tcn_lossless", in_channels=8, hidden_channels=16, num_blocks=3)
    assert model.compression_mode == "lossless"
    assert model.supports_actual_compression is True


def test_registry_builds_spectral_tcn_delta_lossless():
    model = build_model(
        "spectral_tcn_delta_lossless",
        in_channels=8,
        hidden_channels=16,
        num_blocks=3,
    )
    assert model.compression_mode == "lossless"
    assert model.supports_actual_compression is True
    assert model.prediction_mode == "delta"


def test_spectral_tcn_lossless_exact_roundtrip_on_symbol_grid():
    model = build_model("spectral_tcn_lossless", in_channels=8, hidden_channels=8, num_blocks=2)
    # symbol-grid values should survive compress and decompress exactly.
    x_int = torch.randint(0, 10001, (1, 8, 3, 3), dtype=torch.int32)
    x = x_int.to(torch.float32) / 10000.0

    packed = model.compress(x)
    decoded = model.decompress(packed["strings"], packed["shape"])

    assert torch.equal(decoded["x_hat"], x)


def test_spectral_tcn_delta_lossless_exact_roundtrip_on_symbol_grid():
    model = build_model(
        "spectral_tcn_delta_lossless",
        in_channels=8,
        hidden_channels=8,
        num_blocks=2,
    )
    x_int = torch.randint(0, 10001, (1, 8, 3, 3), dtype=torch.int32)
    x = x_int.to(torch.float32) / 10000.0

    packed = model.compress(x)
    decoded = model.decompress(packed["strings"], packed["shape"])

    assert torch.equal(decoded["x_hat"], x)


def test_spectral_tcn_bitplane_zstd_roundtrip_on_symbol_grid():
    pytest.importorskip("zstandard")
    model = build_model(
        "spectral_tcn_delta_lossless",
        in_channels=8,
        hidden_channels=8,
        num_blocks=2,
        raw_fallback=False,
        residual_backend="zstd",
        residual_transform="zigzag+bitplane",
    )
    x_int = torch.randint(0, 10001, (1, 8, 3, 3), dtype=torch.int32)
    x = x_int.to(torch.float32) / 10000.0

    packed = model.compress(x)
    decoded = model.decompress(packed["strings"], packed["shape"])
    header = _read_payload_header(packed["strings"])

    assert header["codec_backend"] == "bitplane_tcn_residual_zstd"
    assert header["entropy_backend"] == "zstd"
    assert header["prediction_mode"] == "delta"
    assert header["transform"] == "zigzag+bitplane"
    assert torch.equal(decoded["x_hat"], x)


def test_spectral_tcn_lossless_raw_float_fallback_is_exact():
    model = build_model("spectral_tcn_lossless", in_channels=6, hidden_channels=8, num_blocks=2)
    # this value is intentionally not aligned to the 1/10000 symbol grid.
    x = torch.rand(1, 6, 2, 2, dtype=torch.float32)
    x[0, 0, 0, 0] = 0.12345679

    packed = model.compress(x)
    decoded = model.decompress(packed["strings"], packed["shape"])

    assert torch.equal(decoded["x_hat"], x)


def test_spectral_tcn_lossless_requires_symbol_grid_without_raw_fallback():
    model = build_model(
        "spectral_tcn_lossless",
        in_channels=6,
        hidden_channels=8,
        num_blocks=2,
        raw_fallback=False,
    )
    x = torch.rand(1, 6, 2, 2, dtype=torch.float32)
    # without fallback, non-grid floats must be rejected instead of silently becoming lossy.
    x[0, 0, 0, 0] = 0.12345679

    with pytest.raises(ValueError, match="not exactly representable"):
        model.compress(x)


def test_spectral_tcn_lossless_accepts_near_integer_symbol_grid():
    model = build_model(
        "spectral_tcn_lossless",
        in_channels=6,
        hidden_channels=8,
        num_blocks=2,
        raw_fallback=False,
    )
    x_int = torch.randint(0, 10001, (1, 6, 2, 2), dtype=torch.int32)
    x = x_int.to(torch.float32) / 10000.0
    x = x + 5e-8
    canonical = x_int.to(torch.float32) / 10000.0

    packed = model.compress(x)
    decoded = model.decompress(packed["strings"], packed["shape"])

    assert torch.equal(decoded["x_hat"], canonical)


def test_spectral_tcn_lossless_training_forward_can_sample_pixels():
    model = build_model(
        "spectral_tcn_lossless",
        in_channels=6,
        hidden_channels=8,
        num_blocks=2,
        pixels_per_patch=4,
    )
    model.train()
    # sampled training returns a smaller pseudo-image of selected pixel spectra.
    x_int = torch.randint(0, 10001, (2, 6, 4, 4), dtype=torch.int32)
    x = x_int.to(torch.float32) / 10000.0
    mask = torch.ones_like(x, dtype=torch.bool)

    outputs = model(x, valid_mask=mask)

    assert outputs["x_hat"].shape == (2, 6, 2, 2)
    assert outputs["x_target"].shape == (2, 6, 2, 2)
    assert outputs["mask_for_loss"].shape == (2, 6, 2, 2)


def test_spectral_tcn_delta_lossless_training_target_is_delta_domain():
    model = build_model(
        "spectral_tcn_delta_lossless",
        in_channels=4,
        hidden_channels=8,
        num_blocks=2,
    )
    x_int = torch.tensor([[[[100]], [[125]], [[115]], [[140]]]], dtype=torch.int32)
    x = x_int.to(torch.float32) / 10000.0

    outputs = model(x)

    # first channel is absolute value, later channels are adjacent-band deltas.
    expected_delta_target = torch.tensor(
        [[[[0.0100]], [[0.0025]], [[-0.0010]], [[0.0025]]]],
        dtype=torch.float32,
    )
    assert torch.allclose(outputs["x_target"], expected_delta_target)


def test_parallel_value_residuals_match_sequential_predictor():
    model = build_model("spectral_tcn_lossless", in_channels=8, hidden_channels=8, num_blocks=2)
    x_int = torch.randint(0, 10001, (1, 8, 3, 3), dtype=torch.int32)

    parallel_residuals = model._residuals_from_symbols(x_int)  # noqa: SLF001
    sequential_predictions = model._predict_symbols_sequential_from_symbols(x_int)  # noqa: SLF001
    sequential_residuals = (x_int - sequential_predictions).to(torch.int32)

    assert torch.equal(parallel_residuals, sequential_residuals)


def test_parallel_delta_residuals_match_sequential_predictor():
    model = build_model(
        "spectral_tcn_delta_lossless",
        in_channels=8,
        hidden_channels=8,
        num_blocks=2,
    )
    x_int = torch.randint(0, 10001, (1, 8, 3, 3), dtype=torch.int32)

    parallel_residuals = model._residuals_from_symbols(x_int)  # noqa: SLF001
    deltas = model._symbols_to_deltas(x_int)  # noqa: SLF001
    sequential_predictions = model._predict_deltas_sequential_from_symbols(x_int)  # noqa: SLF001
    sequential_residuals = (deltas - sequential_predictions).to(torch.int32)

    assert torch.equal(parallel_residuals, sequential_residuals)
