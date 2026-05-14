import pytest
import torch

from hsi_compression.models.registry import build_model


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


def test_spectral_tcn_lossless_raw_float_fallback_is_exact():
    model = build_model("spectral_tcn_lossless", in_channels=6, hidden_channels=8, num_blocks=2)
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

    expected_delta_target = torch.tensor(
        [[[[0.0100]], [[0.0025]], [[-0.0010]], [[0.0025]]]],
        dtype=torch.float32,
    )
    assert torch.allclose(outputs["x_target"], expected_delta_target)
