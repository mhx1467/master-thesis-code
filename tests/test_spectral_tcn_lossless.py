import torch

from hsi_compression.models.registry import build_model


def test_registry_builds_spectral_tcn_lossless():
    model = build_model("spectral_tcn_lossless", in_channels=8, hidden_channels=16, num_blocks=3)
    assert model.compression_mode == "lossless"
    assert model.supports_actual_compression is True


def test_spectral_tcn_lossless_exact_roundtrip_on_symbol_grid():
    model = build_model("spectral_tcn_lossless", in_channels=8, hidden_channels=8, num_blocks=2)
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
