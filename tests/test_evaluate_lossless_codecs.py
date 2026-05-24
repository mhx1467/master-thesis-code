import pytest
import torch

from hsi_compression.models.registry import build_model
from scripts.evaluate_lossless_codecs import DEFAULT_CODECS, TCN_CODECS, run_tcn_residual_codec


def test_tcn_codec_registry_covers_bitplane_tcn_residual_codec():
    assert "bitplane_tcn_residual_zstd" in DEFAULT_CODECS
    assert "bitplane_tcn_residual_zstd" in TCN_CODECS


def test_tcn_residual_codec_uses_public_bitplane_zstd_path():
    pytest.importorskip("zstandard")
    model = build_model(
        "spectral_tcn_delta_lossless",
        in_channels=8,
        hidden_channels=8,
        num_blocks=2,
        raw_fallback=False,
    )
    x_int = torch.randint(0, 10001, (1, 8, 2, 2), dtype=torch.int32)
    x = x_int.to(torch.float32) / 10000.0

    result = run_tcn_residual_codec(
        codec="bitplane_tcn_residual_zstd",
        model=model,
        x=x,
        device=torch.device("cpu"),
        level=3,
        original_bits_per_channel=16.0,
        residual_backend="zstd",
        residual_transform="zigzag+bitplane",
    )

    assert result.skipped is False
    assert result.codec_backend == "bitplane_tcn_residual_zstd"
    assert result.exact_reconstruction is True
    assert result.mismatch_count == 0
