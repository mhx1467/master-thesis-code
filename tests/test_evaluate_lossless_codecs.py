from scripts.evaluate_lossless_codecs import DEFAULT_CODECS, TCN_CODECS


def test_tcn_codec_registry_covers_bitplane_tcn_residual_codec():
    assert "bitplane_tcn_residual_zstd" in DEFAULT_CODECS
    assert "bitplane_tcn_residual_zstd" in TCN_CODECS
