import torch

from hsi_compression.models.registry import build_model


def _build_tiny_sensor_aware_model(in_channels: int = 8):
    return build_model(
        "hierarchical_spectral_mamba_sensor_aware",
        in_channels=in_channels,
        latent_channels=8,
        group_size=2,
        spectral_d_model=8,
        spectral_mlp_hidden_dim=16,
        spectral_out_channels=8,
        num_summary_tokens=2,
        num_local_blocks=0,
        num_global_blocks=0,
        wavelength_embedding_dim=6,
        wavelength_num_frequencies=2,
        spatial_embed_channels=4,
        spatial_context_channels=8,
        spectral_chunk_size=128,
        decoder_band_chunk_size=4,
        spectral_augmentation={"enabled": False},
    )


def test_sensor_aware_hierarchical_mamba_forward_contract():
    model = _build_tiny_sensor_aware_model(in_channels=8)
    x = torch.rand(1, 8, 16, 16)
    mask = torch.ones_like(x, dtype=torch.bool)

    outputs = model(x, valid_mask=mask)

    assert outputs["x_hat"].shape == x.shape
    assert outputs["z"].shape == (1, 8, 4, 4)
    assert outputs["likelihoods"].shape == outputs["z"].shape
    assert outputs["summary_attn"].shape == (64, 2, 4)


def test_sensor_aware_state_dict_is_channel_count_agnostic():
    source = _build_tiny_sensor_aware_model(in_channels=8)
    target = _build_tiny_sensor_aware_model(in_channels=10)

    target.load_state_dict(source.state_dict(), strict=True)

    x = torch.rand(1, 10, 16, 16)
    outputs = target(x)
    assert outputs["x_hat"].shape == x.shape


def test_sensor_aware_compress_decompress_uses_runtime_output_channels():
    model = _build_tiny_sensor_aware_model(in_channels=10)
    model.eval()
    model.update(force=True)
    x = torch.rand(1, 10, 16, 16)

    packed = model.compress(x)
    decoded = model.decompress(
        packed["strings"],
        packed["shape"],
        z_shape=packed["z_shape"],
        output_channels=packed["output_channels"],
    )

    assert packed["output_channels"] == 10
    assert decoded["x_hat"].shape == x.shape
    assert decoded["z_hat"].shape == packed["z_shape"]


def test_sensor_adapter_is_identity_at_initialization():
    model = _build_tiny_sensor_aware_model(in_channels=8)
    x = torch.rand(1, 8, 4, 4)
    wavelengths = model._wavelengths(8, device=x.device, dtype=x.dtype)
    embedding = model.wavelength_embedding(wavelengths)

    adapted = model.sensor_adapter(x, embedding)

    torch.testing.assert_close(adapted, x)
