import torch

from hsi_compression.models.registry import build_model


def _build_tiny_hyperprior_model():
    return build_model(
        "hierarchical_spectral_mamba_hyperprior",
        in_channels=8,
        latent_channels=8,
        group_size=2,
        spectral_d_model=8,
        spectral_mlp_hidden_dim=16,
        spectral_out_channels=8,
        num_summary_tokens=2,
        num_local_blocks=0,
        num_global_blocks=0,
        spatial_embed_channels=4,
        spatial_context_channels=8,
        hyper_channels=8,
        hyper_latent_channels=4,
        spectral_chunk_size=128,
    )


def test_existing_hierarchical_mamba_registry_key_stays_entropy_bottleneck_only():
    model = build_model(
        "hierarchical_spectral_mamba_ae",
        in_channels=8,
        latent_channels=8,
        group_size=2,
        spectral_d_model=8,
        spectral_mlp_hidden_dim=16,
        spectral_out_channels=8,
        num_summary_tokens=2,
        num_local_blocks=0,
        num_global_blocks=0,
        spatial_embed_channels=4,
        spatial_context_channels=8,
        spectral_chunk_size=128,
    )

    assert model.__class__.__name__ == "HierarchicalSpectralMambaAutoencoder"
    assert hasattr(model, "entropy_bottleneck")
    assert not hasattr(model, "gaussian_conditional")


def test_registry_builds_hierarchical_mamba_hyperprior_without_mamba_blocks():
    model = _build_tiny_hyperprior_model()

    assert model.latent_channels == 8
    assert model.hyper_latent_channels == 4
    assert model.proxy_bpppc > 0.0


def test_hierarchical_mamba_hyperprior_forward_reports_combined_likelihoods():
    model = _build_tiny_hyperprior_model()
    x = torch.rand(1, 8, 16, 16)

    outputs = model(x)

    assert outputs["x_hat"].shape == x.shape
    assert outputs["z"].shape == (1, 8, 4, 4)
    assert outputs["hyper_latent"].shape == (1, 4, 1, 1)
    assert outputs["z_likelihoods"].shape == outputs["z"].shape
    assert outputs["hyper_likelihoods"].shape == outputs["hyper_latent"].shape
    assert outputs["likelihoods"].numel() == (
        outputs["z_likelihoods"].numel() + outputs["hyper_likelihoods"].numel()
    )


def test_hierarchical_mamba_hyperprior_compress_decompress_contract():
    model = _build_tiny_hyperprior_model()
    model.eval()
    model.update(force=True)
    x = torch.rand(1, 8, 16, 16)

    packed = model.compress(x)
    decoded = model.decompress(packed["strings"], packed["shape"], z_shape=packed["z_shape"])

    assert isinstance(packed["strings"], list)
    assert len(packed["strings"]) == 2
    assert packed["shape"] == torch.Size([1, 1])
    assert packed["z_shape"] == (1, 8, 4, 4)
    assert decoded["x_hat"].shape == x.shape
    assert decoded["z_hat"].shape == packed["z_shape"]
