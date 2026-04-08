from hsi_compression.models import (
    Baseline1DPixelAutoencoder,
    Baseline2DAutoencoder,
    Baseline3DPatchAutoencoder,
    HierarchicalSpectralMambaAutoencoder,
    Hybrid2D3DAutoencoderLIC,
    SpectralMambaAutoencoder,
    SSCNet,
)


def build_baseline_2d_ae(in_channels: int, **kwargs):
    return Baseline2DAutoencoder(
        in_channels=in_channels,
        hidden_channels=tuple(kwargs.get("hidden_channels", (128, 64))),
        latent_channels=kwargs.get("latent_channels", 16),
        output_activation=kwargs.get("output_activation", "sigmoid"),
    )


def build_baseline_1d_pixel_ae(in_channels: int, **kwargs):
    return Baseline1DPixelAutoencoder(
        in_channels=in_channels,
        latent_channels=kwargs.get("latent_channels", 16),
        hidden_channels=kwargs.get("hidden_channels", 64),
        output_activation=kwargs.get("output_activation", "sigmoid"),
    )


def build_baseline_3d_patch_ae(in_channels: int, **kwargs):
    return Baseline3DPatchAutoencoder(
        in_channels=in_channels,
        latent_channels=kwargs.get("latent_channels", 16),
        hidden_channels=tuple(kwargs.get("hidden_channels", (32, 64))),
        output_activation=kwargs.get("output_activation", "sigmoid"),
    )


def build_hybrid_2d3d_ae_lic(in_channels: int, **kwargs):
    return Hybrid2D3DAutoencoderLIC(
        in_channels=in_channels,
        latent_channels=kwargs.get("latent_channels", 16),
        hidden_channels=tuple(kwargs.get("hidden_channels", (32, 64))),
        spectral_reduced=kwargs.get("spectral_reduced", 32),
        output_activation=kwargs.get("output_activation", "sigmoid"),
    )


def build_baseline_2d_patch_ae(in_channels: int, **kwargs):
    return build_baseline_2d_ae(in_channels=in_channels, **kwargs)


def build_baseline_2d_patch_ae_lic(in_channels: int, **kwargs):
    return build_baseline_2d_ae(in_channels=in_channels, **kwargs)


def build_spectral_mamba_ae(in_channels: int, **kwargs):
    return SpectralMambaAutoencoder(
        in_channels=in_channels,
        latent_channels=kwargs.get("latent_channels", 96),
        group_size=kwargs.get("group_size", 1),
        spectral_d_model=kwargs.get("spectral_d_model", 64),
        spectral_mlp_hidden_dim=kwargs.get("spectral_mlp_hidden_dim", 128),
        spectral_out_channels=kwargs.get("spectral_out_channels", 96),
        spatial_embed_channels=kwargs.get("spatial_embed_channels", 16),
        spatial_context_channels=kwargs.get("spatial_context_channels", 64),
        num_spectral_blocks=kwargs.get("num_spectral_blocks", 3),
        mamba_d_state=kwargs.get("mamba_d_state", 16),
        mamba_d_conv=kwargs.get("mamba_d_conv", 4),
        mamba_expand=kwargs.get("mamba_expand", 2),
        pooling=kwargs.get("pooling", "attention"),
        use_affine_conditioning=kwargs.get("use_affine_conditioning", True),
        refinement_depth=kwargs.get("refinement_depth", 3),
        refinement_hidden_channels=kwargs.get("refinement_hidden_channels", 16),
        output_activation=kwargs.get("output_activation", "sigmoid"),
        dropout=kwargs.get("dropout", 0.0),
    )


def build_hierarchical_spectral_mamba_ae(in_channels: int, **kwargs):
    return HierarchicalSpectralMambaAutoencoder(
        in_channels=in_channels,
        latent_channels=kwargs.get("latent_channels", 96),
        group_size=kwargs.get("group_size", 4),
        spectral_d_model=kwargs.get("spectral_d_model", 64),
        spectral_mlp_hidden_dim=kwargs.get("spectral_mlp_hidden_dim", 128),
        spectral_out_channels=kwargs.get("spectral_out_channels", 96),
        num_summary_tokens=kwargs.get("num_summary_tokens", 4),
        num_local_blocks=kwargs.get("num_local_blocks", 2),
        num_global_blocks=kwargs.get("num_global_blocks", 1),
        spatial_embed_channels=kwargs.get("spatial_embed_channels", 16),
        spatial_context_channels=kwargs.get("spatial_context_channels", 64),
        mamba_d_state=kwargs.get("mamba_d_state", 16),
        mamba_d_conv=kwargs.get("mamba_d_conv", 4),
        mamba_expand=kwargs.get("mamba_expand", 2),
        use_affine_conditioning=kwargs.get("use_affine_conditioning", True),
        spectral_chunk_size=kwargs.get("spectral_chunk_size", 512),
        output_activation=kwargs.get("output_activation", "sigmoid"),
        dropout=kwargs.get("dropout", 0.0),
    )


def build_sscnet(in_channels: int, **kwargs):
    return SSCNet(
        in_channels=in_channels,
        latent_channels=kwargs.get("latent_channels", 1024),
    )


MODEL_REGISTRY = {
    "baseline_1d_pixel_ae": build_baseline_1d_pixel_ae,
    "baseline_2d_ae": build_baseline_2d_ae,
    "baseline_2d_patch_ae": build_baseline_2d_patch_ae,
    "baseline_2d_patch_ae_lic": build_baseline_2d_patch_ae_lic,
    "baseline_3d_patch_ae": build_baseline_3d_patch_ae,
    "hierarchical_spectral_mamba_ae": build_hierarchical_spectral_mamba_ae,
    "hybrid_2d3d_ae_lic": build_hybrid_2d3d_ae_lic,
    "sscnet": build_sscnet,
    "spectral_mamba_ae": build_spectral_mamba_ae,
    "spectral_first_mamba_ae_v2": build_spectral_mamba_ae,
}

LEGACY_MODELS = {
    "tiny_ae",
    "baseline_1d_ae",
    "baseline_1d_ae_v2",
    "baseline_3d_ae",
    "baseline_3d_fullbands_ae",
    "pixelwise_mamba_ae",
    "pixelwise_spectral_mamba_ae",
    "spectral_first_mamba_ae",
}


def build_model(model_name: str, in_channels: int, **kwargs):
    if model_name in LEGACY_MODELS:
        raise ValueError(
            f"Model '{model_name}' has been moved to legacy and is no longer supported "
            f"in the active benchmark pipeline."
        )
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_name: {model_name}. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](in_channels=in_channels, **kwargs)
