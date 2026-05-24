def build_baseline_2d_ae(in_channels: int, **kwargs):
    from hsi_compression.models.baseline_2d_ae import Baseline2DAutoencoder

    return Baseline2DAutoencoder(
        in_channels=in_channels,
        hidden_channels=tuple(kwargs.get("hidden_channels", (128, 64))),
        latent_channels=kwargs.get("latent_channels", 16),
        output_activation=kwargs.get("output_activation", "sigmoid"),
    )


def build_baseline_1d_pixel_ae(in_channels: int, **kwargs):
    from hsi_compression.models.baseline_1d_pixel_ae import Baseline1DPixelAutoencoder

    return Baseline1DPixelAutoencoder(
        in_channels=in_channels,
        latent_channels=kwargs.get("latent_channels", 16),
        hidden_channels=kwargs.get("hidden_channels", 64),
        output_activation=kwargs.get("output_activation", "sigmoid"),
    )


def build_baseline_3d_patch_ae(in_channels: int, **kwargs):
    from hsi_compression.models.baseline_3d_patch_ae import Baseline3DPatchAutoencoder

    return Baseline3DPatchAutoencoder(
        in_channels=in_channels,
        latent_channels=kwargs.get("latent_channels", 16),
        hidden_channels=tuple(kwargs.get("hidden_channels", (32, 64))),
        output_activation=kwargs.get("output_activation", "sigmoid"),
    )


def build_hybrid_2d3d_ae_lic(in_channels: int, **kwargs):
    from hsi_compression.models.hybrid_2d3d_ae_lic import Hybrid2D3DAutoencoderLIC

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
    from hsi_compression.models.spectral_first_mamba_ae_v2 import (
        SpectralFirstMambaAutoencoderV2,
    )

    return SpectralFirstMambaAutoencoderV2(
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
    from hsi_compression.models.hierarchical_spectral_mamba_ae import (
        HierarchicalSpectralMambaAutoencoder,
    )

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
        use_spatial_conditioning=kwargs.get("use_spatial_conditioning", True),
        use_affine_conditioning=kwargs.get("use_affine_conditioning", True),
        spectral_chunk_size=kwargs.get("spectral_chunk_size", 512),
        output_activation=kwargs.get("output_activation", "sigmoid"),
        dropout=kwargs.get("dropout", 0.0),
    )


def build_hierarchical_spectral_mamba_hyperprior(in_channels: int, **kwargs):
    from hsi_compression.models.hierarchical_spectral_mamba_hyperprior import (
        HierarchicalSpectralMambaHyperpriorAutoencoder,
    )

    return HierarchicalSpectralMambaHyperpriorAutoencoder(
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
        use_spatial_conditioning=kwargs.get("use_spatial_conditioning", True),
        use_affine_conditioning=kwargs.get("use_affine_conditioning", True),
        spectral_chunk_size=kwargs.get("spectral_chunk_size", 512),
        output_activation=kwargs.get("output_activation", "sigmoid"),
        dropout=kwargs.get("dropout", 0.0),
        hyper_channels=kwargs.get("hyper_channels", 96),
        hyper_latent_channels=kwargs.get("hyper_latent_channels", 64),
        scale_bound=kwargs.get("scale_bound", 0.11),
        scale_table_min=kwargs.get("scale_table_min", 0.11),
        scale_table_max=kwargs.get("scale_table_max", 256.0),
        scale_table_levels=kwargs.get("scale_table_levels", 64),
    )


def build_spectral_tcn_lossless(in_channels: int, **kwargs):
    from hsi_compression.models.spectral_tcn_lossless import SpectralTCNLossless

    return SpectralTCNLossless(
        in_channels=in_channels,
        hidden_channels=kwargs.get("hidden_channels", 48),
        num_blocks=kwargs.get("num_blocks", 6),
        kernel_size=kwargs.get("kernel_size", 3),
        dropout=kwargs.get("dropout", 0.0),
        output_activation=kwargs.get("output_activation", "sigmoid"),
        symbol_scale=kwargs.get("symbol_scale", 10000),
        zlib_level=kwargs.get("zlib_level", 9),
        raw_fallback=kwargs.get("raw_fallback", True),
        pixels_per_patch=kwargs.get("pixels_per_patch"),
        prediction_mode=kwargs.get("prediction_mode", "value"),
        residual_backend=kwargs.get("residual_backend", "zlib"),
        residual_transform=kwargs.get("residual_transform", "none"),
    )


def build_spectral_tcn_delta_lossless(in_channels: int, **kwargs):
    # delta mode predicts spectral differences, so tanh is the natural bounded output
    kwargs = {
        **kwargs,
        "prediction_mode": "delta",
        "output_activation": kwargs.get("output_activation", "tanh"),
    }
    return build_spectral_tcn_lossless(in_channels=in_channels, **kwargs)


MODEL_REGISTRY = {
    "baseline_1d_pixel_ae": build_baseline_1d_pixel_ae,
    "baseline_2d_ae": build_baseline_2d_ae,
    "baseline_2d_patch_ae": build_baseline_2d_patch_ae,
    "baseline_2d_patch_ae_lic": build_baseline_2d_patch_ae_lic,
    "baseline_3d_patch_ae": build_baseline_3d_patch_ae,
    "hierarchical_spectral_mamba_ae": build_hierarchical_spectral_mamba_ae,
    "hierarchical_spectral_mamba_hyperprior": build_hierarchical_spectral_mamba_hyperprior,
    "hybrid_2d3d_ae_lic": build_hybrid_2d3d_ae_lic,
    "spectral_tcn_delta_lossless": build_spectral_tcn_delta_lossless,
    "spectral_tcn_lossless": build_spectral_tcn_lossless,
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
        # legacy models stay importable historically but are blocked from active benchmark configs
        raise ValueError(
            f"Model '{model_name}' has been moved to legacy and is no longer supported "
            f"in the active benchmark pipeline."
        )
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_name: {model_name}. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](in_channels=in_channels, **kwargs)
