from hsi_compression.models import (
    Baseline1DAutoencoder,
    Baseline1DAutoencoderV2,
    Baseline2DAutoencoder,
    Baseline3DAutoencoder,
    Baseline3DFullBandsAutoencoder,
    SpectralFirstMambaAutoencoder,
    PixelwiseSpectralMambaAutoencoder,
    TCNHSIAutoencoder,
    TCNHSIAutoencoderV2,
    TinyHSIAutoencoder,
)


def build_tiny_ae(in_channels: int, **kwargs):
    return TinyHSIAutoencoder(
        bands=in_channels,
        latent_channels=kwargs.get("latent_channels", 16),
    )


def build_baseline_2d_ae(in_channels: int, **kwargs):
    return Baseline2DAutoencoder(
        in_channels=in_channels,
        hidden_channels=tuple(kwargs.get("hidden_channels", (128, 64))),
        latent_channels=kwargs.get("latent_channels", 16),
    )


def build_baseline_1d_ae(in_channels: int, **kwargs):
    return Baseline1DAutoencoder(
        in_channels=in_channels,
        latent_channels=kwargs.get("latent_channels", 16),
        spectral_hidden_channels=kwargs.get("spectral_hidden_channels", 64),
    )


def build_baseline_1d_ae_v2(in_channels: int, **kwargs):
    return Baseline1DAutoencoderV2(
        in_channels=in_channels,
        latent_channels=kwargs.get("latent_channels", 16),
        spectral_hidden_channels=kwargs.get("spectral_hidden_channels", 64),
        spatial_stem_channels=tuple(kwargs.get("spatial_stem_channels", (64, 128))),
    )


def build_baseline_3d_ae(in_channels: int, **kwargs):
    return Baseline3DAutoencoder(
        in_channels=in_channels,
        latent_channels=kwargs.get("latent_channels", 16),
        hidden_channels=tuple(kwargs.get("hidden_channels", (32, 64))),
        spectral_reduced=kwargs.get("spectral_reduced", 32),
        output_activation=kwargs.get("output_activation", "sigmoid"),
    )


def build_baseline_3d_fullbands_ae(in_channels: int, **kwargs):
    return Baseline3DFullBandsAutoencoder(
        in_channels=in_channels,
        latent_channels=kwargs.get("latent_channels", 4),
        hidden_channels=tuple(kwargs.get("hidden_channels", (8, 16, 32))),
        output_activation=kwargs.get("output_activation", "sigmoid"),
    )


def build_spectral_first_mamba_ae(in_channels: int, **kwargs):
    return SpectralFirstMambaAutoencoder(
        in_channels=in_channels,
        latent_channels=kwargs.get("latent_channels", 8),
        spectral_d_model=kwargs.get("spectral_d_model", 32),
        spectral_out_channels=kwargs.get("spectral_out_channels", 64),
        spatial_embed_channels=kwargs.get("spatial_embed_channels", 16),
        spatial_context_channels=kwargs.get("spatial_context_channels", 64),
        num_spectral_blocks=kwargs.get("num_spectral_blocks", 2),
        mamba_d_state=kwargs.get("mamba_d_state", 16),
        mamba_d_conv=kwargs.get("mamba_d_conv", 4),
        mamba_expand=kwargs.get("mamba_expand", 2),
        use_affine_conditioning=kwargs.get("use_affine_conditioning", False),
        output_activation=kwargs.get("output_activation", "sigmoid"),
        dropout=kwargs.get("dropout", 0.0),
    )



def build_pixelwise_spectral_mamba_ae(in_channels: int, **kwargs):
    return PixelwiseSpectralMambaAutoencoder(
        in_channels=in_channels,
        latent_channels=kwargs.get("latent_channels", 16),
        group_size=kwargs.get("group_size", 4),
        d_model=kwargs.get("d_model", 64),
        mlp_hidden_dim=kwargs.get("mlp_hidden_dim", 256),
        num_mamba_blocks=kwargs.get("num_mamba_blocks", 4),
        mamba_d_state=kwargs.get("mamba_d_state", 16),
        mamba_d_conv=kwargs.get("mamba_d_conv", 4),
        mamba_expand=kwargs.get("mamba_expand", 2),
        pooling=kwargs.get("pooling", "attention"),
        refinement_hidden_channels=kwargs.get("refinement_hidden_channels", 16),
        pixels_per_patch=kwargs.get("pixels_per_patch", 512),
        eval_chunk_size=kwargs.get("eval_chunk_size", 8192),
        output_activation=kwargs.get("output_activation", "sigmoid"),
        dropout=kwargs.get("dropout", 0.0),
    )

def build_tcn_hsi_ae(in_channels: int, **kwargs):
    return TCNHSIAutoencoder(
        in_channels=in_channels,
        encoder_channels=tuple(kwargs.get("encoder_channels", (128, 64))),
        latent_channels=kwargs.get("latent_channels", 8),
        tcn_hidden_channels=kwargs.get("tcn_hidden_channels", 64),
        tcn_num_layers=kwargs.get("tcn_num_layers", 4),
        tcn_kernel_size=kwargs.get("tcn_kernel_size", 3),
        dropout=kwargs.get("dropout", 0.0),
    )


def build_tcn_hsi_ae_v2(in_channels: int, **kwargs):
    return TCNHSIAutoencoderV2(
        in_channels=in_channels,
        spectral_hidden_channels=kwargs.get("spectral_hidden_channels", 32),
        spectral_num_layers=kwargs.get("spectral_num_layers", 4),
        spectral_kernel_size=kwargs.get("spectral_kernel_size", 3),
        encoder_channels=tuple(kwargs.get("encoder_channels", (128, 64))),
        latent_channels=kwargs.get("latent_channels", 8),
        dropout=kwargs.get("dropout", 0.0),
    )


MODEL_REGISTRY = {
    "tiny_ae": build_tiny_ae,
    "baseline_1d_ae": build_baseline_1d_ae,
    "baseline_1d_ae_v2": build_baseline_1d_ae_v2,
    "baseline_2d_ae": build_baseline_2d_ae,
    "baseline_3d_ae": build_baseline_3d_ae,
    "baseline_3d_fullbands_ae": build_baseline_3d_fullbands_ae,
    "spectral_first_mamba_ae": build_spectral_first_mamba_ae,
    "pixelwise_spectral_mamba_ae": build_pixelwise_spectral_mamba_ae,
    "tcn_hsi_ae": build_tcn_hsi_ae,
    "tcn_hsi_ae_v2": build_tcn_hsi_ae_v2,
}


def build_model(model_name: str, in_channels: int, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_name: {model_name}. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](in_channels=in_channels, **kwargs)
