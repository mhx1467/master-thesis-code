from hsi_compression.models import (
    Baseline1DAutoencoder,
    Baseline2DAutoencoder,
    Baseline3DAutoencoder,
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


def build_baseline_3d_ae(in_channels: int, **kwargs):
    return Baseline3DAutoencoder(
        in_channels=in_channels,
        latent_channels=kwargs.get("latent_channels", 16),
        hidden_channels=tuple(kwargs.get("hidden_channels", (32, 64))),
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
    "baseline_2d_ae": build_baseline_2d_ae,
    "baseline_3d_ae": build_baseline_3d_ae,
    "tcn_hsi_ae": build_tcn_hsi_ae,
    "tcn_hsi_ae_v2": build_tcn_hsi_ae_v2,
}


def build_model(model_name: str, in_channels: int, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_name: {model_name}. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](in_channels=in_channels, **kwargs)
