from hsi_compression.models import (
    TinyHSIAutoencoder,
    Baseline2DAutoencoder,
    TCNHSIAutoencoder,
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


MODEL_REGISTRY = {
    "tiny_ae": build_tiny_ae,
    "baseline_2d_ae": build_baseline_2d_ae,
    "tcn_hsi_ae": build_tcn_hsi_ae,
}


def build_model(model_name: str, in_channels: int, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_name: {model_name}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](in_channels=in_channels, **kwargs)