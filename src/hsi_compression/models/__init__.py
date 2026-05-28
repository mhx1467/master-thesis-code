from __future__ import annotations

__all__ = [
    "Baseline1DPixelAutoencoder",
    "Baseline2DAutoencoder",
    "Baseline3DPatchAutoencoder",
    "HierarchicalSpectralMambaAutoencoder",
    "HierarchicalSpectralMambaSensorAwareAutoencoder",
    "Hybrid2D3DAutoencoderLIC",
    "SpectralTCNLossless",
    "SpectralMambaAutoencoder",
    "SpectralFirstMambaAutoencoderV2",
]


def __getattr__(name: str):
    if name == "Baseline1DPixelAutoencoder":
        from .baseline_1d_pixel_ae import Baseline1DPixelAutoencoder

        return Baseline1DPixelAutoencoder
    if name == "Baseline2DAutoencoder":
        from .baseline_2d_ae import Baseline2DAutoencoder

        return Baseline2DAutoencoder
    if name == "Baseline3DPatchAutoencoder":
        from .baseline_3d_patch_ae import Baseline3DPatchAutoencoder

        return Baseline3DPatchAutoencoder
    if name == "HierarchicalSpectralMambaAutoencoder":
        from .hierarchical_spectral_mamba_ae import HierarchicalSpectralMambaAutoencoder

        return HierarchicalSpectralMambaAutoencoder
    if name == "HierarchicalSpectralMambaSensorAwareAutoencoder":
        from .hierarchical_spectral_mamba_sensor_aware import (
            HierarchicalSpectralMambaSensorAwareAutoencoder,
        )

        return HierarchicalSpectralMambaSensorAwareAutoencoder
    if name == "Hybrid2D3DAutoencoderLIC":
        from .hybrid_2d3d_ae_lic import Hybrid2D3DAutoencoderLIC

        return Hybrid2D3DAutoencoderLIC
    if name in {"SpectralMambaAutoencoder", "SpectralFirstMambaAutoencoderV2"}:
        from .spectral_first_mamba_ae_v2 import SpectralFirstMambaAutoencoderV2

        return SpectralFirstMambaAutoencoderV2
    if name == "SpectralTCNLossless":
        from .spectral_tcn_lossless import SpectralTCNLossless

        return SpectralTCNLossless
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
