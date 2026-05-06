from .baseline_1d_pixel_ae import Baseline1DPixelAutoencoder
from .baseline_2d_ae import Baseline2DAutoencoder
from .baseline_3d_patch_ae import Baseline3DPatchAutoencoder
from .hierarchical_spectral_mamba_ae import HierarchicalSpectralMambaAutoencoder
from .hybrid_2d3d_ae_lic import Hybrid2D3DAutoencoderLIC
from .spectral_first_mamba_ae_v2 import SpectralFirstMambaAutoencoderV2
from .spectral_tcn_lossless import SpectralTCNLossless

SpectralMambaAutoencoder = SpectralFirstMambaAutoencoderV2

__all__ = [
    "Baseline1DPixelAutoencoder",
    "Baseline2DAutoencoder",
    "Baseline3DPatchAutoencoder",
    "HierarchicalSpectralMambaAutoencoder",
    "Hybrid2D3DAutoencoderLIC",
    "SpectralTCNLossless",
    "SpectralMambaAutoencoder",
    "SpectralFirstMambaAutoencoderV2",
]
