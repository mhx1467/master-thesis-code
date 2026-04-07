from hsi_compression.models.legacy.baseline_3d_ae import Baseline3DAutoencoder


class Hybrid2D3DAutoencoderLIC(Baseline3DAutoencoder):
    """
    Compatibility wrapper for the previous active 3D baseline.

    The model first reduces the spectral axis with 2D convolutions and only then
    applies 3D processing, so it is exposed under a hybrid name in the internal
    benchmark.
    """
