# legacy models that are no longer part of the active rd benchmark.
# these models are kept for backwards compatibility with older checkpoints.

from .baseline_1d_ae_v2 import Baseline1DAutoencoderV2
from .baseline_3d_ae import Baseline3DAutoencoder

__all__ = [
    "Baseline1DAutoencoderV2",
    "Baseline3DAutoencoder",
]
