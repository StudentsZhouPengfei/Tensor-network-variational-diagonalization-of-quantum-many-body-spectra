"""Tensor-network variational diagonalization."""

from .loss import hilbert_schmidt_residual_squared, logarithmic_hs_loss
from .models import transverse_field_ising_mpo
from .train import TNVDConfig, train_tnvd

__all__ = [
    "TNVDConfig",
    "hilbert_schmidt_residual_squared",
    "logarithmic_hs_loss",
    "train_tnvd",
    "transverse_field_ising_mpo",
]

__version__ = "0.1.0"
