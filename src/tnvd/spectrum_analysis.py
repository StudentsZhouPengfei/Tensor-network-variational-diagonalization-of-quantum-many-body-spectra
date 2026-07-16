"""Small reusable spectrum-state analysis functions."""

from __future__ import annotations

import numpy as np


def virtual_entropy_curve(values: np.ndarray) -> np.ndarray:
    """Return binary-cut virtual entropies for a power-of-two energy vector."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    num_spins = int(np.log2(values.size))
    if 2**num_spins != values.size:
        raise ValueError("energy-vector length must be a power of two")
    norm = np.linalg.norm(values)
    if norm == 0:
        raise ValueError("energy vector must not be identically zero")
    normalized = values / norm
    entropies = []
    for cut in range(1, num_spins):
        singular_values = np.linalg.svd(normalized.reshape(2**cut, -1), compute_uv=False)
        probabilities = singular_values**2
        probabilities = probabilities[probabilities > 0]
        entropies.append(float(-np.sum(probabilities * np.log2(probabilities))))
    return np.asarray(entropies)
