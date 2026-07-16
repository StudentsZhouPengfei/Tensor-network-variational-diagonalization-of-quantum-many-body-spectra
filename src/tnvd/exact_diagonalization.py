"""Small-system exact references distilled from the original ED scripts."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np
import torch as tc

from .mpo_factory import mpo_to_dense, transverse_field_ising_mpo


def diagonalize_tfim(
    num_spins: int,
    field_x: float = 0.5,
    coupling: float = 1.0,
    longitudinal_fields: Sequence[float] | None = None,
):
    """Return all TFIM eigenvalues/eigenvectors for a small open chain on CPU."""
    hamiltonian = mpo_to_dense(
        transverse_field_ising_mpo(
            num_spins,
            field_x=field_x,
            coupling=coupling,
            longitudinal_fields=longitudinal_fields,
        )
    )
    return tc.linalg.eigh(hamiltonian)


def fixed_magnetization_basis(num_spins: int, num_down: int) -> list[int]:
    """Return computational-basis integers with exactly ``num_down`` set bits."""
    if not 0 <= num_down <= num_spins:
        raise ValueError("num_down must lie between zero and num_spins")
    basis = []
    for occupied in combinations(range(num_spins), num_down):
        state = 0
        for site in occupied:
            state |= 1 << (num_spins - 1 - site)
        basis.append(state)
    return basis


def xxz_sector_hamiltonian(
    num_spins: int,
    num_down: int,
    longitudinal_fields: Sequence[float] | None = None,
    coupling_xy: float = 1.0,
    coupling_z: float = 1.0,
):
    """Construct one conserved-magnetization block of the open random-field XXZ chain."""
    fields = [0.0] * num_spins if longitudinal_fields is None else list(longitudinal_fields)
    if len(fields) != num_spins:
        raise ValueError("longitudinal_fields must contain one value per spin")
    basis = fixed_magnetization_basis(num_spins, num_down)
    index = {state: position for position, state in enumerate(basis)}
    block = np.zeros((len(basis), len(basis)), dtype=np.float64)

    for row, state in enumerate(basis):
        spins = np.array(
            [0.5 if not state & (1 << (num_spins - 1 - site)) else -0.5 for site in range(num_spins)]
        )
        block[row, row] = np.dot(fields, spins) + coupling_z * np.dot(spins[:-1], spins[1:])
        for site in range(num_spins - 1):
            left = 1 << (num_spins - 1 - site)
            right = 1 << (num_spins - 2 - site)
            if bool(state & left) != bool(state & right):
                block[row, index[state ^ left ^ right]] += 0.5 * coupling_xy
    return tc.from_numpy(block), basis


def diagonalize_xxz_sector(*args, **kwargs):
    """Return one XXZ-sector eigensystem and its computational basis."""
    block, basis = xxz_sector_hamiltonian(*args, **kwargs)
    eigenvalues, eigenvectors = tc.linalg.eigh(block)
    return eigenvalues, eigenvectors, basis


def adjacent_gap_ratio(eigenvalues, central_fraction: float = 0.5) -> float:
    """Mean adjacent-gap ratio after restricting to the central spectral fraction."""
    values = tc.as_tensor(eigenvalues, dtype=tc.float64).sort().values
    if not 0.0 < central_fraction <= 1.0:
        raise ValueError("central_fraction must lie in (0, 1]")
    keep = max(3, int(values.numel() * central_fraction))
    start = (values.numel() - keep) // 2
    gaps = tc.diff(values[start : start + keep])
    if gaps.numel() < 2:
        raise ValueError("at least three eigenvalues are required")
    ratios = tc.minimum(gaps[:-1], gaps[1:]) / tc.maximum(gaps[:-1], gaps[1:]).clamp_min(
        tc.finfo(gaps.dtype).tiny
    )
    return float(ratios.mean())
