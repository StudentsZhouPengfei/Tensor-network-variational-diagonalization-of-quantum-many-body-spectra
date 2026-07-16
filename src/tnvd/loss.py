"""Tensor contractions for the TNVD objective."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def mpo_inner(up: Sequence[torch.Tensor], down: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return ``Tr(up @ down^dagger)`` for two MPOs with matching physical legs."""
    if len(up) != len(down) or not up:
        raise ValueError("MPOs must be non-empty and have the same length")
    environment = torch.einsum("astb,astd->bd", up[0], down[0].conj())
    for site in range(1, len(up)):
        if site < len(up) - 1:
            environment = torch.einsum(
                "ac,astb,cstd->bd", environment, up[site], down[site].conj()
            )
        else:
            environment = torch.einsum(
                "ac,astb,cstb->", environment, up[site], down[site].conj()
            )
    return environment


def spectrum_mps_norm(mps: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return the squared Euclidean norm of the spectrum MPS."""
    environment = torch.einsum("asb,asd->bd", mps[0], mps[0].conj())
    for site in range(1, len(mps)):
        if site < len(mps) - 1:
            environment = torch.einsum(
                "ac,asb,csd->bd", environment, mps[site], mps[site].conj()
            )
        else:
            environment = torch.einsum(
                "ac,asb,csb->", environment, mps[site], mps[site].conj()
            )
    return environment


def diagonal_mpo_mps_overlap(
    spectrum_mps: Sequence[torch.Tensor], evolved_mpo: Sequence[torch.Tensor]
) -> torch.Tensor:
    r"""Contract ``sum_r E_r <r|U^dagger H U|r>``."""
    if len(spectrum_mps) != len(evolved_mpo):
        raise ValueError("MPS and MPO lengths must match")
    contracted_sites = [
        torch.einsum("dbbe,abc->adce", mpo_site, mps_site)
        for mps_site, mpo_site in zip(spectrum_mps, evolved_mpo)
    ]
    environment = contracted_sites[0]
    for site in contracted_sites[1:]:
        environment = torch.einsum("qwer,erty->qwty", environment, site)
    return torch.einsum("abab->", environment)


def hilbert_schmidt_residual_squared(
    spectrum_mps: Sequence[torch.Tensor],
    evolved_mpo: Sequence[torch.Tensor],
    hamiltonian_norm_squared: torch.Tensor,
) -> torch.Tensor:
    r"""Compute ``||H-H_tilde||_HS^2`` without a square root."""
    mps_complex = [
        tensor if tensor.is_complex() else torch.complex(tensor, torch.zeros_like(tensor))
        for tensor in spectrum_mps
    ]
    ansatz_norm = spectrum_mps_norm(mps_complex)
    cross = diagonal_mpo_mps_overlap(mps_complex, evolved_mpo)
    residual = hamiltonian_norm_squared - cross - cross.conj() + ansatz_norm
    return residual.real


def logarithmic_hs_loss(
    residual_squared: torch.Tensor, num_spins: int, *, epsilon=None
) -> torch.Tensor:
    r"""Paper objective ``F = log2(||H-H_tilde||_HS^2) - N``."""
    if epsilon is None:
        epsilon = torch.finfo(residual_squared.dtype).tiny
    return torch.log2(residual_squared.clamp_min(epsilon)) - num_spins
