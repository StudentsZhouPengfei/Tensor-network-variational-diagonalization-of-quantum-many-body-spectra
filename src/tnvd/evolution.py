"""Differentiable circuit evolution of a Hamiltonian MPO."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def _split_two_site_operator(
    tensor: torch.Tensor, cutoff: int, *, svd_noise: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``(Dl, bra1, ket1, bra2, ket2, Dr)`` back into two MPO sites."""
    shape = tensor.shape
    matrix = tensor.reshape(shape[0] * shape[1] * shape[2], -1)
    if svd_noise:
        scale = torch.linalg.vector_norm(matrix).detach().clamp_min(1.0)
        noise = torch.randn_like(matrix) * (svd_noise * scale)
        matrix_for_svd = matrix + noise
    else:
        matrix_for_svd = matrix
    u, singular_values, vh = torch.linalg.svd(matrix_for_svd, full_matrices=False)
    rank = min(cutoff, singular_values.numel())
    left = u[:, :rank].reshape(shape[0], shape[1], shape[2], rank)
    right = (singular_values[:rank].to(vh.dtype).diag() @ vh[:rank]).reshape(
        rank, shape[3], shape[4], shape[5]
    )
    return left, right


def conjugate_mpo_pair(
    left: torch.Tensor,
    right: torch.Tensor,
    gate: torch.Tensor,
    cutoff: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ``U^dagger H U`` on two neighbouring physical sites and truncate."""
    # The contraction below maps G -> G H G^dagger.  Feed G=U^dagger so that
    # the returned MPO is in the variational label basis, U^dagger H U.
    effective_gate = gate.reshape(4, 4).conj().transpose(0, 1).reshape_as(gate)
    pair = torch.einsum("qwer,rtyu->qwetyu", left, right)
    ket_evolved = torch.einsum("qwerty,uiwr->queity", pair, effective_gate)
    conjugated = torch.einsum(
        "qwerty,uiet->qwuriy", ket_evolved, effective_gate.conj()
    )
    return _split_two_site_operator(conjugated, cutoff)


def evolve_mpo(
    mpo: Sequence[torch.Tensor],
    unitary_layers: Sequence[Sequence[torch.Tensor]],
    cutoff: int,
) -> list[torch.Tensor]:
    """Evolve an MPO through alternating even/odd nearest-neighbour layers."""
    if cutoff < 1:
        raise ValueError("cutoff must be positive")
    evolved = [tensor.clone() for tensor in mpo]
    for layer_index, layer in enumerate(unitary_layers):
        start = layer_index % 2
        sites = list(range(start, len(evolved) - 1, 2))
        if len(layer) != len(sites):
            raise ValueError("circuit layer does not match the MPO length")
        for site, gate in zip(sites, layer):
            evolved[site], evolved[site + 1] = conjugate_mpo_pair(
                evolved[site], evolved[site + 1], gate, cutoff
            )
    return evolved
