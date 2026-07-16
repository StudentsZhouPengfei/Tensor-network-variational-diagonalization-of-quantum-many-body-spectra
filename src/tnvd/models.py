"""Hamiltonian builders used by the reference implementation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

import torch


def spin_half_operators(
    *, device: Union[torch.device, str] = "cpu", dtype: torch.dtype = torch.complex128
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(identity, Sx, Sy, Sz)`` for a spin-1/2 site."""
    identity = torch.eye(2, dtype=dtype, device=device)
    sx = torch.tensor([[0.0, 0.5], [0.5, 0.0]], dtype=dtype, device=device)
    sy = torch.tensor([[0.0, -0.5j], [0.5j, 0.0]], dtype=dtype, device=device)
    sz = torch.tensor([[0.5, 0.0], [0.0, -0.5]], dtype=dtype, device=device)
    return identity, sx, sy, sz


def transverse_field_ising_mpo(
    num_spins: int,
    field_x: float = 0.5,
    coupling: float = 1.0,
    longitudinal_fields: Optional[Sequence[float]] = None,
    *,
    device: Union[torch.device, str] = "cpu",
    dtype: torch.dtype = torch.complex128,
) -> list[torch.Tensor]:
    r"""Build the open-chain MPO

    .. math:: H=-J\sum_n S_n^zS_{n+1}^z-h_x\sum_n S_n^x+\sum_n w_nS_n^z.

    Site tensors use index order ``(left bond, bra, ket, right bond)``.
    """
    if num_spins < 2:
        raise ValueError("num_spins must be at least 2")
    if longitudinal_fields is None:
        fields = [0.0] * num_spins
    else:
        fields = [float(value) for value in longitudinal_fields]
        if len(fields) != num_spins:
            raise ValueError("longitudinal_fields must contain one value per spin")

    identity, sx, _, sz = spin_half_operators(device=device, dtype=dtype)
    local = [-field_x * sx + field * sz for field in fields]
    zero = torch.zeros_like(identity)

    first = torch.stack((local[0], -coupling * sz, identity), dim=-1).unsqueeze(0)
    mpo = [first]
    for site in range(1, num_spins - 1):
        rows = (
            (identity, zero, zero),
            (sz, zero, zero),
            (local[site], -coupling * sz, identity),
        )
        mpo.append(torch.stack([torch.stack(row, dim=-1) for row in rows], dim=0))
    last = torch.stack((identity, sz, local[-1]), dim=0).unsqueeze(-1)
    mpo.append(last)
    return mpo


def mpo_to_dense(mpo: Sequence[torch.Tensor]) -> torch.Tensor:
    """Contract a small MPO to a dense matrix for tests and diagnostics."""
    if not mpo:
        raise ValueError("mpo must not be empty")
    tensor = mpo[0].squeeze(0)
    # tensor: (bra_0, ket_0, right_bond)
    for site in mpo[1:]:
        tensor = torch.einsum("...a,aijb->...ijb", tensor, site)
    tensor = tensor.squeeze(-1)
    n = len(mpo)
    bra_axes = list(range(0, 2 * n, 2))
    ket_axes = list(range(1, 2 * n, 2))
    return tensor.permute(*(bra_axes + ket_axes)).reshape(2**n, 2**n)


def dense_transverse_field_ising(
    num_spins: int,
    field_x: float = 0.5,
    coupling: float = 1.0,
    longitudinal_fields: Optional[Sequence[float]] = None,
    *,
    device: Union[torch.device, str] = "cpu",
    dtype: torch.dtype = torch.complex128,
) -> torch.Tensor:
    """Independent dense construction of the same Ising Hamiltonian."""
    fields = [0.0] * num_spins if longitudinal_fields is None else list(longitudinal_fields)
    if len(fields) != num_spins:
        raise ValueError("longitudinal_fields must contain one value per spin")
    identity, sx, _, sz = spin_half_operators(device=device, dtype=dtype)
    dim = 2**num_spins
    hamiltonian = torch.zeros((dim, dim), dtype=dtype, device=device)

    def kron_sites(operators: dict[int, torch.Tensor]) -> torch.Tensor:
        result = torch.ones(1, dtype=dtype, device=device)
        for site in range(num_spins):
            result = torch.kron(result, operators.get(site, identity))
        return result

    for site in range(num_spins):
        hamiltonian += -field_x * kron_sites({site: sx})
        hamiltonian += float(fields[site]) * kron_sites({site: sz})
        if site < num_spins - 1:
            hamiltonian += -coupling * kron_sites({site: sz, site + 1: sz})
    return hamiltonian


def load_mpo_checkpoint(
    path: Union[str, Path],
    *,
    num_spins: Optional[int] = None,
    device: Union[torch.device, str] = "cpu",
    dtype: torch.dtype = torch.complex128,
) -> list[torch.Tensor]:
    """Load and validate an MPO saved as a list of rank-4 PyTorch tensors."""
    raw = torch.load(Path(path), map_location=device)
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        raise ValueError("MPO checkpoint must contain a list of at least two tensors")
    if num_spins is not None and len(raw) != num_spins:
        raise ValueError(f"MPO contains {len(raw)} sites, but num_spins={num_spins}")
    mpo = []
    for site_index, tensor in enumerate(raw):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
            raise ValueError(f"MPO site {site_index} is not a rank-4 torch.Tensor")
        if tensor.shape[1:3] != (2, 2):
            raise ValueError(f"MPO site {site_index} must have physical dimensions (2, 2)")
        mpo.append(tensor.detach().to(device=device, dtype=dtype))
    if mpo[0].shape[0] != 1 or mpo[-1].shape[-1] != 1:
        raise ValueError("MPO must have open boundaries of dimension one")
    for site in range(len(mpo) - 1):
        if mpo[site].shape[-1] != mpo[site + 1].shape[0]:
            raise ValueError(f"MPO virtual bond mismatch between sites {site} and {site + 1}")
    return mpo
