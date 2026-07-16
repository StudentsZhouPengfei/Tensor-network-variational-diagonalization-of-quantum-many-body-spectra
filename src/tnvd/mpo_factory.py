"""Small self-contained MPO builders for connecting models to the original TNVD core."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch as tc


def spin_half_operators(device="cpu", dtype=tc.complex128):
    identity = tc.eye(2, dtype=dtype, device=device)
    sx = tc.tensor([[0.0, 0.5], [0.5, 0.0]], dtype=dtype, device=device)
    sz = tc.tensor([[0.5, 0.0], [0.0, -0.5]], dtype=dtype, device=device)
    return identity, sx, sz


def transverse_field_ising_mpo(
    num_spins: int,
    field_x: float = 0.5,
    coupling: float = 1.0,
    longitudinal_fields: Optional[Sequence[float]] = None,
    *,
    device="cpu",
    dtype=tc.complex128,
):
    r"""Return the open-chain MPO used by the original contraction code.

    H = -J sum_n S^z_n S^z_{n+1} - h_x sum_n S^x_n + sum_n w_n S^z_n.
    The index order is (left virtual, bra, ket, right virtual).
    """
    if num_spins < 2:
        raise ValueError("num_spins must be at least 2")
    fields = [0.0] * num_spins if longitudinal_fields is None else list(longitudinal_fields)
    if len(fields) != num_spins:
        raise ValueError("longitudinal_fields must contain one value per spin")

    identity, sx, sz = spin_half_operators(device=device, dtype=dtype)
    local = [-field_x * sx + float(field) * sz for field in fields]
    zero = tc.zeros_like(identity)

    mpo = [tc.stack((local[0], -coupling * sz, identity), dim=-1).unsqueeze(0)]
    for site in range(1, num_spins - 1):
        rows = (
            (identity, zero, zero),
            (sz, zero, zero),
            (local[site], -coupling * sz, identity),
        )
        mpo.append(tc.stack([tc.stack(row, dim=-1) for row in rows], dim=0))
    mpo.append(tc.stack((identity, sz, local[-1]), dim=0).unsqueeze(-1))
    return mpo


def validate_mpo(mpo, num_spins: Optional[int] = None):
    if not isinstance(mpo, (list, tuple)) or len(mpo) < 2:
        raise ValueError("MPO must be a list of at least two tensors")
    if num_spins is not None and len(mpo) != num_spins:
        raise ValueError(f"MPO has {len(mpo)} sites, expected {num_spins}")
    for site, tensor in enumerate(mpo):
        if not isinstance(tensor, tc.Tensor) or tensor.ndim != 4:
            raise ValueError(f"MPO site {site} must be a rank-4 torch.Tensor")
        if tuple(tensor.shape[1:3]) != (2, 2):
            raise ValueError(f"MPO site {site} must have physical dimensions (2, 2)")
    if mpo[0].shape[0] != 1 or mpo[-1].shape[-1] != 1:
        raise ValueError("MPO must have open boundaries of dimension one")
    for site in range(len(mpo) - 1):
        if mpo[site].shape[-1] != mpo[site + 1].shape[0]:
            raise ValueError(f"MPO virtual bond mismatch at sites {site}/{site + 1}")
    return mpo


def save_mpo(mpo, path):
    validate_mpo(mpo)
    tc.save(list(mpo), path)


def load_mpo(path, num_spins: Optional[int] = None, device="cpu"):
    """Load a trusted research MPO and validate the original index convention."""
    mpo = tc.load(path, map_location=device)
    validate_mpo(mpo, num_spins=num_spins)
    return [tensor.to(device=device, dtype=tc.complex128) for tensor in mpo]


def mpo_to_dense(mpo):
    """Contract a small MPO for exact test comparisons only."""
    validate_mpo(mpo)
    tensor = mpo[0].squeeze(0)
    for site in mpo[1:]:
        tensor = tc.einsum("...a,aijb->...ijb", tensor, site)
    tensor = tensor.squeeze(-1)
    num_spins = len(mpo)
    bra_axes = list(range(0, 2 * num_spins, 2))
    ket_axes = list(range(1, 2 * num_spins, 2))
    return tensor.permute(*(bra_axes + ket_axes)).reshape(2**num_spins, 2**num_spins)
