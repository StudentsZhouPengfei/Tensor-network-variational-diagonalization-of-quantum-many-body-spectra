"""Lightweight model adapters built on Hao-Kai Zhang's finite-state AutoMPO."""

from __future__ import annotations

from collections.abc import Sequence

import torch as tc

from ._vendor.autompo import GenSpinOpr, fsa, named_data
from .mpo_factory import validate_mpo


def _fields(values: Sequence[float] | None, num_spins: int) -> list[float]:
    fields = [0.0] * num_spins if values is None else [float(value) for value in values]
    if len(fields) != num_spins:
        raise ValueError("longitudinal_fields must contain one value per spin")
    return fields


def _to_tnvd_tensors(arrays, *, device="cpu", dtype=tc.complex128):
    # AutoMPO emits (left bond, right bond, bra, ket). TNVD uses (left, bra, ket, right).
    tensors = [
        tc.as_tensor(array, dtype=dtype, device=device).permute(0, 2, 3, 1).contiguous()
        for array in arrays
    ]
    return validate_mpo(tensors)


def autompo_tfim(
    num_spins: int,
    field_x: float = 0.5,
    coupling: float = 1.0,
    longitudinal_fields: Sequence[float] | None = None,
    *,
    device="cpu",
    dtype=tc.complex128,
):
    r"""Generate ``-J Sz Sz - hx Sx + sum_i w_i Sz`` with the vendored AutoMPO."""
    if num_spins < 2:
        raise ValueError("num_spins must be at least 2")
    fields = _fields(longitudinal_fields, num_spins)
    sx = named_data("Sx", GenSpinOpr("Sx"))
    sz = named_data("Sz", GenSpinOpr("Sz"))
    automaton = fsa(num_spins)
    for site in range(num_spins):
        if site < num_spins - 1:
            automaton.Add(-float(coupling), [sz, sz], [site, site + 1])
        automaton.Add(-float(field_x), [sx], [site])
        if fields[site] != 0.0:
            automaton.Add(fields[site], [sz], [site])
    return _to_tnvd_tensors(automaton.GenMPO(), device=device, dtype=dtype)


def autompo_random_field_xxz(
    num_spins: int,
    longitudinal_fields: Sequence[float] | None = None,
    coupling_xy: float = 1.0,
    coupling_z: float = 1.0,
    *,
    device="cpu",
    dtype=tc.complex128,
):
    r"""Generate an open random-field XXZ MPO.

    ``H = Jxy/2 sum(Sp Sm + Sm Sp) + Jz sum(Sz Sz) + sum_i w_i Sz``.
    """
    if num_spins < 2:
        raise ValueError("num_spins must be at least 2")
    fields = _fields(longitudinal_fields, num_spins)
    sp = named_data("Sp", GenSpinOpr("Sp"))
    sm = named_data("Sm", GenSpinOpr("Sm"))
    sz = named_data("Sz", GenSpinOpr("Sz"))
    automaton = fsa(num_spins)
    for site in range(num_spins):
        if site < num_spins - 1:
            automaton.Add(0.5 * float(coupling_xy), [sp, sm], [site, site + 1])
            automaton.Add(0.5 * float(coupling_xy), [sm, sp], [site, site + 1])
            automaton.Add(float(coupling_z), [sz, sz], [site, site + 1])
        if fields[site] != 0.0:
            automaton.Add(fields[site], [sz], [site])
    return _to_tnvd_tensors(automaton.GenMPO(), device=device, dtype=dtype)
