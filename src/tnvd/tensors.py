"""Variational tensor constructors and circuit parameterization."""

from __future__ import annotations

import torch
from typing import Optional, Union


def spectrum_mps(
    num_spins: int,
    bond_dim: int,
    *,
    device: Union[torch.device, str],
    dtype: torch.dtype = torch.float64,
    generator: Optional[torch.Generator] = None,
) -> torch.nn.ParameterList:
    """Create a real-valued open-boundary MPS for the energy tensor."""
    if num_spins < 2 or bond_dim < 1:
        raise ValueError("num_spins >= 2 and bond_dim >= 1 are required")
    tensors: list[torch.nn.Parameter] = []
    left_dim = 1
    for site in range(num_spins):
        right_dim = 1 if site == num_spins - 1 else min(bond_dim, 2 ** (site + 1))
        if site < num_spins - 1:
            right_dim = min(right_dim, 2 ** (num_spins - site - 1))
        value = torch.randn(
            left_dim, 2, right_dim, dtype=dtype, device=device, generator=generator
        ) / (2.0 * max(left_dim, 1)) ** 0.5
        tensors.append(torch.nn.Parameter(value))
        left_dim = right_dim
    return torch.nn.ParameterList(tensors)


def circuit_latents(
    num_spins: int,
    num_layers: int,
    *,
    device: Union[torch.device, str],
    dtype: torch.dtype = torch.complex128,
    generator: Optional[torch.Generator] = None,
) -> torch.nn.ModuleList:
    """Create near-identity latent matrices for a brick-wall circuit."""
    if num_layers < 0:
        raise ValueError("num_layers must be non-negative")
    layers = torch.nn.ModuleList()
    for layer in range(num_layers):
        parity = layer % 2
        sites = list(range(parity, num_spins - 1, 2))
        params = torch.nn.ParameterList()
        for _ in sites:
            real = torch.randn((4, 4), dtype=torch.float64, device=device, generator=generator)
            imag = torch.randn((4, 4), dtype=torch.float64, device=device, generator=generator)
            latent = torch.eye(4, dtype=dtype, device=device) + 1e-3 * torch.complex(real, imag)
            params.append(torch.nn.Parameter(latent))
        layers.append(params)
    return layers


def polar_unitary(latent: torch.Tensor) -> torch.Tensor:
    """Project a full-rank latent 4x4 matrix to its unitary polar factor."""
    u, _, vh = torch.linalg.svd(latent, full_matrices=False)
    return (u @ vh).reshape(2, 2, 2, 2)


def unitary_layers(latents: torch.nn.ModuleList) -> list[list[torch.Tensor]]:
    return [[polar_unitary(latent) for latent in layer] for layer in latents]
