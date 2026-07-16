from pathlib import Path

import torch

from tnvd.mpo_factory import load_mpo, mpo_to_dense, save_mpo, transverse_field_ising_mpo


def dense_ising(num_spins: int, field_x: float) -> torch.Tensor:
    identity = torch.eye(2, dtype=torch.complex128)
    sx = torch.tensor([[0.0, 0.5], [0.5, 0.0]], dtype=torch.complex128)
    sz = torch.tensor([[0.5, 0.0], [0.0, -0.5]], dtype=torch.complex128)
    result = torch.zeros((2**num_spins, 2**num_spins), dtype=torch.complex128)

    def product(operators):
        tensor = torch.ones(1, dtype=torch.complex128)
        for site in range(num_spins):
            tensor = torch.kron(tensor, operators.get(site, identity))
        return tensor

    for site in range(num_spins):
        result -= field_x * product({site: sx})
        if site < num_spins - 1:
            result -= product({site: sz, site + 1: sz})
    return result


def test_generated_mpo_matches_dense_hamiltonian() -> None:
    mpo = transverse_field_ising_mpo(4, field_x=0.7)
    torch.testing.assert_close(mpo_to_dense(mpo), dense_ising(4, 0.7))


def test_research_mpo_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "hamiltonian.pth"
    expected = transverse_field_ising_mpo(3)
    save_mpo(expected, path)
    actual = load_mpo(path, num_spins=3)
    for actual_site, expected_site in zip(actual, expected):
        torch.testing.assert_close(actual_site, expected_site)
