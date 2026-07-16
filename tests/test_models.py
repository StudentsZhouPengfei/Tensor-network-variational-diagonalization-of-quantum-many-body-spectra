import torch

from tnvd.models import (
    dense_transverse_field_ising,
    mpo_to_dense,
    transverse_field_ising_mpo,
    load_mpo_checkpoint,
)


def test_ising_mpo_matches_dense_hamiltonian() -> None:
    fields = [0.1, -0.2, 0.05, 0.3]
    mpo = transverse_field_ising_mpo(4, field_x=0.7, coupling=1.2, longitudinal_fields=fields)
    expected = dense_transverse_field_ising(
        4, field_x=0.7, coupling=1.2, longitudinal_fields=fields
    )
    torch.testing.assert_close(mpo_to_dense(mpo), expected, atol=1e-12, rtol=1e-12)


def test_saved_mpo_round_trip(tmp_path) -> None:
    path = tmp_path / "ising_mpo.pth"
    expected = transverse_field_ising_mpo(3)
    torch.save(expected, path)
    loaded = load_mpo_checkpoint(path, num_spins=3)
    for actual_site, expected_site in zip(loaded, expected):
        torch.testing.assert_close(actual_site, expected_site)
