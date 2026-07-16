import numpy as np
import torch

from analysis.spectrum_label_control import virtual_entropy_curve
from tnvd.autompo_models import autompo_random_field_xxz, autompo_tfim
from tnvd.exact_diagonalization import adjacent_gap_ratio, xxz_sector_hamiltonian
from tnvd.mpo_factory import mpo_to_dense, transverse_field_ising_mpo


def test_vendored_autompo_matches_direct_tfim_builder() -> None:
    fields = [0.1, -0.2, 0.3]
    direct = mpo_to_dense(
        transverse_field_ising_mpo(3, field_x=0.4, coupling=0.7, longitudinal_fields=fields)
    )
    generated = mpo_to_dense(
        autompo_tfim(3, field_x=0.4, coupling=0.7, longitudinal_fields=fields)
    )
    torch.testing.assert_close(generated, direct)


def test_xxz_sector_matches_autompo_dense_block() -> None:
    fields = [0.2, -0.1, 0.05, -0.15]
    block, basis = xxz_sector_hamiltonian(4, 2, fields, coupling_xy=0.8, coupling_z=1.2)
    dense = mpo_to_dense(
        autompo_random_field_xxz(4, fields, coupling_xy=0.8, coupling_z=1.2)
    ).real
    selected = dense[np.ix_(basis, basis)]
    torch.testing.assert_close(selected, block)


def test_gap_ratio_and_label_entropy_are_finite() -> None:
    assert np.isfinite(adjacent_gap_ratio(torch.tensor([0.0, 1.0, 2.5, 4.0, 7.0])))
    entropy = virtual_entropy_curve(np.arange(1.0, 17.0))
    assert entropy.shape == (3,)
    assert np.isfinite(entropy).all()
