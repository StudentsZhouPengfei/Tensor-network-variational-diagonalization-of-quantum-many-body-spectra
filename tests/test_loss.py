import torch

from tnvd.tn_utils import compute_loss, paper_loss_from_residual


def test_paper_loss_definition() -> None:
    residual = torch.tensor(32.0, dtype=torch.float64)
    torch.testing.assert_close(
        paper_loss_from_residual(residual, num_spins=4),
        torch.tensor(1.0, dtype=torch.float64),
    )


def test_original_compute_loss_uses_logarithmic_residual() -> None:
    spectrum_mps = [
        torch.zeros((1, 2, 1), dtype=torch.complex128),
        torch.zeros((1, 2, 1), dtype=torch.complex128),
    ]
    pauli_z = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.complex128))
    identity = torch.eye(2, dtype=torch.complex128)
    mpo = [pauli_z.reshape(1, 2, 2, 1), identity.reshape(1, 2, 2, 1)]
    loss = compute_loss(spectrum_mps, mpo, torch.tensor(4.0, dtype=torch.complex128))
    torch.testing.assert_close(loss, torch.tensor(0.0, dtype=torch.float64))
