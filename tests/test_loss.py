import torch

from tnvd.loss import logarithmic_hs_loss


def test_logarithmic_loss_is_the_paper_definition() -> None:
    residual_squared = torch.tensor(32.0, dtype=torch.float64)
    loss = logarithmic_hs_loss(residual_squared, num_spins=4)
    torch.testing.assert_close(loss, torch.tensor(1.0, dtype=torch.float64))
