from pathlib import Path
import json

import torch

from tnvd.evolution import evolve_mpo
from tnvd.models import mpo_to_dense, transverse_field_ising_mpo
from tnvd.tensors import polar_unitary
from tnvd.train import TNVDConfig, train_tnvd


def test_polar_gate_is_unitary() -> None:
    torch.manual_seed(3)
    latent = torch.randn(4, 4, dtype=torch.complex128)
    gate = polar_unitary(latent).reshape(4, 4)
    torch.testing.assert_close(gate.conj().T @ gate, torch.eye(4, dtype=gate.dtype))


def test_mpo_evolution_is_u_dagger_h_u() -> None:
    torch.manual_seed(4)
    mpo = transverse_field_ising_mpo(2)
    dense_h = mpo_to_dense(mpo)
    gate = polar_unitary(torch.randn(4, 4, dtype=torch.complex128))
    dense_u = gate.reshape(4, 4)
    evolved = mpo_to_dense(evolve_mpo(mpo, [[gate]], cutoff=16))
    expected = dense_u.conj().T @ dense_h @ dense_u
    torch.testing.assert_close(evolved, expected, atol=1e-10, rtol=1e-10)


def test_small_training_pipeline(tmp_path: Path) -> None:
    history = train_tnvd(
        TNVDConfig(
            num_spins=3,
            num_layers=1,
            spectrum_bond_dim=2,
            mpo_cutoff=6,
            epochs=2,
            learning_rate=1e-3,
            device="cpu",
            output_dir=str(tmp_path),
            print_every=1,
        )
    )
    assert len(history) == 2
    assert all(torch.isfinite(torch.tensor(row["paper_loss"])) for row in history)
    assert all(row["raw_residual_squared"] >= -1e-10 for row in history)
    for filename in ("config.json", "history.csv", "loss.pdf", "checkpoint.pt"):
        assert (tmp_path / filename).is_file()
    metadata = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    for key in ("seed", "python_version", "pytorch_version", "platform", "command"):
        assert key in metadata
