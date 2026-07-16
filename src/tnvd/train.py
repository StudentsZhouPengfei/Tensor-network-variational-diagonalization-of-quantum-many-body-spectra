"""Training loop for the self-contained TNVD reference example."""

from __future__ import annotations

import csv
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .evolution import evolve_mpo
from .loss import hilbert_schmidt_residual_squared, logarithmic_hs_loss, mpo_inner
from .models import load_mpo_checkpoint, transverse_field_ising_mpo
from .tensors import circuit_latents, spectrum_mps, unitary_layers


@dataclass
class TNVDConfig:
    """Configuration for a single transverse-field Ising TNVD run."""

    num_spins: int = 6
    field_x: float = 0.5
    coupling: float = 1.0
    num_layers: int = 2
    spectrum_bond_dim: int = 8
    mpo_cutoff: int = 16
    epochs: int = 200
    learning_rate: float = 2e-3
    seed: int = 7
    device: str = "auto"
    output_dir: str = "results/default"
    print_every: int = 10
    mpo_file: str = ""

    def validate(self) -> None:
        if self.num_spins < 2:
            raise ValueError("num_spins must be at least 2")
        for name in ("spectrum_bond_dim", "mpo_cutoff", "epochs", "print_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.num_layers < 0:
            raise ValueError("num_layers must be non-negative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _write_history(history: list[dict[str, float]], output_dir: Path) -> None:
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot([row["epoch"] for row in history], [row["paper_loss"] for row in history])
    axis.set(xlabel="Epoch", ylabel=r"$F=\log_2\|H-\widetilde H\|_{HS}^2-N$")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "loss.pdf")
    plt.close(figure)


def train_tnvd(config: TNVDConfig) -> list[dict[str, float]]:
    """Train TNVD and return the per-epoch diagnostics."""
    config.validate()
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_record = asdict(config) | {
        "resolved_device": str(device),
        "dtype": "float64/complex128",
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "platform": platform.platform(),
        "cuda_version": torch.version.cuda,
        "command": " ".join(sys.argv),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if config.mpo_file:
        hamiltonian_mpo = load_mpo_checkpoint(
            config.mpo_file, num_spins=config.num_spins, device=device
        )
    else:
        hamiltonian_mpo = transverse_field_ising_mpo(
            config.num_spins,
            field_x=config.field_x,
            coupling=config.coupling,
            device=device,
        )
    hamiltonian_norm_squared = mpo_inner(hamiltonian_mpo, hamiltonian_mpo).real.detach()
    energies = spectrum_mps(config.num_spins, config.spectrum_bond_dim, device=device)
    latents = circuit_latents(config.num_spins, config.num_layers, device=device)
    parameters = list(energies.parameters()) + list(latents.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    for epoch in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        evolved_mpo = evolve_mpo(
            hamiltonian_mpo, unitary_layers(latents), config.mpo_cutoff
        )
        residual_squared = hilbert_schmidt_residual_squared(
            energies, evolved_mpo, hamiltonian_norm_squared
        )
        paper_loss = logarithmic_hs_loss(residual_squared, config.num_spins)
        if not torch.isfinite(paper_loss):
            raise FloatingPointError("non-finite paper loss encountered")
        paper_loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        optimizer.step()

        row = {
            "epoch": epoch,
            "paper_loss": float(paper_loss.detach().cpu()),
            "raw_residual_squared": float(residual_squared.detach().cpu()),
        }
        history.append(row)
        if row["paper_loss"] < best_loss:
            best_loss = row["paper_loss"]
            torch.save(
                {
                    "config": config_record,
                    "epoch": epoch,
                    "paper_loss": best_loss,
                    "spectrum_mps": [tensor.detach().cpu() for tensor in energies],
                    "circuit_latents": [
                        [tensor.detach().cpu() for tensor in layer] for layer in latents
                    ],
                },
                output_dir / "checkpoint.pt",
            )
        if epoch % config.print_every == 0 or epoch == config.epochs - 1:
            print(
                f"epoch={epoch:05d} paper_loss={row['paper_loss']:.10f} "
                f"residual_squared={row['raw_residual_squared']:.10e}"
            )

    _write_history(history, output_dir)
    print(f"Finished. Best paper loss: {best_loss:.10f}. Results: {output_dir.resolve()}")
    return history
