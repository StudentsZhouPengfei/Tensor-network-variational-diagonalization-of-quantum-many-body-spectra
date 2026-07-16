"""Command-line entry point."""

from __future__ import annotations

import argparse

from .train import TNVDConfig, train_tnvd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a tensor-network variational diagonalization model."
    )
    parser.add_argument(
        "--quickstart",
        action="store_true",
        help="run a small end-to-end CPU-friendly smoke test",
    )
    parser.add_argument("--spins", type=int, default=6, help="number of spin-1/2 sites")
    parser.add_argument("--field-x", type=float, default=0.5, help="transverse field h_x")
    parser.add_argument("--coupling", type=float, default=1.0, help="nearest-neighbour J")
    parser.add_argument("--layers", type=int, default=2, help="brick-wall circuit layers")
    parser.add_argument("--spectrum-bond", type=int, default=8, help="spectrum-MPS bond cap")
    parser.add_argument("--mpo-cutoff", type=int, default=16, help="evolved-MPO bond cap")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--output", default="results/default")
    parser.add_argument("--mpo-file", default="", help="optional path to a saved MPO tensor list")
    parser.add_argument("--print-every", type=int, default=10)
    return parser


def config_from_args(args: argparse.Namespace) -> TNVDConfig:
    if args.quickstart:
        return TNVDConfig(
            num_spins=4,
            field_x=0.5,
            num_layers=1,
            spectrum_bond_dim=4,
            mpo_cutoff=8,
            epochs=5,
            learning_rate=5e-3,
            seed=args.seed,
            device="cpu" if args.device == "auto" else args.device,
            output_dir="results/quickstart" if args.output == "results/default" else args.output,
            print_every=1,
        )
    return TNVDConfig(
        num_spins=args.spins,
        field_x=args.field_x,
        coupling=args.coupling,
        num_layers=args.layers,
        spectrum_bond_dim=args.spectrum_bond,
        mpo_cutoff=args.mpo_cutoff,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        output_dir=args.output,
        print_every=args.print_every,
        mpo_file=args.mpo_file,
    )


def main() -> None:
    args = build_parser().parse_args()
    train_tnvd(config_from_args(args))
