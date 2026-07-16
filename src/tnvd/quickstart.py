"""A thin, reproducible wrapper around the original TNVD automation engine."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
from pathlib import Path

import torch as tc

from .config import config
from .mpo_factory import load_mpo, save_mpo, transverse_field_ising_mpo
from .run_automation import main as run_original_automation


def run_quickstart(
    output_dir="results/quickstart",
    *,
    num_spins=4,
    epochs=5,
    max_layers=1,
    seed=7,
    field_x=0.5,
    spectrum_bond=4,
    mpo_cutoff=8,
    mpo_file=None,
):
    """Generate an MPO, inject a small config, and run the original engine unchanged."""
    tc.manual_seed(seed)
    output = Path(output_dir).resolve()
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    mpo_path = output / "ising_mpo.pth"
    if mpo_file is None:
        mpo = transverse_field_ising_mpo(num_spins, field_x=field_x)
    else:
        mpo = load_mpo(mpo_file, num_spins=num_spins)
    save_mpo(mpo, mpo_path)

    original_config = copy.deepcopy(config)
    try:
        config["system"].update(
            spin_num=num_spins,
            cut_entanglement_dims=mpo_cutoff,
            mdims=spectrum_bond,
            save_qising_name=str(mpo_path),
        )
        config["schedule"].update(
            start_layer=0,
            max_layers=max_layers,
            max_epoch_per_layer=epochs,
            base_lr=2e-3,
            nl0_lr=2e-2,
            eta_min=1e-6,
        )
        config["convergence"].update(patience=max(epochs + 1, 10), absolute_tol=1e-6)
        config["init_strategy"].update(mode="random", fallback_on_failure=True)
        config["logging"].update(
            dt_print=1,
            log_file=str(output / "automation_record.log"),
            plot_save_dir=str(checkpoints),
        )
        metadata = {
            "purpose": "small smoke test of the original TNVD execution path",
            "config": config,
            "seed": seed,
            "source_mpo": "generated Ising MPO" if mpo_file is None else str(mpo_file),
            "python_version": platform.python_version(),
            "pytorch_version": tc.__version__,
            "platform": platform.platform(),
            "cuda_version": tc.version.cuda,
            "command": " ".join(sys.argv),
        }
        (output / "quickstart_config.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        run_original_automation()
    finally:
        config.clear()
        config.update(original_config)
    return output


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run a small test through the original TNVD research-code path."
    )
    parser.add_argument(
        "--quickstart", action="store_true", help="explicit alias for the default small run"
    )
    parser.add_argument("--output", default="results/quickstart")
    parser.add_argument("--spins", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--field-x", type=float, default=0.5)
    parser.add_argument("--spectrum-bond", type=int, default=4)
    parser.add_argument("--mpo-cutoff", type=int, default=8)
    parser.add_argument("--mpo-file", default=None)
    return parser


def main():
    args = build_parser().parse_args()
    output = run_quickstart(
        args.output,
        num_spins=args.spins,
        epochs=args.epochs,
        max_layers=args.layers,
        seed=args.seed,
        field_x=args.field_x,
        spectrum_bond=args.spectrum_bond,
        mpo_cutoff=args.mpo_cutoff,
        mpo_file=args.mpo_file,
    )
    print(f"Quickstart complete: {output}")
