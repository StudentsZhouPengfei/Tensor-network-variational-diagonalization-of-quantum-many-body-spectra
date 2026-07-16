#!/usr/bin/env python3
"""Plot the bundled discarded-weight and ground-state Schmidt-tail tables."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("schmidt-resources.png"))
    args = parser.parse_args()

    discarded = np.genfromtxt(
        args.data_dir / "discarded_schmidt_weights.csv", delimiter=",", names=True
    )
    tail_path = args.data_dir / "ground_state_schmidt_tail.csv"
    lines = tail_path.read_text(encoding="utf-8").splitlines()
    table = "\n".join(line for line in lines if not line.startswith('"#'))
    tail = np.genfromtxt(io.StringIO(table), delimiter=",", names=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for model, marker in (("tfim", "o"), ("xxz", "s")):
        weights = discarded[f"{model}_discarded_weight"]
        positive = weights > 0
        axes[0].loglog(
            discarded["schmidt_rank"][positive],
            weights[positive],
            marker + "-",
            label=model.upper() if model == "xxz" else "TFIM",
        )
    axes[0].set(xlabel=r"reference rank $\chi_s$", ylabel="discarded weight")

    axes[1].semilogy(tail["Eigenvalue_Index_i"], tail["TFIM_ECS_Data"], "o", label="TFIM")
    axes[1].semilogy(tail["Eigenvalue_Index_i"], tail["XXZ_ECS_Data"], "s", label="XXZ")
    for model in ("TFIM", "XXZ"):
        fitted = tail[f"{model}_Tail_Fitted_ECS"]
        valid = np.isfinite(fitted) & (fitted > 0)
        axes[1].semilogy(tail["Eigenvalue_Index_i"][valid], fitted[valid], "--", linewidth=1)
    axes[1].set(xlabel="Schmidt index", ylabel="ground-state Schmidt weight")
    for axis in axes:
        axis.legend(frameon=False)
        axis.grid(alpha=0.25)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
