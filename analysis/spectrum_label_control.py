#!/usr/bin/env python3
"""Reproduce the random-label spectrum-state control from energy vectors."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_vector(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        values = np.load(path)
    elif path.suffix == ".npz":
        archive = np.load(path)
        values = archive["eigenvalues"]
    else:
        import torch

        loaded = torch.load(path, map_location="cpu")
        values = loaded.detach().numpy() if isinstance(loaded, torch.Tensor) else np.asarray(loaded)
    return np.asarray(values, dtype=np.float64).reshape(-1)


def virtual_entropy_curve(values: np.ndarray) -> np.ndarray:
    num_spins = int(np.log2(values.size))
    if 2**num_spins != values.size:
        raise ValueError("energy-vector length must be a power of two")
    normalized = values / np.linalg.norm(values)
    entropies = []
    for cut in range(1, num_spins):
        singular_values = np.linalg.svd(normalized.reshape(2**cut, -1), compute_uv=False)
        probabilities = singular_values**2
        probabilities = probabilities[probabilities > 0]
        entropies.append(float(-np.sum(probabilities * np.log2(probabilities))))
    return np.asarray(entropies)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ed", required=True, type=Path, help="sorted ED energies")
    parser.add_argument("--tnvd", type=Path, help="optional unsorted TNVD energy vector")
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--output", type=Path, default=Path("spectrum-label-control.png"))
    parser.add_argument("--csv", type=Path, help="optional table of cut entropies")
    args = parser.parse_args()

    ed = np.sort(load_vector(args.ed))
    random_ed = ed[np.random.default_rng(args.seed).permutation(ed.size)]
    curves = {"random-permuted ED": virtual_entropy_curve(random_ed), "sorted ED": virtual_entropy_curve(ed)}
    if args.tnvd:
        curves["TNVD labels"] = virtual_entropy_curve(load_vector(args.tnvd))

    cuts = np.arange(1, next(iter(curves.values())).size + 1)
    for label, entropy in curves.items():
        plt.plot(cuts, entropy, marker="o", label=label)
    plt.xlabel("binary-label cut")
    plt.ylabel("virtual entropy (bits)")
    plt.legend(frameon=False)
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200)
    print(f"saved: {args.output}")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        labels = list(curves)
        table = np.column_stack([cuts, *(curves[label] for label in labels)])
        np.savetxt(args.csv, table, delimiter=",", header="cut," + ",".join(labels), comments="")
        print(f"saved: {args.csv}")


if __name__ == "__main__":
    main()
