#!/usr/bin/env python3
"""Command-line small-system ED references for TNVD validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tnvd.exact_diagonalization import (
    adjacent_gap_ratio,
    diagonalize_tfim,
    diagonalize_xxz_sector,
)


def parse_fields(text: str | None, num_spins: int):
    if text is None:
        return [0.0] * num_spins
    fields = [float(value) for value in text.split(",")]
    if len(fields) != num_spins:
        raise ValueError(f"expected {num_spins} comma-separated fields")
    return fields


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="model", required=True)
    for model in ("tfim", "xxz"):
        sub = subparsers.add_parser(model)
        sub.add_argument("--spins", type=int, default=8)
        sub.add_argument("--fields", help="comma-separated longitudinal fields")
        sub.add_argument("--output", type=Path)
        sub.add_argument("--vectors", action="store_true", help="also save eigenvectors")
    subparsers.choices["tfim"].add_argument("--field-x", type=float, default=0.5)
    subparsers.choices["tfim"].add_argument("--coupling", type=float, default=1.0)
    subparsers.choices["xxz"].add_argument("--sector", type=int)
    subparsers.choices["xxz"].add_argument("--coupling-xy", type=float, default=1.0)
    subparsers.choices["xxz"].add_argument("--coupling-z", type=float, default=1.0)
    return parser


def main():
    args = build_parser().parse_args()
    fields = parse_fields(args.fields, args.spins)
    if args.model == "tfim":
        eigenvalues, eigenvectors = diagonalize_tfim(
            args.spins,
            field_x=args.field_x,
            coupling=args.coupling,
            longitudinal_fields=fields,
        )
        metadata = {"model": "tfim"}
    else:
        sector = args.spins // 2 if args.sector is None else args.sector
        eigenvalues, eigenvectors, basis = diagonalize_xxz_sector(
            args.spins,
            sector,
            longitudinal_fields=fields,
            coupling_xy=args.coupling_xy,
            coupling_z=args.coupling_z,
        )
        metadata = {"model": "xxz", "sector": sector, "basis": np.asarray(basis)}
        if eigenvalues.numel() >= 3:
            metadata["central_half_gap_ratio"] = adjacent_gap_ratio(eigenvalues)

    print(f"{args.model.upper()}: {eigenvalues.numel()} eigenvalues")
    print(f"energy range: [{eigenvalues[0]:.12g}, {eigenvalues[-1]:.12g}]")
    if "central_half_gap_ratio" in metadata:
        print(f"central-50% mean adjacent-gap ratio: {metadata['central_half_gap_ratio']:.8f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"eigenvalues": eigenvalues.numpy(), **metadata}
        if args.vectors:
            payload["eigenvectors"] = eigenvectors.numpy()
        np.savez_compressed(args.output, **payload)
        print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
