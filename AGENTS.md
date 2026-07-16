# Maintainer and Codex guide

This repository is the standalone public TNVD implementation. Keep it self-contained: do not add manuscript sources, private data, absolute paths, or large experiment checkpoints.

## Scientific invariants

- The optimized objective is `F = log2(||H - H_tilde||_HS^2) - N`. Do not replace it with a norm or square-root distance.
- MPO sites use `(left bond, bra, ket, right bond)`.
- The circuit evolution consumed by the diagonal contraction is `U^dagger H U`.
- Spectrum-MPS physical index `r_n` is a learned binary eigenstate label, not a promise of sorted energy order.
- Claims must remain conditional on fixed tensor resources and compressibility; TNVD is not an exact diagonalizer for arbitrary Hamiltonians.

## Change protocol

1. Put model-specific MPO construction in `src/tnvd/models.py` or a new model module.
2. Keep contractions in `loss.py` and circuit/MPO evolution in `evolution.py` model-agnostic.
3. Add a small dense-reference test for every new Hamiltonian or tensor-index transformation.
4. Run `pytest`, `python -m compileall -q src tests`, and `tnvd --quickstart` before committing.
5. Never commit `results/`, `.pt`, or `.pth` files unless a small fixture is explicitly justified.

Read `docs/ARCHITECTURE.md` before changing contraction order or checkpoint structure.
