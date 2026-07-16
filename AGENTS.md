# Maintainer and Codex guide

This repository publishes a minimally modified copy of the working TNVD research code. The original contraction/SVD/evolution path is the scientific reference. Engineering changes belong around that core, not inside it.

## Non-negotiable invariants

- Preserve `class_evolve_TNO_cut_dims.py`, `run_automation.py`, and `tn_utils.py` unless a change is required by a demonstrated bug or the manuscript definition.
- The public scientific change is the loss `F = log2(||H - H_tilde||_HS^2) - N`; never restore the square-root distance.
- MPO sites use `(left bond, bra, ket, right bond)` and are validated at the adapter boundary.
- Spectrum-MPS index `r_n` is a learned binary label, not a sorted-energy index.
- Claims remain conditional on fixed resources and joint compressibility.
- Do not add manuscript sources, private paths, or large checkpoints.

## Preferred extension pattern

1. Add a model through `mpo_factory.py` or load a trusted research MPO.
2. Add a thin preset/wrapper that mutates a copy of `config`.
3. Reuse `run_automation.main()` and the original tensor kernels.
4. Add a small exact test at the MPO boundary and an end-to-end smoke test.
5. Document any unavoidable core change in `docs/ORIGINAL_CORE.md`.

Do not replace the research engine with a newly designed abstraction merely to make the layout look more conventional. Read `docs/ARCHITECTURE.md` and `docs/ORIGINAL_CORE.md` before editing.
