# Exact-reference tools

These commands are small-system validation utilities distilled from the original
TFIM and block-diagonal XXZ scripts. They are intentionally separate from the
stable TNVD training core.

```bash
# Full TFIM spectrum (dimension 2^N)
python tools/exact_diagonalization.py tfim --spins 8 --field-x 0.5

# Zero-magnetization XXZ block and central-50% adjacent-gap ratio
python tools/exact_diagonalization.py xxz --spins 10 --sector 5
```

Use `--fields` with exactly one comma-separated longitudinal field per site.
Use `--output result.npz` for a portable NumPy archive and add `--vectors` only
when eigenvectors are actually needed. Full eigensystems grow exponentially;
these utilities are references for validation, not scalable solvers.

## Provenance

The compact implementations preserve the Hamiltonian conventions and sector
construction of the following pre-publication research scripts while replacing
GPU/path/output hard-coding with functions and CLI arguments:

| Original role | Original SHA-1 |
|---|---|
| TFIM full-space ED | `7a198508affda55869d5418240eb82277bf12baf` |
| XXZ magnetization-block ED | `64eba2d33d678b014bece34286cd4396bdffca75` |
| XXZ level-spacing analysis | `c73f2e7daa0357ec37486155f7069d2a2675b5a0` |
