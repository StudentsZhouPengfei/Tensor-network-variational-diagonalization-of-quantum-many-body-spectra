# TNVD architecture

## Data flow

```text
Hamiltonian parameters or --mpo-file
                 |
                 v
        Hamiltonian MPO H
                 |
        circuit conjugation + SVD cutoff chi_t
                 |
                 v
             U^dagger H U -----+
                                |
spectrum MPS E_r ---------------+--> HS residual squared --> log2(.) - N
```

The loss is evaluated without constructing a dense `2^N x 2^N` operator. Dense construction exists only in tests for small systems.

## Module boundaries

- `models.py` owns Hamiltonian MPO creation/loading and small dense references.
- `tensors.py` owns trainable spectrum-MPS tensors and latent-to-unitary projection.
- `evolution.py` conjugates the MPO and applies the finite `chi_t` SVD cutoff.
- `loss.py` owns tensor-network contractions and the paper loss definition.
- `train.py` owns optimization, reproducibility metadata, checkpoints, and diagnostics.
- `cli.py` translates user-facing arguments into a `TNVDConfig`.

## Checkpoint contract

`checkpoint.pt` is a dictionary containing the resolved configuration, epoch, best paper loss, CPU spectrum-MPS tensors, and CPU circuit latent tensors. It deliberately stores latent parameters rather than optimizer internals so analysis does not depend on an optimizer version.

An external Hamiltonian checkpoint is a separate object: a list of rank-4 MPO tensors. The `--mpo-file` loader validates it before training.

## High-value next refactors

The current release focuses on a trustworthy minimal path. The following additions should be implemented only when needed and covered by tests:

1. layer-wise warm starts and optimizer-state resume;
2. random-field XXZ as a second built-in model;
3. direct spectrum-MPS sampling and exact-diagonalization comparison commands;
4. configurable alternating optimization of spectrum and circuit tensors;
5. richer truncation diagnostics, including discarded singular-value weight;
6. reproduction presets kept as small configuration files, not duplicated source trees.

Avoid restoring the old experiment layout in which every parameter set contained another copy of the same Python files. One implementation plus explicit configurations is easier to reproduce and audit.
