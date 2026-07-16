# Lightweight analysis scripts

These scripts retain the scientific conventions of the research plotting
workspace while removing fixed local paths and large checkpoint assumptions.

- `spectrum_label_control.py` computes binary-cut virtual entropies for sorted
  ED, a deterministic random permutation of the same ED energies, and an
  optional unsorted TNVD energy vector.
- `plot_schmidt_resources.py` plots the two small CSV tables bundled in `data/`.

Pre-publication source fingerprints:

| Source role | Original SHA-1 |
|---|---|
| Random-label/TNVD spectrum-state control | `64fc113fc041074cc941e1ac8056dbbfa19bdf3b` |
| ED discarded-weight analysis | `fd5bb17ec315cbbbd46db4305838114ecf124163` |
| Ground-state Schmidt-tail fitter | `9fe6b447620462a8e359936a830976a712f70d08` |
| Matched Ising–XXZ plotting source | `25059ff934d1484eeffb62fe45b51d04669f19c1` |

The scripts are analysis references, not part of the TNVD optimization path.
Use `--csv` with the label-control command to export the computed entropy curves
as a human-readable table.
