# Lightweight manuscript data

This directory contains small, human-readable tables selected from the research
analysis workspace. It deliberately excludes full eigenvector matrices, TNVD
checkpoints, and other large binary artifacts.

The original exported ground-state table has SHA-1
`94de6b62b9da55779358ab233aa2ba5df798f539`. See [`analysis/README.md`](../analysis/README.md) for the
fingerprints of the scripts from which the other compact tables were selected.

## Tables

### `matched_ising_xxz_benchmark.csv`

Values used by the matched Ising–XXZ comparison at $N=14$, $N_L=10$,
$\chi_a=16$, and $\chi_t=48$. Columns contain the disorder-strength label,
mean adjacent-gap ratio, mean absolute energy error, manuscript loss, and the
reported ED/TNVD ground-state entanglement values. The numbers were extracted
without numerical modification from
`ChatGPT_plot_mbl_scaling_transition_ed_vd_label_fixed.py` in the research
analysis workspace.

The table records results, not complete disorder realizations. The legacy model
scripts contain many hard-coded field arrays whose names and selected values are
not uniformly self-describing; they are intentionally not promoted to public
paper presets until each realization is checked against the manuscript metadata.

### `discarded_schmidt_weights.csv`

Reference-rank dependence of the discarded Schmidt weight for the matched TFIM
and XXZ comparison. Here $\chi_s$ is a diagnostic Schmidt rank, not the internal
TNVD bond dimensions $\chi_a$ or $\chi_t$.

### `ground_state_schmidt_tail.csv`

Ground-state Schmidt weights and fitted tails underlying the quoted decay
exponents $\alpha_{\mathrm{XXZ}}\simeq0.893$ and
$\alpha_{\mathrm{TFIM}}\simeq1.735$. The first two comment rows retain the fit
metadata from the original exported table.

## Replot

From an editable installation at the repository root:

```bash
python analysis/plot_schmidt_resources.py --output results/schmidt-resources.png
```

All tables use decimal text so they can be inspected without PyTorch or pickle.
They are manuscript-supporting data, not outputs of the small quickstart.
