# Original research-core provenance

The stable public core was selected from the clean `N=14`, `W=0`, `chi_a=16`, `chi_t=48` Ising run. Cross-checking showed that the same `run_automation.py` and `tn_utils.py` were also used by the other inspected Ising and XXZ run directories; model-specific data enter through the MPO and configuration.

## Pre-publication source fingerprints

| File | Original SHA-1 |
|---|---|
| `class_evolve_TNO_cut_dims.py` | `3f8f925c8d4d76e3f4e5db1574dc0aa2bdac7f2f` |
| `config.py` | `1859d6e20152357e53b6d2059f9951298403da79` |
| `run_automation.py` | `19bf8ac66de9c31f31a8bf3710dff0d0810902ac` |
| `tn_utils.py` | `840178904fd9d49485833f88b97f89a5d33d0e18` |

## Minimal public modifications

| File | Modification | Reason |
|---|---|---|
| `run_automation.py` | package-relative imports only | allow `python -m tnvd...` and editable installation |
| `tn_utils.py` | package-relative imports | packaging only |
| `tn_utils.py` | replace `sqrt(residual_squared)` by `log2(residual_squared) - N` | match the manuscript's logarithmic Hilbert-Schmidt loss |
| `class_evolve_TNO_cut_dims.py` | line-ending normalization only; no logic change | preserve the validated evolution/SVD implementation |
| `config.py` | final-newline normalization only; no configuration change | preserve the paper-scale example and warm-start format |

`mpo_factory.py`, `quickstart.py`, tests, CI, and documentation are new outer-layer files. They do not replace the research kernels.
