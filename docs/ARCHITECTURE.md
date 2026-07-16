# TNVD release architecture

## Design rule

The public package uses a **stable core + thin adapters** design. The stable core is copied from the working research runs. Packaging, MPO generation/loading, small presets, metadata, and tests are placed around it.

```text
generated MPO or trusted .pth MPO
               |
       mpo_factory validation
               |
       quickstart config overlay
               |
               v
      original run_automation.py
               |
   original tn_utils.py contractions
               |
original class_evolve_TNO_cut_dims.py
               |
 checkpoints + paper-loss curve + metadata
```

## Files and ownership

- `class_evolve_TNO_cut_dims.py`: original differentiable MPO evolution, orthogonalization, robust complex SVD, and truncation machinery.
- `tn_utils.py`: original tensor constructors, circuit evolution, warm starts, contractions, plotting, and loss entrypoint. Only package imports and the manuscript loss are changed.
- `run_automation.py`: original layer-growth, alternating optimization, scheduler, checkpoint, and recovery loop. Only package imports are changed.
- `config.py`: preserved paper-scale Ising configuration example.
- `mpo_factory.py`: new boundary adapter for a generated Ising MPO or an existing trusted MPO.
- `quickstart.py`: new small configuration overlay that invokes the original engine.

## Why this is reproducible

The code producing the smoke-test data follows the same execution path as the research runs: the same gate projection, MPO evolution, SVD truncation, spectrum-MPS contraction, alternating optimization, layer growth, and checkpoint format. The quickstart changes resource values, initialization mode, paths, and the number of epochs; it does not substitute another TNVD implementation.

## Checkpoints

The original engine writes separate files per circuit depth:

- `Physical_tensorsNL=<n>.pth`
- `Eigen_MPSsNL=<n>.pth`
- `Entanglement_layers_listNL=<n>.pth` for `n > 0`
- `Minimize_Energy_NL_<n>.pdf`

This format is intentionally retained so existing data and warm-start workflows remain compatible.

## Extension order

Prefer, in order: a new configuration, a new MPO adapter, a new analysis script, and only then a core change. If a core change is unavoidable, isolate it, test it against a small exact contraction, and update `docs/ORIGINAL_CORE.md`.
