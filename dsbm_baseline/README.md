# DSBM baseline

This adapter applies the authors' original vector DSBM implementation from
`DSBM-Gaussian.py` without modifying that checkout. It fits an unconditional
DSBM-IMF bridge (`first_coupling=ind`) for each pair of adjacent retained
marginals and composes those bridges for complete trajectory evaluation.

The default retained intervals are:

- Maizels: D3 -> D3.8 and D3.8 -> D8.
- CITE/Multi, held-out D3: D2 -> D4 and D4 -> D7.
- CITE/Multi, held-out D4: D2 -> D3 and D3 -> D7.

Cell-type labels and classifiers are not supplied to DSBM during training.
They are used only by the existing final lineage-violation evaluators. CITE and
Multi use their dataset-specific `_all_days` classifiers. Maizels additionally
reports omitted-day EMD/MMD, including the D3.4 and D6 hyperparameter-validation
EMD aggregate.

The defaults retain the paper's 40 alternating backward/forward passes, 20 SDE
steps, two 256-unit SiLU layers, and independent initial coupling. The default
step budgets are 10,000 optimizer updates per CITE/Multi interval and 50,000 per
Maizels interval. A held-out subset of each observed endpoint is used only to
select the best forward pass. One translation and one scalar scale are shared
across all retained marginals, so preprocessing does not anisotropically change
the PCA geometry; all reported metrics are evaluated back in the original PCA
coordinates.

By default the upstream checkout is expected at `~/Desktop/dsbm-pytorch`. Set
`DSBM_DIR` or pass `--dsbm_dir` to use another location. CITE/Multi data are
resolved through `CITE_MULTI_DATA_DIR`, as in the other baselines.
The active Python environment must contain the flow-maps dependencies plus the
upstream standalone script's dependencies (notably `hydra-core` and
`omegaconf`).

Run all experiments with:

```bash
DSBM_DIR=/path/to/dsbm-pytorch \
CITE_MULTI_DATA_DIR=/path/to/flow-maps-data \
FLOWMAPS_PYTHON=python \
./run_dsbm_all.sh
```

Extra CLI arguments are forwarded to every experiment, for example
`./run_dsbm_all.sh --seeds 0`.
