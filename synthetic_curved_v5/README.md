# Configurable curved-v5 synthetic benchmark

This directory is an isolated port of the synthetic experiment used in Section
5.1 and Table 2 of the trajectory-inference paper. It contains the complete
ground-truth generator, the three reported methods, and one new comparison:

- **SF2M**: velocity and score matching followed by a matched zero-fate
  continuation.
- **SSFM**: a directly trained strong stochastic flow map with local bridge
  matching and own-map consistency.
- **CLIFT**: the same strong map with a classifier-based lineage loss.
- **real_clift**: the Maizels-style end-to-end objective: a 75/25 split of
  local Euler--Maruyama targets and off-diagonal semigroup targets, with the
  lineage loss evaluated on the same current-model off-diagonal prediction.
  The ordinary lineage loss uses the detached frozen-classifier distribution
  at the off-diagonal input and the classifier distribution at the predicted
  endpoint. Its two-half-step target comes from an EMA teacher, while its saved
  and evaluated checkpoint always contains the instantaneous student parameters.

The implementation is self-contained and does not modify or import the existing
Maizels, CITE/MULTI, LARRY, or Schiebinger experiment code.

It is adapted from the MIT-licensed `memory_crossing_experiment/curved_*`
implementation in `flow_maps_benji`; the upstream license is preserved in
`UPSTREAM_LICENSE.txt`. The machine-specific audit/freezing framework has been
replaced by a portable runner, while the generator, observed-only fitting
boundary, model objectives, five training seeds, and reported evaluation
budgets are retained.

## Quick smoke test

From the repository root, using a Python environment containing PyTorch, NumPy,
SciPy, and Matplotlib:

```bash
python -m pip install -r synthetic_curved_v5/requirements.txt
python -m synthetic_curved_v5.run all \
  --config synthetic_curved_v5/configs/smoke.json \
  --root outputs/synthetic_curved_v5_smoke
```

## Paper configuration

The full paper settings are in `configs/paper.json`. The run is intentionally
split into collision-safe stages:

```bash
python -m synthetic_curved_v5.run prepare \
  --config synthetic_curved_v5/configs/paper.json \
  --root outputs/synthetic_curved_v5_paper

python -m synthetic_curved_v5.run train \
  --root outputs/synthetic_curved_v5_paper

python -m synthetic_curved_v5.run evaluate \
  --root outputs/synthetic_curved_v5_paper
```

To train and evaluate only the Maizels-style variant:

```bash
python -m synthetic_curved_v5.run all \
  --config synthetic_curved_v5/configs/paper.json \
  --root outputs/synthetic_curved_v5_real_clift \
  --methods real_clift \
  --priors ground_truth_ot \
  --device cpu
```

After evaluation, reproduce the paper-style three-panel conditional trajectory
figure with lineage-violating paths coloured red:

```bash
python -m synthetic_curved_v5.plot_violating_trajectories \
  --root outputs/synthetic_curved_v5_real_clift \
  --sf2m-root outputs/synthetic_curved_v5_ground_truth \
  --seed 21 \
  --prior ground_truth_ot
```

This defaults to the paper conditioning point at approximately
`t=0.55, x=(0.585, -0.17)`. It draws 48 fresh trajectories on the paper's
dense `0.005` output grid using 4,160 integration/map steps per unit time and
writes `evaluation/conditional_trajectories.{pdf,png,svg}`. The SF2M root must
use the same prepared dataset and compatible SF2M configuration; no retraining
is performed.

`real_clift` uses the same fixed, precomputed OT plans as the original methods.
It trains for the configured fixed budget without early stopping. Its maximum
off-diagonal horizon grows geometrically from `local_step_fraction` to
`max_horizon_fraction` during the first `horizon_curriculum_fraction` of
training. The local endpoint error is divided by the squared horizon so that
the Euler--Maruyama target supplies horizon-independent drift supervision. The
paper config uses the one-seed-selected learning rate `1e-3` and lineage weight
`1.35`; these remain ordinary JSON settings for subsequent multi-seed studies.

The paper configuration trains five seeds. It is therefore a real experiment,
not a quick unit test.

The default generator was checked numerically against the upstream checkout:
means, width schedule, density/score/velocity fields, drift, and seeded
Euler--Maruyama paths agree, as do the initial SF2M/strong-map networks, OT
plans, and bridge batches on CPU.

## Editing the experiment

Copy `configs/paper.json`, give the copy a new name, edit it, and use a new
output directory. Every formerly hard-coded synthetic-process choice is exposed:

- `dataset.ramps` and `dataset.phenotype_coefficients`: branching times,
  curvature and terminal geometry.
- `dataset.progress_coefficients`: optional branch-specific progress offsets;
  these can create transport ambiguity without collapsing cell-type regions.
- `dataset.std_knots` and `dataset.std_values`: cloud width and anisotropy.
- `dataset.mixture_weights`: terminal population masses.
- `dataset.true_diffusion`: ground-truth diffusion floor, peaks, widths and
  progress diffusion.
- `dataset.labels`: the five hard cell-type regions.
- `sf2m`, `strong_map`, `real_clift`, and `network`: optimization and
  architecture settings.
- `evaluation`: conditional and marginal Monte Carlo and integration budgets.
- `seeds`: independent neural-network fitting seeds.

Small one-off changes can be supplied without editing JSON:

```bash
python -m synthetic_curved_v5.run all \
  --config synthetic_curved_v5/configs/smoke.json \
  --root outputs/synthetic_variant \
  --set dataset.true_diffusion.first_diffusion_peak=0.35 \
  --set strong_map.fate_weight=0.5
```

Overrides are written into `resolved_config.json`, so every result is tied to
the exact settings used. The runner also checks that the port source files have
not changed between stages. Existing output roots are refused during preparation.

To run only part of the comparison, use comma-separated filters:

```bash
python -m synthetic_curved_v5.run train \
  --root outputs/synthetic_curved_v5_paper \
  --methods ssfm,clift --seeds 21,22 --priors ground_truth_ot --device cpu
```

The two configs are intentionally separate: `paper.json` holds the complete
five-seed settings, while `smoke.json` exercises the entire pipeline quickly
without producing scientifically meaningful scores.

## Synthetic-process stress study

`process_study.py` contains controlled generator variants for testing whether
the ordinary classifier-to-classifier lineage loss helps specifically when
lineage-blind OT takes a geometric shortcut. In the `committed_swap_*`
profiles, both child fates are identifiable before the last interval, but
their relative progress ordering changes afterwards. The oracle paths remain
lineage-valid while Euclidean OT begins to pair incompatible siblings. The
`close_only` control changes separation without reversing that ordering.

Run the moderate profile over five seeds with unchanged SF2M, SSFM and Real
CLIFT objectives using the quicker screening evaluation budget:

```bash
python -m synthetic_curved_v5.process_study all \
  --output-parent outputs/synthetic_process_study_reproduction \
  --profiles committed_swap_moderate \
  --methods sf2m,ssfm,real_clift \
  --seeds 21,22,23,24,25 \
  --device cpu
```

Add `--full-evaluation` for the original paper Monte Carlo and integration
budget. Each output root records the complete resolved configuration and a
`process_diagnostics.json` file. To inspect a standalone configuration before
training, run:

```bash
python -m synthetic_curved_v5.process_diagnostics \
  --config synthetic_curved_v5/configs/paper.json
```

## Output structure

```text
<root>/
  resolved_config.json
  provenance.json
  dataset/{observed_snapshots,training_data,oracle_paths}.npz + dataset.png
  classifier/{classifier.pt,manifest.json,splits.npz}
  inputs/learning_data.npz
  models/seed_<seed>/<prior>/{sf2m,ssfm,clift,real_clift}/model.pt
  evaluation/{per_fit/,summary.json,summary.csv,summary.png,conditional_paths.png}
```

Only the snapshots at `dataset.training_times` enter classifier or model
training. The other marginals and oracle process are opened only by the
evaluation stage.

Core checks can be rerun with:

```bash
python -m unittest discover -s synthetic_curved_v5/tests -v
```

## Reproduction scope

The scientific ingredients and numerical defaults match the curved-v5 paper
experiment. The extensive source-freezing, machine-specific runtime calibration,
and independent audit machinery from the original development repository are
not copied. Conditional SF2M rates in `paper.json` are the final recorded
runtime-matched rates (280/290 steps per unit); marginal evaluation retains 400
SF2M steps and 4,160 map steps per unit, as in Table 2.
