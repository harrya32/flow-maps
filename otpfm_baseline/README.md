# OTP-FM on CITE/Multi and Maizels

OTP-FM with independent or precomputed OT coupling and either W2Inf or
W2Potential on the same CITE/Multi PCA100 H5AD files and Maizels PCA50 CSV
as the constrained flow-map experiments.
The runner imports the local
`OTP-FM` checkout; its sources are used without modification. Use the existing
`mfm_env` environment (Python 3.10+, PyTorch, POT, AnnData, pandas, SciPy,
scikit-learn, Matplotlib, pytorch-lightning, and W&B).
For the Lightning 2.2 environment, pin `setuptools==75.8.0` to supply its
legacy `pkg_resources` import.

```bash
conda activate mfm_env
python -m pip install setuptools==75.8.0
bash run_otpfm_cite.sh
bash run_otpfm_multi.sh
bash run_otpfm_maizels.sh --dataset_location "$PWD"

# All three datasets; both CITE/Multi folds, five seeds.
bash run_otpfm_all.sh --seeds 0 1 2 3 4

# Upstream OT-coupled tuples with the W2Inf potential.
bash run_otpfm_all.sh --ot_coupling

# W2Potential, which computes OT against predicted minibatch positions.
bash run_otpfm_all.sh --potential w2

# Combine both options.
bash run_otpfm_all.sh --potential w2 --ot_coupling

# A single fold and the elapsed-day clock.
bash run_otpfm_cite.sh --heldout_day 4 --cite_multi_time_mode real_time

# One batch of training and validation, with capped evaluation.
bash run_otpfm_cite.sh --heldout_day 4 --smoke_test --wandb_mode offline
```

For CITE/Multi, `--dataset_location` accepts the benchmark data directory or
one H5AD file; otherwise `CITE_MULTI_DATA_DIR` / `~/Desktop/flow-maps-data`
is used. Scripts accept `FLOWMAPS_PYTHON` to select another Python executable. Outputs are
written under `outputs/otpfm_{cite,multi}_pca100/<run-name>/` or
`outputs/otpfm_maizels_pca50/<run-name>/`. The default is one seed, 42;
CITE/Multi run both omitted-day folds and Maizels runs one schedule per seed.

## Maizels PCA50

```bash
# D3/D3.8/D8 retained, physical time, independent coupling and W2Inf.
bash run_otpfm_maizels.sh --dataset_location "$PWD"

# Precomputed OT tuples, or the actual W2Potential, or both.
bash run_otpfm_maizels.sh --dataset_location "$PWD" --ot_coupling
bash run_otpfm_maizels.sh --dataset_location "$PWD" --potential w2
bash run_otpfm_maizels.sh --dataset_location "$PWD" --potential w2 --ot_coupling

# Alternative schedule or clock.
bash run_otpfm_maizels.sh --maizels_schedule d3_d8
bash run_otpfm_maizels.sh --maizels_time_mode equal_time

# One training/validation batch, with capped evaluation.
bash run_otpfm_maizels.sh --smoke_test --device cuda --wandb_mode offline
```

`--dataset_location` accepts `celltype_classification_pca50_dataset.csv.gz`,
its directory, or a directory containing it under `data/`. Without an override,
the runner checks the repository root before the shared Maizels default path.
`run_otpfm_maizels_3marginal.sh` is an alias for the default three-day schedule.
`run_otpfm_all.sh` now includes Maizels; a shared `--dataset_location` directory
should contain the two H5AD files and the Maizels CSV.

The default schedule retains D3, D3.8 and D8. Its real-time clock places them
at 0, 0.16 and 1, with one potential at D3.8. `equal_time` places the retained
days at 0, 0.5 and 1 and interpolates the omitted days using the suite's existing
clock. All network and training hyperparameters use the same CITE presets
listed below, with the input/output dimension changed to 50.

The shared 90/10 cell split (seed 701) is applied before normalization or OT
alignment. Training and checkpoint selection use only retained-day training
and validation tuples, respectively. The seven omitted days are used only for
evaluation: each is predicted from the preceding retained population, and
raw-PCA W1 and RBF MMD2 are logged per day and averaged. Validation loss selects
the checkpoint; the distribution metrics remain diagnostics. Final evaluation
also logs observed-interval and D3-to-D8 rollout distances.

Trajectories start from held-out D3 cells and use the existing lineage graph,
50 future check times, and both the D3/D3.8/D8 and all-days classifiers.
Classifier overrides use `--classifier_path` and `--full_data_classifier_path`.
W&B receives the same `distribution_eval/`, `maizels/` and `final_eval/` metrics
as the suite, plus `OTPFM/test_EMD` for the mean omitted-day composed-map W1.
Direct predictions, 50-step composed maps and 50-step Euler predictions are
reported separately. Composed-map trajectories and intermediate distributions
are plotted in W&B and saved as PNGs.

The endpoint schedule `d3_d8` evaluates all eight omitted days. With no retained
intermediate day there is no potential, so it runs the upstream MeanFlow loss
with OTP corrections disabled. OT coupling still controls endpoint pairing;
the potential option is inactive for this schedule. The available all-days
classifier is used for both classifier metrics unless overridden, matching
the other endpoint baseline runs.

## Upstream defaults and training

The runner loads upstream `configs/citeseq/defaults.json` and the matching
variant preset, selected from the effective coupling and potential options:

| Potential / coupling | Upstream preset | Epochs | LR schedule |
| --- | --- | --- | --- |
| W2Inf / independent | `W2Inf.json` | 80 | None |
| W2Inf / precomputed OT | `W2.json` | 200 | Cosine |
| W2Potential / either coupling | Shared `defaults.json` (no dedicated CITE preset) | 200 | Cosine |

Shared settings are 10 hidden layers of width 768, SiLU, LayerNorm, dropout 0.2,
residual connections every two layers, 64-dimensional position/time embeddings, MeanFlow with
adaptive weighting, Adam at 0.003, batch size 256,
EMA decay 0.99, strength 300 and a delta temporal kernel. It reuses the upstream
model/potential factories, tuple dataset, MaxStdScaler, and base
trainer, including its exact curriculum conversion. `--config_path` provides
JSON overrides after preset selection; CLI options take precedence. Explicit
epoch/scheduler overrides are preserved. `upstream_config_preset` is recorded
in the effective config and W&B metadata.

`--ot_coupling` reuses `compute_ot_alignments`: full-population squared-Euclidean
OT between consecutive retained days, uniform cell masses, and row-wise argmax
matches chained through the retained days (D2 to D7 for CITE/Multi, D3 to D8
for Maizels). Training and validation alignments are computed separately
after the shared cell split.
`--ot_method emd` is the default; `--ot_method sinkhorn --ot_reg 0.01` selects
the upstream Sinkhorn implementation. These settings affect startup pairings;
W2Potential always uses its upstream exact minibatch OT gradient. The argmax
conversion can repeat target cells and does not retain the full plan's masses.
`--no-ot_coupling` explicitly selects independent tuples, including when
overriding a JSON config. The paper's CITE `W2.json` preset means W2Inf plus
OT-coupled tuples; `--potential w2` selects the actual `W2Potential` class.

For CITE/Multi, benchmark adaptations are PCA100, the global four-day clock,
and the shared 90/10 cell splits (split seed 701). Training tuples contain only training
cells from the three retained days. MaxStdScaler is fitted on those training
cells and inverted before all distances and classifier checks. The omitted
day is excluded from normalization, training, and checkpoint selection.
W2Inf uses `solver_type="direct"` with its matrix cache. W2Potential uses the
upstream `anderson_safe` iterative solver because its target assignment changes
with the predicted positions; the upstream direct formula assumes a fixed
assignment. `picard_steps` defaults to 5 and can be changed through JSON;
other upstream iterative solvers can be selected with JSON `solver_type`.
Validation ranks snapshots by
unweighted MSE at full OTP strength on retained-day validation tuples; training
keeps the upstream adaptive loss and curriculum. The matching EMA snapshot is
used for final evaluation.

W&B uses the `self-distill-flow-maps` project and the same entity default as
the constrained runs; override with `--wandb_project`, `--wandb_entity`, or
`WANDB_ENTITY`. `WANDB_MODE` is respected unless `--wandb_mode` is passed.
The runner enables W&B's bundled core backend, avoiding the older Python
backend's `ServerSettingsType` initialization failure with the current cloud API.
Each run evaluates every 20 epochs (`--eval_every_epochs` overrides this)
and evaluates its validation-selected checkpoint at completion.
The existing CITE/Multi metric names are preserved:

- `distribution_eval/{day}_{direct,flowmap,euler}_{emd,rbf_mmd2}` and
  `final_eval/{direct,flowmap,euler}_mean_{emd,rbf_mmd2}`: all D2 cells pushed
  to the omitted day; flow-map and Euler sampling use 50 steps over that horizon.
- `mfm/test_EMD` and `mfm/test_EMD_flowmap`: all cells from the preceding real
  day pushed to the omitted day with 100 Euler or composed-map steps.
- `maizels/{full_data_classifier/}model_{direct,flowmap,euler}_invalid_trajectory_pct`
  and `final_eval/full_data_classifier/{direct,flowmap,euler}_invalid_trajectory_pct`:
  held-out D2 cells sampled once, checked at 50 times through D7 with the existing
  lineage graph, zero confidence/margin thresholds, and both observed-days and
  all-days classifiers. The all-days composed-map paths are plotted in W&B.

Population metrics use exact uniform-mass Euclidean W1 and the existing biased
multi-bandwidth RBF MMD. All metrics use the selected equal- or real-time clock.
`--eval_max_points 0` (default) uses complete populations; a positive value
caps each population without replacement. `--smoke_test` caps at 64 cells and
uses one training/validation batch while retaining the upstream network. With
OT coupling, smoke tests also cap each alignment population at one batch before
the dense OT solve; full runs use every retained training/validation cell.

Each run saves `experiment.json` (effective config, clock, normalization, and
classifier paths), `models/best.pt` (upstream model plus its matching EMA),
`losses.csv`, periodic upstream checkpoints, and `final_metrics.json`.
Run names and W&B configuration distinguish the potential class, coupling,
startup OT method, and solver, so the variants have separate output directories.

```bash
python -m unittest discover -s tests -p 'test_otpfm_baseline.py'
python -m unittest discover -s tests -p 'test_otpfm_maizels.py'
```
