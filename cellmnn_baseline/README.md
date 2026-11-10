# Cell-MNN baseline

Adapter for the local `cell-mnn` checkout of [Learning Explicit Single-Cell
Dynamics Using ODE Representations](https://arxiv.org/abs/2510.02903v2).
It reuses their `CellMNN` encoder, initialization, Laplacian MMD,
`MnnDataset` population sampling and AdamW optimizer. It supports CITE/Multi
PCA100 and Maizels PCA50 with the shared benchmark cell holdouts and classifiers.

## Run

Use the existing PyTorch/MFM environment. Install the additional dependencies once.
The requirements pin `setuptools==75.8.0`, which supplies the legacy
`pkg_resources` module imported by Lightning 2.2.

```bash
conda activate mfm_env
python -m pip install -r cellmnn_baseline/requirements.txt
bash run_cellmnn_all.sh
```

Individual runners and overrides:

```bash
bash run_cellmnn_cite.sh --heldout_day 3 --cite_multi_time_mode real_time
bash run_cellmnn_multi.sh --heldout_day 4
bash run_cellmnn_maizels.sh                         # D3, D3.8, D8
bash run_cellmnn_maizels.sh --maizels_schedule d3_d8
bash run_cellmnn_all.sh --seeds 0 1 2 3 4
bash run_cellmnn_all.sh --smoke_test --wandb_mode offline
```

`run_cellmnn_maizels_3marginal.sh` is also available. The default is seed 42;
CITE/Multi each run both omitted-day folds, D3 and D4. CITE/Multi default to the
equal clock, and Maizels defaults to the real clock, matching the suite. Both
accept equal/real clocks. Set `FLOWMAPS_PYTHON` to select a Python executable.
Data paths use the existing local datasets; `--dataset_location` overrides them.

## Defaults

The benchmark defaults use the hyperparameters selected on Maizels and the
larger encoder from the network-size comparison. `--preset auto` selects
`cite100` for CITE/Multi or `cite50` for Maizels; both use these tuned settings.
Other defaults are read directly from the upstream `train_mnn.py`.

| Setting | CITE/Multi PCA100 (`cite100`) | Maizels PCA50 (`cite50`) |
| --- | --- | --- |
| MLP width / hidden depth | 1024 / 3 | 1024 / 3 |
| Activation / final-layer initialization scale | LeakyReLU / 0.01 | LeakyReLU / 0.01 |
| Learning rate | 0.0003 | 0.0003 |
| Batch size | 1028 | 1028 |
| Optimizer / weight decay | AdamW / 0.00001 | AdamW / 0.00001 |
| Future discount / Laplacian MMD sigma | 0.1 / 0.3 | 0.1 / 0.3 |
| Kinetic / invertibility weights | 0.1 / 1.0 | 0.1 / 1.0 |
| Maximum epochs / time limit | 1000 / 240 minutes | 1000 / 240 minutes |
| Validation interval / early-stopping patience | 10 epochs / 10 checks | 10 epochs / 10 checks |

The encoder has three hidden layers of width 1024, matching the hidden layout
of our Maizels constrained flow map: its `n_hidden=2` adds two layers after the
first. `--width 96 --depth 4` selects the earlier Cell-MNN encoder.

The batch size **1028** is the value printed in the paper. `--preset script`
uses the ordinary upstream runner's defaults, including width 96, depth 4,
learning rate 0.0002, batch size 200 and MMD sigma 1.0.
CLI hyperparameter overrides take precedence over `--config_path` JSON overrides.
For example, `--batch_size 200 --lr 0.0002` explicitly reduces the training batch
and selects that learning rate. `--num_const_dims 1` enables their zero-eigenvalue
variant; the default predicts all eigenvalues.

PCA coordinates are divided by the standard deviation of PC1, fitted only on
retained training cells. This is one global scale, preserving relative PC
variance. Sampling reverses it before distances or classifier checks. The
normalized benchmark clock is mapped onto the original source/target day range
(2 to 7, or 3 to 8), preserving the per-day units of their kinetic loss. Equal
clocks preserve these endpoints and warp intermediate times accordingly.

## Training and checkpoint selection

Training uses the upstream discounted population MMD plus kinetic and
invertibility objective. Initial cells are sampled evenly from retained days
except the final day; independent target populations are sampled from every
retained day. No OT preprocessing, cell-type supervision, or paired cell labels
are used.

Cell splits match the suite: 90% training and 10% validation, fixed split seed
701. Validation evaluates **the same complete loss** on retained-day validation
cells using reproducible batches, without advancing training RNGs. The lowest
`val_loss` selects `best.pt` and controls early stopping. Omitted-day EMDs and
lineage violation rates do not influence checkpoint selection. The held-out day
is also excluded from normalization.

The original decoder exponentiates a diagonal matrix. The adapter performs the
equivalent elementwise exponential and broadcasts the inverse basis matrix,
avoiding a dense batch-by-time-by-PCA-by-PCA tensor. Tests compare both predictions
and gradients against upstream. Kinetic regularization retains their 0.1-day
quadrature; exact retained supervision times are added when needed for an equal
clock. No upstream source files are edited. On MPS, the population MMD, matrix
inverse and determinant run explicitly on CPU with gradients preserved through
device copies. PyTorch 2.3's batched MPS inverse retains native GPU memory across
calls, so CPU inversion also applies during sampling.
Unused GPU buffers are also released after each training/validation batch, once
Metal has finished. This works around PyTorch 2.3's allocator crash and prevents
reserved memory accumulating across batches. The encoder and trajectory
multiplications still run on MPS; the paper's loss and batch size are preserved.

## Evaluation and outputs

Evaluation uses raw PCA units and the existing exact uniform-mass Euclidean W1,
multi-bandwidth RBF MMD and classifier descendant checks. The training MMD is
their Laplacian kernel; the evaluation RBF MMD remains the suite's metric.

- CITE/Multi reuse the OTP-FM evaluation: D2 to omitted day, plus local preceding
  day to omitted day. `cellmnn/test_EMD` and `test_EMD` report the native analytical
  local prediction.
- Maizels predicts each omitted day from its nearest retained left endpoint.
  Both the D3/D8 and D3/D3.8/D8 schedules are supported.
- All experiments score held-out initial cells through the final day at 50 future
  points with observed-days and all-days classifiers, without replacement.
- All Cell-MNN evaluation uses `direct`: the paper's analytical trajectory with
  one local operator inferred at the initial cell/time. The same operator
  supplies all 50 future lineage-check positions and the trajectory plot.
- Final evaluation reports observed-interval EMDs from the actual measured
  source population for each interval under `final_eval/observed_*`.

W&B uses `self-distill-flow-maps`, with the existing entity default and core
backend. Outputs go to `outputs/cellmnn_{cite,multi,maizels}_pca{100,100,50}/`:
`experiment.json`, `losses.csv`, `best.pt`, `final_metrics.json` and plots.
The selected best weights are restored for final metrics. Periodic evaluation
runs every 20 epochs; `--eval_every_epochs 0` selects final-only evaluation.

`--eval_max_points 0` evaluates full populations. A positive cap samples without
replacement; `--eval_batch_size` changes only inference chunking.
`--train_batches` and `--val_batches` can explicitly change batches per epoch.
Smoke mode uses one epoch and train/validation batch and caps evaluation at 64
cells, preserving the selected network and batch size.

## Maizels hyperparameter sweep

```bash
conda activate mfm_env

# Preview the complete grid and commands without launching training.
python scripts/sweep_maizels_cellmnn_hparams.py --dataset-location "$PWD" --device cuda --dry-run

# Run the grid; later invocations resume completed trials.
python scripts/sweep_maizels_cellmnn_hparams.py --dataset-location "$PWD" --device cuda

# A single-setting integration check with the two default seeds.
python scripts/sweep_maizels_cellmnn_hparams.py --dataset-location "$PWD" --device cuda \
  --learning-rates 0.0003 --kinetic-weights 0.1 --mmd-sigmas 0.3 \
  --smoke-test --wandb-mode offline
```

The default grid has **27 settings × 2 seeds = 54 training runs**:

| Parameter | Values |
| --- | --- |
| Learning rate | 0.0003, 0.001, 0.003 |
| Kinetic weight (`lambda_kinetic`) | 0.01, 0.1, 1 |
| Laplacian MMD bandwidth (`mmd_sigma`) | 0.3, 1, 3 |
| Training seed | 0, 1 |

`gamma` stays fixed at the base configuration's value, **0.1** by default.
`--gamma` overrides that single value. The default retains D3, D3.8 and D8,
uses real time, and inherits the Maizels `cite50` preset: width 1024, depth 3,
batch size 1028, up to 1000 epochs, and a 240-minute training limit per trial.
`--epochs`, `--time-limit`, `--batch-size`, `--patience`,
`--check-val-every-n-epoch`, `--train-batches` and `--val-batches` override
the training budget.
`--config-path` accepts the runner's JSON overrides for other fixed settings;
the three swept values take precedence over that JSON.

Each trial selects `best.pt` by retained-day validation loss and restores it
before final evaluation. The sweep ranks fully completed settings by mean
across seeds of `final_eval/direct_mean_emd_hparam_val_times`: exact raw-PCA W1
averaged over **D3.4 and D6**, using the native analytical **direct** sampler.
The other omitted days and trajectory checks are recorded as diagnostics.
Validation losses are not compared across settings.
Evaluation defaults to final-only (`--eval-every-epochs 0`); validation and early
stopping still run at the configured validation interval.

Outputs live in `outputs/maizels_cellmnn_hparam_sweep/`:

- `sweep_plan.json`: all settings, seeds, run count and effective base config.
- `results.csv`: per-seed objectives, final metrics, status and artifact paths.
- `summary.csv`: mean/std across seeds; only complete seed pairs receive ranks.
- `best_setting.json` and `best_config.json`: selected setting and reusable
  training overrides. Derived dimensions and sweep metadata are kept out of
  the strict training JSON.
- One directory per setting/seed with its derived JSON, checkpoint, metrics
  and plots. Output folders and W&B names include the unique trial ID.

Completed trials with a readable, finite direct-EMD objective are skipped;
failed or incomplete trials are retried. `--rerun-completed` forces reruns,
`--fail-fast` stops on the first failure, and `--dry-run` writes no files.
Comma-separated `--learning-rates`, `--kinetic-weights`, `--mmd-sigmas` and
`--seeds` replace the defaults. Changing effective fixed settings or evaluation
options changes trial IDs so prior results are not reused for a different run.

Rerun the selected config with the same schedule and clock:

```bash
bash run_cellmnn_maizels_3marginal.sh \
  --config_path outputs/maizels_cellmnn_hparam_sweep/best_config.json \
  --seeds 0 1
```

```bash
python -m unittest discover -s tests -p 'test_cellmnn_baseline.py' -v
python -m unittest discover -s tests -p 'test_cellmnn_hparam_sweep.py' -v
```

