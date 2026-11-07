# Cell-MNN baseline

Adapter for the local `cell-mnn` checkout of [Learning Explicit Single-Cell
Dynamics Using ODE Representations](https://arxiv.org/abs/2510.02903v2).
It reuses their `CellMNN` encoder, initialization, Laplacian MMD,
`MnnDataset` population sampling and AdamW optimizer. It supports CITE/Multi
PCA100 and Maizels PCA50 with the shared benchmark cell holdouts and classifiers.

## Run

Use the existing PyTorch/MFM environment. Install the additional dependency once:

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

The ordinary upstream runner's defaults are read directly from `train_mnn.py`.
For our higher-dimensional representations, `--preset auto` adds the paper's
Appendix G.1 settings:

| Setting | CITE/Multi PCA100 (`cite100`) | Maizels PCA50 (`cite50`) |
| --- | --- | --- |
| MLP width / hidden depth | 96 / 4 | 96 / 4 |
| Activation / final-layer initialization scale | LeakyReLU / 0.01 | LeakyReLU / 0.01 |
| Learning rate | 0.00005 | 0.001 |
| Batch size | 1028 | 1028 |
| Optimizer / weight decay | AdamW / 0.00001 | AdamW / 0.00001 |
| Future discount / Laplacian MMD sigma | 0.1 / 1.0 | 0.1 / 1.0 |
| Kinetic / invertibility weights | 0.1 / 1.0 | 0.1 / 1.0 |
| Maximum epochs / time limit | 1000 / 240 minutes | 1000 / 240 minutes |
| Validation interval / early-stopping patience | 10 epochs / 10 checks | 10 epochs / 10 checks |

The batch size **1028** is the value printed in the paper. `--preset script`
uses the ordinary runner's learning rate 0.0002 and batch size 200 instead.
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
clock. No upstream source files are edited. On MPS, unsupported PyTorch operations
can use CPU fallback.

## Evaluation and outputs

Evaluation uses raw PCA units and the existing exact uniform-mass Euclidean W1,
multi-bandwidth RBF MMD and classifier descendant checks. The training MMD is
their Laplacian kernel; the evaluation RBF MMD remains the suite's metric.

- CITE/Multi reuse the OTP-FM evaluation: D2 to omitted day, plus local preceding
  day to omitted day. `cellmnn/test_EMD` and `test_EMD` report the native analytical
  local prediction. `mfm/test_EMD` retains the 100-step local Euler diagnostic.
- Maizels predicts each omitted day from its nearest retained left endpoint.
  Both the D3/D8 and D3/D3.8/D8 schedules are supported.
- All experiments score held-out initial cells through the final day at 50 future
  points with observed-days and all-days classifiers, without replacement.
- `direct` is the paper's analytical trajectory using one local operator.
  `flowmap` composes analytical steps with a fresh operator at each step;
  `euler` integrates the instantaneous velocity. All three retain the suite's
  `distribution_eval/*` and `maizels/*invalid_trajectory_pct` metric names.
- Final evaluation also reports observed-interval EMDs and carries predictions
  through all retained intervals under the existing `final_eval/observed_*` and
  `final_eval/*_rollout_*` names.

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

```bash
python -m unittest discover -s tests -p 'test_cellmnn_baseline.py' -v
```

