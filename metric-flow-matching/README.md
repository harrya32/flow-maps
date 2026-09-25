# [Metric Flow Matching for Smooth Interpolations on the Data Manifold](https://arxiv.org/abs/2405.14780)

<div align="center">

[![arxiv](https://img.shields.io/badge/arxiv-blue)](https://arxiv.org/abs/2405.14780)
[![twitter](https://img.shields.io/badge/twitter-thread-green)](https://x.com/KKapusniak1/status/1797632928920564014)

</div>

<div align="center">
    <p align="center">
        <img align="middle" src="./assets/arch.gif" alt="Arch" width="500" />
    </p>
</div>

## Installation

To set up the environment, you need to install the required dependencies. You can do this by using the `requirements.txt` file.

```bash
conda create --name myenv python=3.11
conda activate myenv
pip install -r requirements.txt
```

## Datasets

Please download the following datasets to run the experiments.

- **Lidar**: [Link to Lidar dataset](https://github.com/facebookresearch/generalized-schrodinger-bridge-matching?tab=readme-ov-file)
- **Single Cell**:
  - CITE and Multi: [Link to CITE and Multi datasets](https://data.mendeley.com/datasets/hhny5ff7yj/1)
  - EB: [Link to EB dataset](https://github.com/KrishnaswamyLab/TrajectoryNet/tree/master/data)
- **Animal Faces HQ (AFHQ)**: [Link to AFHQ dataset](https://github.com/clovaai/stargan-v2#animal-faces-hq-dataset-afhq)

## Running Experiments

All hyperparameters used for the experiments in the paper are located in the [`config`](./configs) folder, with specific definitions in [`mfm/train/parsers.py`](./mfm/train/parsers.py). To specify the data location, use the `--working_dir` flag. 

To specify the experiment to run use `--config_path` flag, for example:

```bash
python -m mfm.train.main --config_path ./configs/arch/ot-mfm.yaml
```


## Evaluation

For the `arch`, `sphere`, `single cell`, and `images` experiments, evaluation metrics will be logged after training. Plots for `arch`, `lidar`, and `sphere` will also be saved at the end of training in the `--working_dir` folder.

Model checkpoints are saved within the `checkpoints` folder under `--working_dir`. The `geopath` model can be loaded using the `--load_geopath_model_ckpt <checkpoint_path>` flag. Training and evaluation can be resumed from a flow model checkpoint using the `--resume_flow_model_ckpt <checkpoint_path>` flag.

CITE and Multi default to `~/Desktop/flow-maps-data`. Set
`CITE_MULTI_DATA_DIR` to use a different shared data directory.

### CITE/Multi PCA100 experiments

The PCA100 configurations run both leave-one-day-out experiments in sequence:
index 1 holds out day 3 and index 2 holds out day 4. For example, OT-MFM on
CITE is:

```bash
python -m mfm.train.main \
  --config_path configs/single_cell/100dims/ot-mfm_cite.yaml \
  --cite_multi_time_mode equal_time \
  --working_dir /path/to/output
```

Replace `cite` with `multi` for Multi, or use the corresponding `i-mfm_*`
configuration for independent MFM. The PCA100 configurations automatically
evaluate full-population, predecessor-to-omitted-day exact EMD (`test_EMD` and
`mfm/test_EMD`) and multi-bandwidth RBF MMD²
(`final_eval/euler_mean_rbf_mmd2`). The MMD uses the Maizels evaluator's median
bandwidth with multipliers 0.25, 0.5, 1, 2, and 4, computed in memory-bounded
blocks. They also follow the held-out 10% of day-2 cells from day 2 to day 7
with 50 Euler steps and score lineage violations using the dataset's
evaluation-only classifier:

* `../cite-classifiers/celltype_classifier_cite_pca100_all_days.pt`, or
* `../multi-classifiers/celltype_classifier_multi_pca100_all_days.pt`.

The CITE/Multi lineage permits self-transitions and differentiation from HSC
to any of BP, EryP, MasP, MkP, MoP, or NeuP. Final best-checkpoint metrics are
written as `final_eval/euler_mean_emd`, `final_eval/euler_mean_rbf_mmd2`, and
`final_eval/full_data_classifier/euler_invalid_trajectory_pct`. Set
`--cite_multi_eval_classifier_path` to override the classifier. A source cap of
zero uses every held-out day-2 cell exactly once.

Use `--cite_multi_time_mode equal_time|real_time` to select the global clock.
The default `equal_time` maps D2, D3, D4, D7 to `(0, 1/3, 2/3, 1)`, while
`real_time` normalizes elapsed days to `(0, 1/5, 2/5, 1)`. The same clock is
used for geopath training, flow matching, and held-out-day evaluation. The
grid is fixed before D3 or D4 is omitted; retained marginals are not re-spaced.

### Maizels PCA50 experiments

The original endpoint configuration trains on D3 -> D8 pairs and evaluates the
learned velocity rollout against every intermediate day. The CITE-50 MFM
architecture and metric defaults are used, with 10,000 optimizer steps each for
geopath and flow training.

```bash
python -m mfm.train.main \
  --config_path configs/single_cell/50dims/mfm_maizels.yaml \
  --working_dir /path/to/output \
  --maizels_dataset_path /path/to/celltype_classification_pca50_dataset.csv.gz \
  --maizels_classifier_path /path/to/celltype_classifier_pca50.pt \
  --maizels_pair_mode none
```

The three-marginal configurations observe D3, D3.8, and D8. They pass raw
marginal minibatches through MFM's original adjacent-interval loop, which fits
separate RBF metrics for D3 -> D3.8 and D3.8 -> D8. Independent MFM uses random
minibatch alignment; OT-MFM applies MFM's native exact minibatch OT separately
in each interval.

```bash
# Independent MFM
python -m mfm.train.main \
  --config_path configs/single_cell/50dims/i-mfm_maizels_3marginal.yaml \
  --working_dir /path/to/output \
  --maizels_dataset_path /path/to/celltype_classification_pca50_dataset.csv.gz

# OT-MFM
python -m mfm.train.main \
  --config_path configs/single_cell/50dims/ot-mfm_maizels_3marginal.yaml \
  --working_dir /path/to/output \
  --maizels_dataset_path /path/to/celltype_classification_pca50_dataset.csv.gz
```

Both configurations reserve 10% of each observed marginal. Held-out D3 ->
D3.8 and D3.8 -> D8 EMDs are logged under `validation_distribution/*`, including
their raw and source-to-target-normalized mean. The omitted-day metrics remain
enabled under `distribution_eval/*`; each omitted day is rolled out from the
left observed endpoint of its interval, so these remain test diagnostics rather
than model-selection criteria. Real experimental time is used by default
(`D3=0`, `D3.8=0.16`, `D8=1`).

After training, Lightning reloads the flow checkpoint with the lowest
`FlowNet/val_loss_cfm` and reruns these diagnostics. Its final test values are
also written to the W&B summary under `final_eval/*`, including
`final_eval/euler_mean_emd` and
`final_eval/euler_invalid_trajectory_pct`.

For the original endpoint configuration, available pair modes are:

- `none`: independent D3/D8 coupling;
- `ot_plain`: exact OT without biological filtering;
- `endpoint_interpolant`: learn the geopath from independent pairs, then filter
  independent candidates by endpoint lineage and 50 classifier checks along
  the frozen learned geodesic before velocity-field training;
- `ot_endpoint_interpolant`: learn the geopath from plain OT pairs, then solve
  OT on endpoint-compatible edges whose frozen learned geodesics pass the same
  classifier checks.

The biological pair prior therefore does not affect metric or geopath fitting;
it is introduced only when constructing the velocity-training pair pool.
If the learned-geodesic edge mask cannot support uniform balanced marginals,
MFM uses maximum-valid-mass partial OT: no rejected edge is restored, and the
largest transportable valid mass is renormalized for velocity training.

The Maizels runs log to `self-distill-flow-maps`, including all intermediate-day
RBF MMD and exact `test_EMD`-compatible W1 metrics, classifier-invalid Euler
trajectory percentage, PC1/PC2 plots, and the common
loss/gradient/learning-rate scalars.

On Apple Silicon, the metric/geopath phase automatically uses CPU because the
higher-order `torch.func.jvp` backward used by the time-conditioned geopath is
not supported by PyTorch MPS. The subsequent velocity-field phase still uses
the configured GPU/MPS accelerator.

### SF2M baseline

TorchCFM 1.0.5 supplies `SchrodingerBridgeConditionalFlowMatcher`, which is the
conditional bridge used by SF2M. The baseline here trains both required fields:
the probability-flow velocity and the score. Training uses TorchCFM's internal
minibatch coupling exactly once, with the default `exact` approximation from
the package and squared Euclidean ground cost. `sf2m_sigma` is a constant
diffusion on the global biological clock; its bridge variance and velocity are
rescaled correctly for unequal retained intervals.

The paper's geometric alternative is available with
`sf2m_ot_cost: geodesic` and `sf2m_ot_method: sinkhorn`. It implements Eq. 12,

```text
c_geo(x0, x1) = sqrt(-log H_t(x0, x1)),
```

so the squared cost passed to Sinkhorn is `-log H_t`. The heat kernel is
approximated from a k-nearest-neighbour graph over the retained **training**
cells using the low-frequency spectrum of a density-corrected graph Laplacian.
Omitted evaluation days and validation cells are not used to construct the
geometry. As in the paper, the entropic regularization is `2 * sigma^2` for
each local Brownian bridge; only endpoint coupling changes, while the Gaussian
bridge itself remains the same.

The paper does not prescribe numerical graph settings, so the supplied configs
make them explicit: `sf2m_geodesic_knn: 5`,
`sf2m_geodesic_heat_time: 1.0`, and
`sf2m_geodesic_eigenvectors: 256`. The default
`sf2m_geodesic_graph_max_points: 0` uses every retained training cell; set a
positive cap only if graph construction is too expensive. The spectrum is
cached under `<working_dir>/.sf2m_geodesic_cache` and reused on later runs.

```bash
# CITE-seq; replace cite with multi for Multiome.
python -m mfm.train.main \
  --config_path configs/single_cell/100dims/sf2m_cite.yaml \
  --working_dir ../outputs/sf2m_cite_pca100

# Maizels with observed D3, D3.8, and D8 marginals.
python -m mfm.train.main \
  --config_path configs/single_cell/50dims/sf2m_maizels_3marginal.yaml \
  --working_dir ../outputs/sf2m_maizels_pca50 \
  --maizels_dataset_path /path/to/celltype_classification_pca50_dataset.csv.gz
```

For the geometric variant, use the otherwise matching configs:

```bash
python -m mfm.train.main \
  --config_path configs/single_cell/100dims/sf2m_geodesic_cite.yaml \
  --working_dir ../outputs/sf2m_geodesic_cite_pca100

python -m mfm.train.main \
  --config_path configs/single_cell/100dims/sf2m_geodesic_multi.yaml \
  --working_dir ../outputs/sf2m_geodesic_multi_pca100

python -m mfm.train.main \
  --config_path configs/single_cell/50dims/sf2m_geodesic_maizels_3marginal.yaml \
  --working_dir ../outputs/sf2m_geodesic_maizels_pca50 \
  --maizels_dataset_path /path/to/celltype_classification_pca50_dataset.csv.gz
```

Use `sf2m_maizels.yaml` for the endpoint-only D3-to-D8 protocol. CITE and Multi
retain the same omitted-day splits and all-days classifiers as MFM. Maizels
retains its corresponding held-out and classifier evaluations. Maizels and
classifier-path evaluation use 50-step Euler--Maruyama rollouts; CITE/Multi
held-out-day EMD/MMD keeps MFM's 100-step distribution protocol. Runs log
sampler-specific keys such as
`final_eval/euler_maruyama_mean_emd`,
`final_eval/euler_maruyama_mean_rbf_mmd2`, and
`final_eval/euler_maruyama_invalid_trajectory_pct`. The existing
`final_eval/euler_*` keys are also populated as cross-method compatibility
aliases. For Maizels, `D3.4` and `D6` are reserved for hyperparameter
validation, matching the other Maizels methods. Their mean EMD is logged as
`distribution_eval/euler_maruyama_mean_emd_hparam_val_times` during training
and `final_eval/euler_maruyama_mean_emd_hparam_val_times` for the best
checkpoint; the remaining omitted days are aggregated under the corresponding
`*_mean_emd_test_times` keys. Training and validation additionally log separate
`SF2M/*_velocity_loss` and `SF2M/*_score_loss` values.

### Maizels hyperparameter sweeps

Two resumable grid runners select hyperparameters using the same held-out
Maizels days as the flow-map experiments: D3.4 and D6 by default. Each grid
setting is repeated over the requested seeds, evaluated from Lightning's best
validation-loss checkpoint, and ranked by mean exact EMD across those two
days. Per-run metrics are stored in `results.csv`; across-seed means, standard
deviations, and objective ranks are stored in `summary.csv`. Run these commands
from the top-level `flow-maps` directory in the MFM environment.

The SF2M sweep fixes the diffusion scale at `0.2` and the flow learning rate at
the CITE/Multi default of `1e-3`. It varies only the score-loss weight by
default (3 settings before seeds). Alternative values can still be requested
explicitly with `--sf2m-sigmas` and `--flow-learning-rates`:

```bash
python scripts/sweep_maizels_sf2m_hparams.py \
  --dataset-path /path/to/celltype_classification_pca50_dataset.csv.gz \
  --classifier-path /path/to/celltype_classifier_pca50_d3_d3p8_d8.pt \
  --seeds 0,1 \
  --wandb-mode disabled
```

To tune the Geodesic Sinkhorn version, add:

```bash
--config-path metric-flow-matching/configs/single_cell/50dims/sf2m_geodesic_maizels_3marginal.yaml
```

Its graph spectrum is shared across runs under the sweep output directory.

The MFM sweep fixes the flow learning rate at the CITE/Multi default of `1e-3`
and varies the RBF metric's adaptive `rho` and bandwidth multiplier `kappa` by
default (9 settings before seeds):

```bash
python scripts/sweep_maizels_mfm_hparams.py \
  --dataset-path /path/to/celltype_classification_pca50_dataset.csv.gz \
  --classifier-path /path/to/celltype_classifier_pca50_d3_d3p8_d8.pt \
  --seeds 0,1 \
  --wandb-mode disabled
```

The MFM runner also exposes grids for geopath learning rate, metric learning
rate, both weight decays, metric exponent, and RBF centre count. For example,
add `--geopath-learning-rates 0.00003,0.0001,0.0003` or
`--metric-learning-rates 0.003,0.01,0.03`. Select native OT-MFM by passing the
`ot-mfm_maizels_3marginal.yaml` config. Both runners accept `--dry-run`,
`--rerun-completed`, and runtime overrides such as `--max-steps`,
`--batch-size`, `--n-pairs`, and `--patience`.

Every subprocess receives a generated single-seed YAML, leaving the source
configuration unchanged. Final best-checkpoint metrics are also exported to
each run's `final_metrics.json`, so sweep collection does not depend on the
W&B API.

## Citation

If you find this repository helpful for your publications, please consider citing our paper:
```
@article{kapusniak2024metric,
  title={Metric Flow Matching for Smooth Interpolations on the Data Manifold},
  author={Kapusniak, Kacper and Potaptchik, Peter and Reu, Teodora and Zhang, Leo and Tong, Alexander and Bronstein, Michael and Bose, Avishek Joey and Di Giovanni, Francesco},
  journal={arXiv preprint arXiv:2405.14780},
  year={2024}
}
```

## Files Structure
```
mfm
├── dataloaders
│   ├── image_data.py
│   ├── lidar_data.py
│   └── trajectory_data.py
├── flow_matchers
│   ├── ema.py
│   ├── eval_utils.py
│   ├── flow_net_train.py
│   ├── geopath_net_train.py
│   └── models
│       └── mfm.py
├── geo_metrics
│   ├── land.py
│   ├── metric_factory.py
│   └── rbf.py
├── networks
│   ├── flow_networks
│   │   └── mlp.py
│   ├── geopath_networks
│   │   ├── mlp.py
│   │   └── unet.py
│   ├── mlp_base.py
│   ├── unet_base.py
│   └── utils.py
├── train
│   ├── main.py
│   ├── parsers.py
│   └── train_utils.py
└── utils.py
```
