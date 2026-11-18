# Calderon flow experiment with Maizels defaults

The current benchmark trains on **4–6, 8–10 and 12–14 h**, holding out **6–8 and 10–12 h**. Plain OT, interpolant-filtered OT and constrained filtered OT use the Maizels training, validation and logging cadence. Filtered variants now use [batched GPU interpolant checks](gpu_filtering.md); duplicate post-update constraint diagnostics are disabled. Evaluation reports composed-flow-map and Euler W1 on subsets of at most **2,048 source and 2,048 reference cells**. MMD, direct and linear evaluation are disabled, including direct diagnostic plots. The [defaults comparison](maizels_defaults_comparison.md) records these overrides and the remaining dataset-specific differences. [Earlier shared-split defaults](three_time_flow_before_maizels.md) and their saved runs are historical.

## Flow populations

Flow cells now use an independent **90% training / 10% validation** split at each selected window. The same held-out fraction supplies validation pairs and first-window violation sources, as in Maizels; there is no separate flow test-cell fold. The split uses Maizels's helper and holdout seed 701, with offsets 11/303/29 for the first/middle/final windows. Changing the flow initialization seed does not change the holdouts.

| Inferred-age window | Normalized midpoint | Flow training | Flow validation / heldout | Intermediate evaluation |
| --- | ---: | ---: | ---: | ---: |
| 4–6 h | 0 | 47,471 | 5,275 | — |
| 6–8 h | 0.25 | 0 | 0 | 30,876 |
| 8–10 h | 0.5 | 52,204 | 5,800 | — |
| 10–12 h | 0.75 | 0 | 0 | 21,448 |
| 12–14 h | 1 | 46,310 | 5,146 | — |

There are **145,985 flow training cells and 16,221 flow validation cells**. Entire intermediate marginals are excluded from fitting and checkpoint selection. All lineage-assigned cells within 4–14 h are included, including Yolk; windows outside this range are excluded.

The source remains `datasets/calderon2022/all_times/lineage_assigned_hvg3000_pca100_all_times.h5ad`: its HVG3000/PCA100 fit includes all 314,387 lineage-assigned expression profiles. The flow's scalar rescale still uses the unique union of its fitting cells. Classifiers retain their existing frozen scalers, weights and supervised splits.

## Variants and validation

| Config | Maizels recipe | Training and validation pairs | Training and validation objective |
| --- | --- | --- | --- |
| `configs.calderon_pca100` | ID 7 | Plain minibatch OT | Native LSD |
| `configs.calderon_filtered_pca100` | ID 5 | Endpoint mask and 50-point interpolant filtering | Native LSD |
| `configs.calderon_constrained_pca100` | ID 6 | Same filtered OT | Native LSD plus `loss_points_nll`, weight 1000 |
| `configs.calderon_constrained_w2000_pca100` | ID 6 | Same filtered OT | Same objective, weight 2000 |

The default 128-pair training batch contains equal pair counts from 4–6 → 8–10 and 8–10 → 12–14. Pair labels contain `[source_tissue_id, target_tissue_id, t_start, t_end]`, allowing the native time-rescaled interpolant and loss helpers to map local times to the correct global interval. The flow remains unconditional on tissue labels.

Validation now reuses **Maizels's native fixed 1,024-pair batch construction and validation loss**. It uses the same pairing/filtering and objective as the corresponding training variant, including the lineage penalty for the constrained variants. Each check evaluates one fixed batch, with instantaneous flow parameters. Intermediate W1 and violation scores do not select a checkpoint. Early stopping remains patience 10, checked every 100 updates; final evaluation reloads `best.msgpack`.

The optimizer remains RAdam at learning rate 0.0003, batch 128, diagonal fraction 0.75, clipping 10, square-root schedule and 50,000 maximum updates. The flow MLP retains `n_hidden=2`, width 1,024 and GELU, with 100 input/output dimensions. Both EMA decays are still tracked, but validation, plots and final flow evaluation use **instantaneous parameters**. Plain, filtered and constrained OT all use Maizels's `partial` fallback and 20 resampling attempts.

## Classifiers and transitions

Training filtering and the differentiable loss continue to use the [selected-times classifier](three_time_classifier.md), `calderon-classifiers/symmetric_4_8_12/models/symmetric_4_8_12_broad_tissue_selected.json`. Its two-member tissue ensemble has 23 outputs and the existing JAX export at `.cache/calderon2022/symmetric_4_8_12_tissue_ensemble_jax.npz`.

The filtered and constrained defaults set `problem.interpolant_filter_backend="jax"` and `problem.batch_interpolant_checks_across_intervals=True`. Interpolation, preprocessing, ensemble inference and transition checks run together on the device. Independent CPU OT loops submit their pending checks together; their sampling keys, assignments, refinements and partial-OT fallbacks are preserved. Every checked path still evaluates all 50 interior points and the observed final tissue label. Paths with nearly tied predictions are rechecked using the original decoder; this chooses the inference implementation without dropping uncertain paths. Add `--interpolant-filter-backend reference` to use the previous filtering pipeline.

Final violation tracking continues to use the [all-times oracle](all_times_classification.md), `calderon-classifiers/all_times/models/all_times_broad_tissue_selected.json`, with three members and 24 outputs. The launcher copies its frozen weights to `.cache/calderon2022/all_times_tissue_ensemble_jax.npz` when needed for native Maizels diagnostics. This export does not fit a classifier. Labels are aligned by tissue name, including the oracle's additional Blastoderm output.

The supplied `broad_tissue_allowed_pairs.csv` projects branch self/descendant transitions onto tissues. Training restricts this relation to its 23-label vocabulary; the oracle uses all 24 labels. Neither operation adds another transitive closure. Flow splits are independent of the classifiers' existing splits.

## Composed evaluation

Violation evaluation samples **512 first-window held-out cells without replacement**, with Maizels seed 2698. It checks the full first-to-last trajectory at **50 evenly spaced points**, `0.02, 0.04, …, 1`, obtained from a 50-step composed rollout. Every classifier argmax label is checked: the reported violation result is **raw only**, with probability and margin thresholds both zero.

W1 evaluation uses the **native Maizels composed protocol**, independently of that violation trajectory. Each comparison samples **at most 2,048 source and 2,048 reference cells without replacement** from the full marginals; every selected source is propagated once. Sampling is reproducible using the native evaluation seed (training seed + 1701). Euler W1 on the two intermediate marginals uses the same source/reference subsets as composed W1. MMD, linear and direct comparisons are disabled, leaving seven exact OT calculations in final evaluation: four intermediate comparisons and the three observed/complete-rollout comparisons below.

| W1 comparison | Starting population | Target population | Composed steps |
| --- | --- | --- | ---: |
| `heldout_6_8` | 2,048 observed 4–6 h cells, t=0 | 2,048 6–8 h cells, t=0.25 | 50 |
| `heldout_10_12` | 2,048 observed 8–10 h cells, t=0.5 | 2,048 10–12 h cells, t=0.75 | 50 |
| `observed_4_6_to_8_10` | 2,048 observed 4–6 h cells | 2,048 observed 8–10 h cells | 50 |
| `observed_8_10_to_12_14` | 2,048 observed 8–10 h cells | 2,048 observed 12–14 h cells | 50 |
| `source_to_final` | 2,048 observed 4–6 h cells, carried through both intervals | 2,048 observed 12–14 h cells | 50 + 50 |

Thus the source-to-final W1 rollout has 100 updates **because Maizels chains 50 updates per retained interval**. Violation tracking separately uses 50 updates across the entire window. Quarter-time W1 queries require no shared global composition grid.

W1 is exact empirical OT between the selected subsets, with Euclidean ground cost in the original PCA100 coordinates. A 2,048-by-2,048 comparison has about 4.2 million cost entries. `--marginal-cap` changes both source and reference caps; zero opts back into uncapped W1. The shared disk-backed cost construction remains available for larger comparisons, while Calderon no longer computes MMD.

## Logging, commands and artifacts

Logging uses Maizels's native scalar, plot and final-evaluation functions: scalars/progress every update, visual diagnostics every 5,000 updates, and checkpoint saving every 5,000 updates. `logging.constraint_metrics_enabled=False` skips the additional post-update flow/classifier pass for `constraint/*` diagnostics. Training and validation still compute the full constrained objective. Evaluation and plots omit direct trajectories and one-step sampling; composed and Euler sampling remain enabled. The Maizels W&B project/entity defaults are inherited; the launcher uses normal W&B logging and honors the existing `WANDB_MODE` environment setting. Use `--wandb offline` or `--wandb disabled` explicitly when desired.

From the repository root:

```sh
ENABLE_PJRT_COMPATIBILITY=1 JAX_PLATFORMS=METAL,cpu \
  .venv-flowmaps-metal/bin/python py/launchers/calderon_ot_flow_map.py \
  --config configs.calderon_pca100 \
  --classifier-python /opt/anaconda3/envs/maizels2023aa/bin/python
```

Select `configs.calderon_filtered_pca100`, `configs.calderon_constrained_pca100` or `configs.calderon_constrained_w2000_pca100` for the other variants. The CPU interpreter remains `/opt/anaconda3/envs/flowmaps/bin/python`.

The directories under `outputs/` are `calderon_symmetric_4_8_12_maizels_ot_flow_map_seed0`, `calderon_symmetric_4_8_12_maizels_filtered_ot_flow_map_seed0`, and `calderon_symmetric_4_8_12_maizels_constrained_filtered_ot_flow_map_seed0`; the weight-2000 variant adds `_w2000`. Previously saved runs reload their original configurations and protocols. Add **`--reduced-evaluation`** when evaluating a saved Maizels-protocol Calderon run to apply the current evaluation settings; this preserves its training config and checkpoint. For example:

```sh
ENABLE_PJRT_COMPATIBILITY=1 JAX_PLATFORMS=METAL,cpu \
  .venv-flowmaps-metal/bin/python py/launchers/calderon_ot_flow_map.py \
  --stage evaluate --reduced-evaluation \
  --output-dir outputs/calderon_symmetric_4_8_12_maizels_ot_flow_map_seed0 \
  --evaluation-output-dir outputs/calderon_symmetric_4_8_12_maizels_ot_flow_map_seed0/evaluations/w1_2048_reduced \
  --classifier-python /opt/anaconda3/envs/maizels2023aa/bin/python
```

`config.json`, `data_summary.json` and `cell_splits.csv.gz` record flow splits and classifier provenance. `training_history.csv`, native Maizels `.pkl` checkpoints and `best.msgpack`/`last.msgpack` record training. `metrics.json` includes five composed W1 comparisons and raw oracle violations; `maizels_evaluation_metrics.json` also contains the intermediate Euler W1 mean. Individual intermediate Euler scores are logged to W&B. Each composed W1 comparison saves its actual source/generated/reference arrays in `evaluation/*_populations.npz`, allowing exact score recomputation. Violation coordinates, IDs, probabilities and raw decisions are saved separately. `evaluation/evaluation_config.json` records the effective evaluation overrides.

A short CPU check can retain the full validation batch and 50 checking points while capping only its smoke-test populations:

```sh
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  /opt/anaconda3/envs/flowmaps/bin/python py/launchers/calderon_ot_flow_map.py \
  --smoke --validation-bs 1024 --checking-points 50 --flowmap-steps 50 \
  --classifier-python /opt/anaconda3/envs/maizels2023aa/bin/python --wandb disabled
```

Focused protocol checks are in `tests/test_calderon_maizels_flow.py`; verification artifacts are recorded under `.cache/calderon2022/maizels_protocol_*`.

The [GPU filtering report](gpu_filtering.md) records matched OT assignments across 12 minibatches, 76 passing CPU tests, seven additional Metal checks and a constrained training smoke with the full validation batch. In its local benchmark, pair generation fell from 0.397 to 0.326 seconds per 128-pair batch. These timings measure sampling/filtering rather than the complete optimizer update.

Verification passed **76 tests** (44 Calderon checks and 32 Maizels regressions), including checks that removed evaluations never run, both W1 populations are capped, and saved-run overrides preserve the training config. Native plotting produced five diagnostics with direct and MMD calls guarded. Evaluating the existing plain-OT seed-0 checkpoint at step 2,000 on the Apple GPU completed in **21 seconds**, using 2,048-cell W1 populations and all 512 violation sources at 50 checkpoints. Its five composed W1 scores were reproduced from saved population arrays. The training config, checkpoint, training history and data-summary hashes were unchanged. The [current verification report](../../.cache/calderon2022/reduced_evaluation_validation.json) records these checks; the [initial alignment report](../../.cache/calderon2022/maizels_protocol_validation.json) retains the earlier smoke runs and frozen-oracle agreement checks.
