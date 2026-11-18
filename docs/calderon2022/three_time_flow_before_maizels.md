# Historical defaults before Maizels alignment

These notes describe the earlier shared-classifier-split protocol. See [current flow defaults](three_time_flow_map.md) and [current comparison](maizels_defaults_comparison.md). Existing runs retain their saved configurations.


The Calderon flow configs now train on **4–6, 8–10 and 12–14 h**, with **6–8 and 10–12 h** reserved for intermediate W1 evaluation. All lineage-assigned cells in this 4–14 h interval are included, including Yolk. The retained interval contains 23 broad tissue labels. The earlier `Ubi` endpoint experiments are documented in [the historical pilot](flow_map_example.md).

## Populations and clock

The flow reuses the exact [selected-times classifier](three_time_classifier.md) cell splits, stratified by time window and branch ID. Its fitting cells and the classifier's fitting cells coincide; validation and test cells are separate from fitting. The flow split seed is fixed with the already trained classifier, while the flow initialization seed remains configurable.

| Inferred-age window | Normalized midpoint | Flow training | Flow validation | Flow test / held-out marginal |
| --- | ---: | ---: | ---: | ---: |
| 4–6 h | 0 | 42,196 | 5,275 | 5,275 test starting cells |
| 6–8 h | 0.25 | 0 | 0 | 30,876 held-out marginal cells |
| 8–10 h | 0.5 | 46,401 | 5,804 | 5,799 test cells |
| 10–12 h | 0.75 | 0 | 0 | 21,448 held-out marginal cells |
| 12–14 h | 1 | 41,167 | 5,142 | 5,147 test cells |

The three training windows provide **129,764 training / 16,221 validation / 16,221 test cells**. Both entire intermediate marginals are excluded from flow fitting, validation and checkpoint selection. Windows outside 4–14 h are excluded from the flow benchmark.

The feature source is `datasets/calderon2022/all_times/lineage_assigned_hvg3000_pca100_all_times.h5ad`. The existing HVG3000/PCA100 fit includes all 314,387 lineage-assigned expression profiles across the atlas and is reused without refitting. The flow's scalar rescale uses only the union of its three training-cell pools. Classifiers apply their own frozen PC standardizers.

## Flow variants

| Config | Maizels recipe | Training pairs | Lineage penalty |
| --- | --- | --- | --- |
| `configs.calderon_pca100` | ID 7 | Plain minibatch OT | Disabled |
| `configs.calderon_filtered_pca100` | ID 5 | Endpoint mask and 50-point interpolant filtering | Disabled |
| `configs.calderon_constrained_pca100` | ID 6 | Same filtered OT | Native `loss_points_nll`, weight 1000 |
| `configs.calderon_constrained_w2000_pca100` | ID 6 | Same filtered OT | Native `loss_points_nll`, weight 2000 |

Every optimizer batch assigns equal pair counts to **4–6 → 8–10** and **8–10 → 12–14** and shuffles the combined pairs. The default 128-pair batch contains 64 pairs from each interval. Each interval independently recomputes minibatch OT using squared Euclidean PCA distance and the existing Maizels/CITE solver.

Pair labels contain `[source_tissue_id, target_tissue_id, t_start, t_end]`. The native time-rescaled linear interpolant and loss-argument helpers map local samples to the correct global half-interval. The flow MLP remains unconditional on tissue labels. Native LSD, convex stopgrad, 75% diagonal sampling, RAdam, hidden-layer settings, tracked EMA decays and stopping counts are preserved. Validation and final sampling select EMA 0.999, whereas Maizels's automatic evaluation uses instantaneous parameters. The [defaults comparison](maizels_defaults_comparison.md) records all protocol differences.

All variants select checkpoints using **fixed plain-OT validation batches over both adjacent intervals**, with interpolant filtering and the lineage penalty disabled in validation. Neither intermediate W1 nor violation scores select a checkpoint. `best.msgpack` is reloaded for final evaluation.

## Separate classifier roles

| Role | Config field | Selected manifest |
| --- | --- | --- |
| Training filter and differentiable loss | `classifier.manifest_path` | `calderon-classifiers/symmetric_4_8_12/models/symmetric_4_8_12_broad_tissue_selected.json` |
| Final violation tracking | `classifier.evaluation_manifest_path` | `calderon-classifiers/all_times/models/all_times_broad_tissue_selected.json` |

The training classifier is the frozen **residual with mixup/EMA + joint branch/tissue ensemble**, fitted only on selected-time training cells. Its two tissue heads, scalers and calibration are exported to `.cache/calderon2022/symmetric_4_8_12_tissue_ensemble_jax.npz`. The export is shared by filtering and the differentiable native lineage loss. Exporting copies existing weights and does not train a classifier. Re-exporting identical weights preserves the export and verification files byte-for-byte.

The training classifier has **23 outputs**; the all-times reference has **24**, including `Blastoderm`. Observed labels are aligned by tissue name when scoring with the reference. The training mask restricts `broad_tissue_allowed_pairs.csv` to the 23 training outputs; the evaluator uses all 24 reference outputs. Neither operation adds another transitive closure. `Unknown` retains its supplied transitions.

The all-times classifier is intentionally allowed supervised information from all ages. Real-cell reference scores describe its agreement with annotations on the evaluation populations; its independently held-out accuracy is reported in [the all-times classifier experiment](all_times_classification.md).

## Evaluation

Default evaluation propagates **all 5,275 test starting cells** through **100 composed flow-map steps**, once per cell. The reference classifier checks **50 points**, at `t=0.02, 0.04, …, 1.00`. Raw and probability ≥0.9 confidence-filtered violations retain the existing path scan and coverage semantics, starting from each cell's released tissue label.

The 100-step rollout is an explicit deviation from Maizels's 50-step default: it places both quarter-time W1 snapshots and all 50 classifier checks on one uniform composition grid. A 50-step uniform grid would miss W1 times 0.25 and 0.75. This is an evaluation choice; the training update recipe does not require 100 steps.

W1 extracts states from that same uninterrupted composed trajectory:

| Metric key | Window | Composed step | Reference population |
| --- | --- | ---: | --- |
| `heldout_6_8` | 6–8 h | 25 | Entire held-out marginal |
| `training_time_test_8_10` | 8–10 h | 50 | Selected-time test cells |
| `heldout_10_12` | 10–12 h | 75 | Entire held-out marginal |
| `target_test` | 12–14 h | 100 | Selected-time test cells |

The W1 snapshots at 0.25 and 0.75 are extracted independently of the 50-point classifier grid, which does not contain those two times. They do not add violation checks. W1 uses exact empirical OT with Euclidean PCA-space cost, capped by default at 2,048 generated and reference samples per comparison. Available and sampled population sizes are recorded.

## Commands and artifacts

From the repository root, export the training classifier when preparing a fresh checkout:

```sh
/opt/anaconda3/envs/maizels2023aa/bin/python scripts/export_calderon_tissue_ensemble.py
```

The three full runs use the existing launcher and config names:

```sh
/opt/anaconda3/envs/flowmaps/bin/python py/launchers/calderon_ot_flow_map.py \
  --classifier-python /opt/anaconda3/envs/maizels2023aa/bin/python

/opt/anaconda3/envs/flowmaps/bin/python py/launchers/calderon_ot_flow_map.py \
  --config configs.calderon_filtered_pca100 \
  --classifier-python /opt/anaconda3/envs/maizels2023aa/bin/python

/opt/anaconda3/envs/flowmaps/bin/python py/launchers/calderon_ot_flow_map.py \
  --config configs.calderon_constrained_pca100 \
  --classifier-python /opt/anaconda3/envs/maizels2023aa/bin/python
```

For the Apple GPU, replace the flow interpreter with `ENABLE_PJRT_COMPATIBILITY=1 JAX_PLATFORMS=METAL,cpu .venv-flowmaps-metal/bin/python`. Use `configs.calderon_constrained_w2000_pca100` for weight 2000. `--stage prepare` saves populations and config without flow training. `--smoke --checking-points 50 --flowmap-steps 100` performs a two-update check with a small network and capped evaluation.

Defaults write separate directories under `outputs/`: `calderon_symmetric_4_8_12_ot_flow_map_seed0`, `calderon_symmetric_4_8_12_filtered_ot_flow_map_seed0`, and `calderon_symmetric_4_8_12_constrained_filtered_ot_flow_map_seed0`. The weight-2000 config adds `_w2000`. A completed training directory cannot be overwritten by starting a new run. Existing endpoint checkpoints remain reloadable through their saved configs.

`config.json`, `data_summary.json` and `cell_splits.csv.gz` record both classifier roles, source PCA/cache/split hashes and all five populations. `pair_filtering_summary.json` reports filter support and tissue sampling counts for each adjacent interval. `metrics.json` and `evaluation/` contain four W1 comparisons, reference-classifier violation scores, predictions and real-cell diagnostics. `--classifier-manifest` and `--classifier-export` override the training artifacts; `--evaluation-classifier-manifest` overrides the reference when creating a run.

The saved evaluation inputs also contain `marginal_generated_x` and `marginal_step_indices`, so every W1 comparison can be recomputed directly from the generated states at steps 25/50/75/100 while retaining exactly 50 violation checks.

Run the focused checks with:

```sh
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  /opt/anaconda3/envs/flowmaps/bin/python -m unittest discover \
  -s tests -p 'test_calderon_*flow*.py'
```

The frozen JAX export agrees with PyTorch on all **13,568 training-cell/interpolant verification points**, with identical argmax and ≥0.9 confidence decisions and maximum probability error **9.54×10⁻⁷**. The native lineage penalty matches its explicit probability formula; its directional gradient has finite-difference relative error **1.91×10⁻⁴**. These CPU checks are recorded in `.cache/calderon2022/symmetric_4_8_12_tissue_ensemble_jax_cpu_validation.json` and `symmetric_4_8_12_lineage_gradient_cpu_validation.json`.

All **35 focused tests passed**. Each of the three main variants completed a genuine two-update CPU run, including both training intervals, fixed plain-OT validation, saved-checkpoint reload, 100 composed steps, four W1 comparisons and all-times reference scoring at 50 checks. A repeated evaluation reproduced all 50 original coordinates and reference probabilities exactly; all four W1 scores were independently recomputed from saved snapshots. Classifier weights, the original classifier-training implementations and source PCA/cache files were verified unchanged. The [default-experiment audit](../../.cache/calderon2022/three_time_flow_defaults_validation.json) records these checks. The new interval-batch permutation was also checked on the Apple M1 Pro GPU backend.
