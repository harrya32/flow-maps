# Calderon OT flow-map endpoint pilot

This page records the historical two-endpoint `Ubi` pilot and its saved results. **Current configs use the [three-time default experiment](three_time_flow_map.md)**, with separate training and all-times evaluation classifiers. The commands below that create new runs now use those current defaults; saved endpoint runs remain evaluable with `--stage evaluate --output-dir`.

The [dedicated launcher](../../py/launchers/calderon_ot_flow_map.py) and [config](../../py/configs/calderon_pca100.py) implement the proposed **4–6 → 12–14 h `Ubi`** endpoint experiment, with **8–10 h reserved for interpolation evaluation**. The flow is unconditional and uses plain minibatch OT; the tissue classifier and allowed-transition relation are used for final evaluation only.

The full default seed-0 run is complete: early stopping ended at **7,000 steps**, and endpoint validation selected **step 6,000**. See [results, controls and figures](ot_flow_map_results.md). Raw path violations were **54.66%**; the confidence-filtered rate was **22.02% among assessed paths**, with **43.16% assessed-path coverage** and **2.61% all-four-checkpoint coverage**.

A subsequent [50-checkpoint evaluation](ot_flow_map_50_checkpoints.md) of the same saved trajectories reports **63.74% raw violations** and **16.37% filtered violations among assessed paths**, with **60.14% assessed-path coverage** and **1.85% all-50-checkpoint coverage**. Both marginal W1 scores are unchanged.

The same launcher also supports the [filtered-interpolant config](../../py/configs/calderon_filtered_pca100.py). Its [method and reproduction notes](filtered_flow_map.md) describe the Maizels ID 5 training filter, verified frozen classifier, partial transport mass and matched endpoint-validation protocol. It evaluates the resulting flow with 50 checks and the same 100-step composed sampler.

The filtered run is complete: **4,700 updates**, selected **step 3,700**. [Its matched results and figure](filtered_ot_flow_map_results.md) show **58.31% raw violations** and **11.61% confidence-filtered violations among assessed paths**, with **58.28% assessed-path coverage**. W1 is **8.4856** at the held-out middle and **11.1041** at the test endpoint. Both W1 scores are slightly worse than plain OT; the report records the retained coupling mass and tissue-level breakdown.

The [constrained and filtered config](../../py/configs/calderon_constrained_pca100.py) implements Maizels ID 6 on the same endpoint data. [Its method notes](constrained_flow_map.md) document the frozen differentiable tissue ensemble, native weight-1000 transition penalty, unchanged 50-point interpolant filter, and shared plain-OT checkpoint-selection objective. Final evaluation uses 50 checks and 100 composed steps.

Its completed run selected **step 2,500** after **3,500 updates**. [The matched three-variant comparison](constrained_ot_flow_map_results.md) gives **36.39% raw violations**, **13.48% filtered violations among assessed paths**, **86.81% assessed-path coverage** and **3.75% all-50-checkpoint coverage**. W1 is **8.7176** at the held-out middle and **13.2912** at the test endpoint. The strong constraint reduces raw violations but worsens both W1 scores. Among named-tissue starts, filtered violations improve to **29.92%** with **81.18% coverage**; the report explains why the aggregate filtered rate moves in the other direction as coverage changes.

## Populations and clock

| Inferred-age window | Flow train | Flow validation | Flow test / reference |
| --- | ---: | ---: | ---: |
| 4–6 h | 37,128 | 4,641 | 4,641 test starting cells |
| 8–10 h | 0 | 0 | 48,630 reference cells |
| 12–14 h | 35,413 | 4,427 | 4,427 test target cells |

The loader retains branches beginning `Ubi_`. It uses a uniform random 80/10/10 cell split at each endpoint, with deterministic seeds 701 and 702. Validation and test endpoints are disjoint. It saves every selected cell's ID, original PCA row, labels and split to `cell_splits.csv.gz`. The middle window never enters flow-training pairs or checkpoint selection.

The bin midpoints **5, 9 and 13 h** map to **0, 0.5 and 1**. Trajectory labels are checked at normalized times **0.25, 0.5, 0.75 and 1**, corresponding to **7, 9, 11 and 13 h**. These windows are inferred developmental ages, as described in the feasibility report.

## Native Maizels training recipe

The config takes the training, optimization and network settings from **Maizels variant 7, `ot_flow_map`**, and changes the feature dimension to 100. It uses the same `common.state_utils`, `common.interpolant`, `common.loss_args`, `common.losses`, `common.updates` and `common.flow_map` implementations. The new endpoint sampler calls the same plain-OT solver used underneath the Maizels adapter, supplying the Calderon class vocabulary.

- Each optimizer batch independently samples source and target training cells and recomputes exact minibatch OT with **raw squared Euclidean PCA distance**. No tissue mask, classifier rejection, lineage penalty or tissue conditioning enters training.
- LSD self-distillation, convex stopgrad and **75% diagonal samples** are retained.
- The MLP retains `n_hidden=2` and **1,024 units**, GELU activations and 100 outputs. In the repository's implementation this gives an initial hidden layer plus two further hidden layers. RAdam uses learning rate **0.0003** and gradient clipping at 10; batch size is **128**.
- The maximum is **50,000 steps**. Every 100 steps, fixed endpoint-validation OT batches/times measure the loss. Eight batches provide 1,024 validation pairs. Early stopping permits 10 checks without improvement.
- Checkpoint selection evaluates the **0.999 EMA**, which is also used for final sampling. `best.msgpack` is reloaded before final evaluation. Checkpoint hashes and training step are included in the results.

The loader reads `obsm['X_pca']` and metadata from `lineage_assigned_hvg3000_pca100_heldout_8_10.h5ad`; it never loads the full RNA count matrix or refits HVGs/PCA. Coordinates remain in the original PCA space. As in Maizels, the flow network internally uses one scalar rescale estimated from endpoint training cells; generated outputs stay in the original coordinate system. The tissue ensemble applies its own stored PC scaler when classifying them.

## Evaluation

The default evaluates **all 4,641 test source cells**, pushing each forward once with **100 composed flow-map steps**. All checkpoint predictions belong to the same uninterrupted trajectory. Neither source nor target training cells are used for the final population comparisons.

`--checking-points 50` changes the classifier grid to t=0.02, 0.04, …, 1.00 while retaining 100 composition steps. The check count must be even to include t=0.5, and the composition grid must contain every check exactly. `--evaluation-output-dir` saves a separate evaluation under the existing trained run and preserves its original metrics.

At t=0.5, predictions are compared with the reserved 8–10 h marginal. At t=1, they are compared with target **test** cells. The metric is the repository's exact empirical **W1 with Euclidean PCA-space ground cost**. It reproducibly samples at most 2,048 generated and reference points per comparison to keep the dense OT calculation manageable. Available and evaluated population sizes are reported; the full 8–10 h population remains reserved.

The frozen validation-selected tissue ensemble is `heldout_8_10_broad_tissue_selected.json`. The scorer begins from the starting cell's **released tissue label**, then checks successive predicted tissues against `broad_tissue_allowed_pairs.csv`. It loads that projected relation directly, without adding another transitive closure after tissue merging. `Unknown` keeps its supplied allowed pairs. A separate endpoint score checks the start-to-final relation.

Two path scores are reported:

- **Raw:** all four argmax tissue labels are checked. `raw.violation_pct_all_sources` is the percentage of trajectories with at least one forbidden transition, matching the existing Maizels scan semantics.
- **Confidence filtered:** the default requires top probability ≥0.9 and probability margin ≥0.0. Uncertain checkpoints are skipped while retaining the last assessed tissue. Results include violation percentage among assessed paths, coverage of paths/checkpoints, and the percentage among paths whose four checkpoints are all assessed. A path without a confident checkpoint has status `unassessed`; its group's assessed violation rate is JSON `null` when there are no assessed paths.

The output also includes first rejection by checkpoint, scores by starting tissue, per-cell labels/probabilities, and classifier accuracy/coverage on the actual test source and sampled middle/target reference cells. The classifier's previous accuracy on the full held-out 8–10 h window was only **53.25%**, so these remain classifier-based diagnostics of agreement with the reconstructed prior.

The endpoint holdouts apply to **flow training**. The existing reference-atlas HVG/PCA fit and tissue classifier can have seen endpoint flow-test cells. This is the previously declared endpoint/reference-atlas protocol; it does not provide an independent-embryo or clone-fate test.

## Run

From the repository root, use a Python environment with the native flow-map dependencies (the local `flowmaps` environment). The classifier worker can use a separate Python with the already used PyTorch/scientific dependencies. Local examples:

```sh
# Inspect all three populations and save the split/config without training.
/opt/anaconda3/envs/flowmaps/bin/python py/launchers/calderon_ot_flow_map.py \
  --stage prepare \
  --classifier-python /opt/anaconda3/envs/maizels2023aa/bin/python

# Train and evaluate the full default example.
/opt/anaconda3/envs/flowmaps/bin/python py/launchers/calderon_ot_flow_map.py \
  --classifier-python /opt/anaconda3/envs/maizels2023aa/bin/python

# A separate two-step execution check with a small network and capped evaluation.
/opt/anaconda3/envs/flowmaps/bin/python py/launchers/calderon_ot_flow_map.py \
  --smoke \
  --classifier-python /opt/anaconda3/envs/maizels2023aa/bin/python

# Repeat evaluation from the saved best native checkpoint.
/opt/anaconda3/envs/flowmaps/bin/python py/launchers/calderon_ot_flow_map.py \
  --stage evaluate --output-dir outputs/calderon_ubi_ot_flow_map_seed0 \
  --classifier-python /opt/anaconda3/envs/maizels2023aa/bin/python

# Evaluate 50 checks using the original Apple GPU backend.
ENABLE_PJRT_COMPATIBILITY=1 JAX_PLATFORMS=METAL,cpu \
  .venv-flowmaps-metal/bin/python py/launchers/calderon_ot_flow_map.py \
  --stage evaluate --output-dir outputs/calderon_ubi_ot_flow_map_seed0 \
  --evaluation-output-dir outputs/calderon_ubi_ot_flow_map_seed0/evaluations/checkpoints50 \
  --checking-points 50 --classifier-python /opt/anaconda3/envs/maizels2023aa/bin/python
```

When one interpreter has both dependency sets, the worker can use that interpreter and `--classifier-python` can be omitted. The launcher verifies the frozen ensemble can load and predict before starting training. `--dataset-location`, `--classifier-manifest`, `--output-dir`, `--steps`, `--batch-size`, `--learning-rate`, `--flowmap-steps`, `--eval-source-cap`, `--marginal-cap` and confidence thresholds support overrides; `--help` lists them. A composed-step count must include the four checkpoint times exactly. A completed training directory is preserved: use a different output directory for another run.

The default output directory is `outputs/calderon_ubi_ot_flow_map_seed0`. W&B is disabled by default; `--wandb offline` or `--wandb online` opts into tracking. The dedicated launcher handles this config's data and classifier integration.

## Artifacts and verification

- `config.json`, `data_summary.json`, `cell_splits.csv.gz`: complete settings, input hashes and populations.
- `training_history.csv`, `training_summary.json`, `best.msgpack`, `last.msgpack`: optimization and checkpoint selection.
- `metrics.json`: both marginal W1 scores, trajectory diagnostics and provenance.
- `evaluation/`: evaluation coordinates, per-cell tissue predictions, first rejections, confidence coverage, source-tissue strata and frozen-classifier checks on real cells.
- `evaluation/controls/`: unchanged-source trajectory diagnostics and disjoint real-versus-real W1 controls. [The reporting script](../../scripts/report_calderon_ot_flow_map.py) reproduces the results document and scientific figures from the completed run.

The focused unit checks cover data/representation alignment, disjoint splits, label-independent OT, native training settings, continuous composed rollouts, sibling switching/backtracking, agreement with the Maizels checker, confidence abstention and the projected tissue relation. Run them with:

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 /opt/anaconda3/envs/flowmaps/bin/python \
  -m unittest discover -s tests -p 'test_calderon_flow_map.py' -v
```

All **23 focused tests passed**, including dense-grid validation, identical composed rollouts at shared checkpoints, the filtered-interpolant extension and the constrained adapter. The local two-step smoke run exercises genuine OT pairing, LSD updates, best-checkpoint reload, both W1 comparisons and the real frozen tissue ensemble. Repeating the smoke evaluation from its saved checkpoint reproduced the trajectory coordinates and classifier probabilities exactly, along with both W1 scores and violation metrics. For the completed full run, all 4,641 evaluated sources were verified as unique test cells with no flow-training overlap, evaluation coordinates matched the original PCA rows exactly, and step 6,000 was confirmed as the minimum endpoint-validation loss. The saved audit is `evaluation/results_validation.json`; separate checks confirmed that real-versus-real controls used disjoint reserved cells. The fifty-check run also reproduced every original checkpoint coordinate exactly and preserved all original training/evaluation artifacts. The constrained run additionally passed frozen-probability and gradient verification, a genuine constrained smoke run, 10 shared OT/lineage checks, and the matched three-variant artifact audit.
