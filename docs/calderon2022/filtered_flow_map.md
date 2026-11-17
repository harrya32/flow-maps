# Calderon flow map with filtered interpolants

This page records the historical endpoint pilot. The filtered config now uses the [three-time default experiment](three_time_flow_map.md), its selected-times training classifier and the all-times evaluation classifier. New training commands use those defaults; the endpoint results below retain their saved protocol.

[The filtered config](../../py/configs/calderon_filtered_pca100.py) implements the **Maizels ID-5 `bio_prior_ot_flow_map` training recipe** for the same **4–6 → 12–14 h Ubi** population as the plain-OT pilot. It uses the existing [Calderon launcher](../../py/launchers/calderon_ot_flow_map.py). The network is unconditional; there is no generated-path penalty or tissue conditioning.

The source/target cell splits, stored PCA100 coordinates, network initialization, three 1,024-unit hidden layers, LSD objective, 75% diagonal sampling, RAdam learning rate 0.0003, gradient clipping, maximum 50,000 updates and 0.999 sampling EMA match the baseline. [The original population and representation protocol](flow_map_example.md) remains in force.

The full seed-0 run completed **4,700 updates** and selected **step 3,700** using endpoint validation. [Results, the matched comparison and figure](filtered_ot_flow_map_results.md) show **58.31% raw violations** and **11.61% confidence-filtered violations**, with **58.28% assessed-path coverage**. Mean retained coupling mass was **82.54%**. W1 is **8.4856** at the held-out middle and **11.1041** at the test endpoint, slightly worse than the plain-OT baseline.

## How pairs are filtered

Each update draws a fresh 128-cell source block and target block, with replacement, from the endpoint training pools. It builds squared Euclidean costs in the original PCA100 coordinates.

1. Mask endpoint pairs using the existing `broad_tissue_allowed_pairs.csv`. This is the relation projected from branch self/descendant pairs before tissue merging; it receives no further transitive closure. `Unknown` retains its defined transitions.
2. Solve the masked assignment and inspect previously unchecked positive-mass edges.
3. For each such edge, classify **50 interior points** on its linear PCA interpolant, at normalized times `1/51, 2/51, …, 50/51`. Scan from the released source tissue through the predicted tissues and then to the released target tissue. Training uses raw classifier labels: probability and margin thresholds are both zero, as in Maizels ID 5.
4. Remove invalid edges and resolve OT until every positive-mass edge has passed the filter. Sample the 128 optimizer pairs from that plan.

The implementation reuses the native CITE/Multi/Maizels masked-assignment and lazy refinement engine, injecting the Calderon relation and frozen classifier checker. It does not sample a plain OT plan and then silently discard its rejected pairs.

Maizels ID 5 allows **maximum-valid partial assignment** when the hard mask prevents balanced matching. Retained valid mass is renormalized before sampling. Thus the per-block training marginals can differ from the full endpoint snapshots; `pair_filtering_summary.json` reports mean retained mass, rejected/check counts, refinements, empty-block retries and the actual sampled source/target tissue counts. Snapshot populations and final test references are preserved. Empty filtered blocks are resampled within the configured limit, without relaxing the prior.

A [separate endpoint-label support audit](../../scripts/audit_calderon_endpoint_support.py) uses only the saved training label fractions. They permit **99.9575%** global transport mass; the earlier full-snapshot tissue audit permitted 100%. In 2,000 independently sampled 128-cell blocks (seed 3029), maximum endpoint-supported mass averages **98.36%**. This is a support diagnostic for the endpoint mask alone, not an endpoint-only trained model. It provides context for the extra mass excluded by classified interpolants.

## Frozen classifier and verification

The same selected `heldout_8_10_broad_tissue_selected.json` ensemble is used for filtering and final scoring. [The exporter](../../scripts/export_calderon_tissue_ensemble.py) copies its frozen wide/residual weights to an NPZ; it does not fit another classifier. [JAX inference](../../py/common/calderon_classifier.py) uses the stored PCA scaler, normalization statistics, per-member temperatures and ensemble calibration.

Apple's JAX backend cannot lower the exact `erf` operation in GELU. Its compatible float32 error-function polynomial was checked against the original PyTorch probabilities on **6,912 points derived only from flow-training endpoint cells**. Maximum probability error was **6.56×10⁻⁷**, with **100% agreement in tissue decisions and ≥0.9 confidence masks**. Adaptive inference batch sizes were checked separately, also with identical decisions. The export, each original classifier checkpoint, forward implementation and verification inputs have recorded hashes.

All **19 Calderon unit checks** and **7 shared Maizels OT regression checks** passed. These include the observed final-label check, agreement with the native Maizels interpolant checker, OT reassignment after rejected interpolants, explicit relation handling without closure, and identical compositions at shared evaluation times. A two-update GPU smoke run completed real filtered coupling, LSD updates, checkpoint reload and final classifier/W1 evaluation.

## Selection and evaluation

For a matched comparison, selection uses the **same fixed plain-OT endpoint-validation batches and times as the baseline**, rather than selecting only on filtered validation pairs. Eight 128-pair validation batches are evaluated every 100 updates. Early stopping permits 10 checks without improvement. The 8–10 h reference and test endpoints do not select a checkpoint.

Final evaluation pushes **all 4,641 test source cells** through **100 composed flow-map steps**. Tissue labels are checked at **50 points**, `t=0.02, 0.04, …, 1.00`, corresponding to inferred ages 5.16–13.00 h. These generated-path checks differ slightly from the 50 interior training-interpolant checks. Raw and ≥0.9 confidence-filtered violation rates retain the existing abstention/coverage semantics.

W1 uses the composed states at `t=0.5` and `t=1`, against the reserved 8–10 h marginal and 12–14 h test cells. The same 2,048 generated/reference samples and seeds as the baseline are used. Final evaluation applies to the full test populations, including cells whose analogous training regions may receive less mass under the filter.

These are diagnostics of agreement with the frozen reconstructed prior and classifier. The classifier's 8–10 h Ubi accuracy is 48.24%; training-time filtering uses that same classifier, so lower detected violations do not establish independent biological or clone-fate accuracy. The holdouts apply to flow training; the existing reference-atlas PCA/classifier can have seen endpoint flow-test cells.

## Run and artifacts

From the repository root:

```sh
# Export the existing ensemble; requires the PyTorch environment.
/opt/anaconda3/envs/maizels2023aa/bin/python scripts/export_calderon_tissue_ensemble.py

# Verify the GPU decoder against the original predictions.
ENABLE_PJRT_COMPATIBILITY=1 JAX_PLATFORMS=METAL,cpu \
  .venv-flowmaps-metal/bin/python scripts/verify_calderon_tissue_ensemble.py \
  --output .cache/calderon2022/tissue_ensemble_jax_metal_validation.json

# Train and evaluate the full filtered variant.
ENABLE_PJRT_COMPATIBILITY=1 JAX_PLATFORMS=METAL,cpu \
  .venv-flowmaps-metal/bin/python py/launchers/calderon_ot_flow_map.py \
  --config configs.calderon_filtered_pca100 \
  --classifier-python /opt/anaconda3/envs/maizels2023aa/bin/python
```

Full-run artifacts are saved in `outputs/calderon_ubi_filtered_ot_flow_map_seed0`. In addition to the usual config, splits, checkpoints, history and evaluation files, this run records `comparison_protocol.json`, `classifier_backend_verification.json`, `baseline_artifact_hashes.json` and `pair_filtering_summary.json`. The baseline model and four-/fifty-check evaluations remain preserved.

After training, reproduce the label-support audit and comparison report with:

```sh
/opt/anaconda3/envs/flowmaps/bin/python scripts/audit_calderon_endpoint_support.py
/opt/anaconda3/envs/flowmaps/bin/python scripts/report_calderon_filtered_flow_map.py
```
