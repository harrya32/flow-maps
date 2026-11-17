# Calderon constrained and filtered OT flow map

This page records the historical endpoint pilot. The constrained and weight-2000 configs now use the [three-time default experiment](three_time_flow_map.md), with the selected-times classifier in the differentiable loss and the all-times reference for violation evaluation. New training commands use those defaults; the endpoint results below retain their saved protocol.

[The constrained config](../../py/configs/calderon_constrained_pca100.py) adds the repository's **Maizels ID 6** lineage penalty to the [filtered-interpolant experiment](filtered_flow_map.md). It uses the same 4–6 → 12–14 h `Ubi` populations, PCA100 coordinates, endpoint cell splits, seed, network, optimizer and fixed plain-OT validation batches. The 8–10 h marginal remains reserved for final evaluation.

## Constraint and classifier

The native `loss_points_nll` loss evaluates tissue probabilities at the observed interpolant `I_s` and at its learned flow-map output `X_{s,t}(I_s)`. Let these probability vectors be `p_s` and `p_t`, and let `R` be the existing allowed-tissue relation. The additional objective is:

```text
1000 × mean[−log(clip(p_sᵀ R p_t, 10⁻⁶, 1))]
```

The transition coefficient is 1; start-label, final-label and entropy coefficients are zero. These are the native Maizels ID 6 defaults. With batch size 128 and 75% diagonal samples, 32 off-diagonal samples receive this term. The existing off-diagonal normalization preserves the weight of 1000. The LSD self-distillation equations and convex stopgrad remain those of the native implementation. The penalty differentiates through the frozen classifier into the learned flow-map output.

Paired tissue IDs are retained for the loss interface, while the flow network remains **unconditional**. The active transition-only objective uses probability vectors rather than a separate label-prediction loss. This penalizes sampled transitions between two loss points; it provides no hard guarantee about every point of a final composed trajectory.

The classifier is the same validation-selected, calibrated three-network tissue ensemble used by the earlier experiments. [The JAX adapter](../../py/common/calderon_lineage.py) implements its frozen preprocessing and calibrated probabilities with a differentiable float32 forward pass. The output is log probabilities, so the native backend's softmax at temperature 1 recovers the ensemble probabilities. No classifier parameters are optimized during flow training.

The adapter was compared with the original frozen decoder on **6,912 training-cell and interpolant points**: maximum probability error was **5.96×10⁻⁷**, with identical argmax decisions and ≥0.9 confidence masks. The native penalty agreed with its explicit probability formula, its directional derivative agreed with finite differences to a relative error of **3.99×10⁻⁵**, and a gradient step reduced the penalty. These results are saved in `classifier_gradient_verification.json` in the run directory.

`R` is loaded directly from `broad_tissue_allowed_pairs.csv`. It projects branch self/descendant pairs to the tissue vocabulary before merging repeated tissue names. No additional transitive closure is applied after that projection, and `Unknown` retains the supplied pairs.

## Filtering and matched training

The sampler remains the Maizels ID 5 filtered OT sampler: each batch draws 128 source and 128 target training cells, masks forbidden observed endpoint pairs, checks **50 interior points** along candidate linear interpolants using the frozen classifier, and resolves the OT assignment after invalid pairs are rejected. The interior times are `1/51, …, 50/51`; the released source and target tissue labels bound the sequence of predicted interior labels. Training filtering uses raw argmax labels, without confidence abstention.

When a complete uniform assignment is impossible, the native solver finds the maximum supported mass and renormalizes the retained pairs. Consequently, filtered training can change the effective tissue proportions. `pair_filtering_summary.json` records retained mass, rejected candidates, refinement steps and sampled source/target tissue counts across all completed updates.

The original full 50-point classifier pass is active. An optional early-rejection implementation produced identical pairs and solver statistics in a six-batch comparison, but was slower on the Apple GPU because of additional dispatches. The full-run config therefore uses a check chunk of 50.

The network has three hidden layers of 1,024 GELU units and 100 outputs. Training uses RAdam at 0.0003, gradient clipping at 10, batch size 128 and the native 75% diagonal LSD recipe. Each variant starts from the same seed-0 initialization. Every 100 updates, eight fixed plain-OT validation batches evaluate the 0.999 EMA; early stopping allows ten checks without improvement, within a 50,000-update maximum.

For fair checkpoint selection, the constrained launcher builds a **separate plain-OT LSD validation objective with the lineage penalty disabled**. It also disables the training interpolant filter for validation. This is the same endpoint-validation objective used by plain OT and filtered OT. Neither middle-marginal W1 nor trajectory violations select the checkpoint. `best.msgpack` is reloaded for final evaluation.

## Evaluation and evidence

All **4,641 source test cells** are propagated through **100 composed flow-map steps**. The classifier checks each uninterrupted trajectory at 50 times, `0.02, 0.04, …, 1.00`. W1 uses the state after 50 composed steps for the reserved 8–10 h marginal and after 100 steps for the 12–14 h test marginal. All variants use the same 2,048 generated/reference samples and Euclidean distance in the original PCA100 coordinates.

Raw path violations test all predicted labels against the released starting tissue and the allowed relation. Confidence-filtered violations skip predictions below probability 0.9, keeping the last assessed tissue. The reports give the assessed-path denominator, unassessed paths, and coverage at every checking point. An unassessed point is not established to be valid.

The same classifier supplies the training filter, the soft penalty and the final violation scores. Its accuracy on real 8–10 h `Ubi` cells is only **48.24%**. The unchanged-expression control already produces **15.49% raw violations**. These scores assess agreement with the reconstructed prior and frozen classifier; they do not establish independent clone-fate accuracy. Endpoint holdouts exclude cells from flow training, while the existing reference-atlas PCA and classifier can have seen those cells. The comparison uses one training seed.

[The three-variant result report](constrained_ot_flow_map_results.md) includes W1, violations, confidence coverage and separate named-tissue/`Unknown` starting-cell results. Its audit verifies identical endpoint splits, network and optimizer settings, evaluation inputs, graph and classifier hashes, the selected checkpoint, and unchanged reference-run artifacts.

## Reproduction

From the repository root, with the existing dataset and frozen classifier export:

```sh
ENABLE_PJRT_COMPATIBILITY=1 JAX_PLATFORMS=METAL,cpu \
  .venv-flowmaps-metal/bin/python py/launchers/calderon_ot_flow_map.py \
  --config configs.calderon_constrained_pca100 \
  --classifier-python /opt/anaconda3/envs/maizels2023aa/bin/python

JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 \
  /opt/anaconda3/envs/flowmaps/bin/python scripts/report_calderon_constrained_flow_map.py
```

The default output is `outputs/calderon_ubi_constrained_filtered_ot_flow_map_seed0`. A completed directory is preserved; use `--output-dir` for another run. The classifier interpreter loads the same PyTorch ensemble for final evaluation. Training uses the frozen JAX export.

The run saves `comparison_protocol.json`, input and implementation hashes, gradient verification, config, cell splits, training history, selected and final checkpoints, filter statistics, metrics, per-cell evaluation artifacts and `evaluation/results_validation.json`. The report generator recreates the Markdown report and PNG/SVG comparison figure from these artifacts.

To repeat the differentiable decoder and gradient verification without changing the saved run audit:

```sh
ENABLE_PJRT_COMPATIBILITY=1 JAX_PLATFORMS=METAL,cpu \
  .venv-flowmaps-metal/bin/python scripts/verify_calderon_lineage_gradient.py \
  --output .cache/calderon2022/lineage_gradient_recheck.json
```

All **23 Calderon checks** and **10 shared OT/lineage checks** passed. A separate two-update constrained smoke run completed real filtered OT pairing, differentiable constraint updates, shared validation, best-checkpoint reload, composed sampling and classifier/W1 evaluation.
