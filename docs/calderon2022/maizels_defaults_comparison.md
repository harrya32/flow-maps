# Calderon defaults after Maizels alignment

The comparison uses seed 0 and corresponding Maizels IDs 7/5/6 with the three-time D3/D3.8/D8 schedule, real elapsed time and no `MAIZELS_*` environment overrides. The literal Maizels base config still defaults to the two-endpoint D3/D8 schedule. The [previous comparison](maizels_defaults_comparison_before_alignment.md) records the defaults before this change.

## Current settings

| Setting | Current Calderon behavior |
| --- | --- |
| Cell split | Independent 90/10 training/heldout split at every retained time; heldout reused for validation and first-time violation sources |
| Validation objective | Same pairing, filtering and loss as training; includes the lineage penalty when enabled |
| Validation batching | One fixed 1,024-pair batch over the adjacent retained intervals |
| Flow parameters | Instantaneous parameters for validation, plots and final sampling |
| Composed W1 | 50 steps from the appropriate observed source to each queried target; chained source-to-final evaluation uses 50 per retained interval |
| Violation sampling | 512 first-time held-out cells without replacement, 50 full-window composed steps/checks |
| W1 populations | At most 2,048 source and 2,048 reference cells without replacement; current evaluation override |
| Evaluation diagnostics | Composed-flow-map and Euler W1/violations; MMD, direct and linear evaluation disabled |
| Confidence reporting | Raw argmax violations only; probability/margin thresholds zero |
| OT fallback | `partial` for plain, filtered and constrained variants |
| Logging | Native Maizels cadence; duplicate post-update constraint diagnostics and direct plots omitted |
| Interpolant filtering | Fused JAX checks, batched across the independent training intervals, with reference checks for near-ties |

The source-to-final W1 rollout uses 100 updates for this three-time benchmark: **50 in each of two intervals**, as in Maizels. This is distinct from the 50-step full-window violation rollout. Intermediate W1 starts from the nearest observed left training marginal rather than from the held-out violation sample.

The optimizer, MLP hidden layers, tracked EMA decays, stopping counts, 50-point training interpolant filter and weight-1000 native constraint continue to match Maizels. The weight-2000 config is an explicit separate option.

## Remaining differences

| Setting | Maizels three-time experiment | Calderon |
| --- | --- | --- |
| Feature dimension | PCA50 | Existing all-times HVG3000/PCA100; network input/output dimension 100 |
| Training times | D3, D3.8, D8 → 0, 0.16, 1 | 4–6, 8–10, 12–14 h midpoints → 0, 0.5, 1 |
| Intermediate times | Seven omitted days; D3.4/D6 designated for separate hyperparameter-time summaries | 6–8 and 10–12 h, both reserved for evaluation |
| Scalar rescale | Standard deviation of the virtual interval-pair distribution, weighting marginals 1/4, 1/2, 1/4 | Standard deviation of the unique union of fitting cells, weighting each fitting cell once |
| Classifier weights/vocabularies | Schedule-specific training classifier and separate full-data classifier | Existing 23-tissue two-member training ensemble and 24-tissue three-member oracle, with their existing supervised splits |
| Allowable transitions | Maizels descendant reachability | Projected branch-to-tissue relation, without an additional closure of the tissue graph |
| Flow batching | Whole-population diagnostic sampling | Saved violation trajectories are batched at 256; native W1/plot diagnostics retain Maizels sampling |
| Artifacts | Maizels dataset/run paths and generic native checkpoints | Calderon paths, native checkpoints plus existing `.msgpack` checkpoints and saved population/provenance files |
| Evaluation populations | Uncapped source/reference populations | At most 2,048 source/reference cells per comparison |
| Evaluation metrics/samplers | Direct, composed-flow-map, Euler and linear W1/MMD | Composed-flow-map and Euler W1; MMD, linear and direct evaluation disabled |
| Constraint scalar diagnostics | Additional post-update constraint calculation | Disabled; the full training/validation objective remains enabled |
| Filter implementation | Existing classifier/transition-check pipeline | Fused JAX inference and pending checks combined across intervals; all 50 points, assignments and fallbacks retained |

Calderon retains exact empirical Euclidean W1 on its selected subsets. Shared W1/MMD helpers remain available to other experiments; Maizels keeps its existing defaults. Previously saved Calderon runs use their recorded settings unless `--reduced-evaluation` is supplied during checkpoint evaluation.

See the [current experiment details](three_time_flow_map.md), [base config](../../py/configs/calderon_pca100.py), [launcher](../../py/launchers/calderon_ot_flow_map.py), [native logging adapter](../../py/common/calderon_logging.py), and [complete config-key comparison](../../.cache/calderon2022/maizels_defaults_diff.json). The raw comparison also lists inactive/global-OT fields and differences in paths or configuration structure.

The [current verification report](../../.cache/calderon2022/reduced_evaluation_validation.json) records 76 passing tests, native plotting and the existing seed-0 checkpoint's 21-second evaluation with 2,048-cell W1 populations. The [initial alignment report](../../.cache/calderon2022/maizels_protocol_validation.json) retains the original CPU/GPU smoke runs and oracle-export agreement checks.

The later [GPU filtering report](gpu_filtering.md) documents the filtering/logging optimizations and their separate correctness and timing checks. The raw config-key comparison predates these additional settings.
