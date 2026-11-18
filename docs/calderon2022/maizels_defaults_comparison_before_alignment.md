# Historical defaults before Maizels alignment

These notes describe the earlier shared-classifier-split protocol. See [current flow defaults](three_time_flow_map.md) and [current comparison](maizels_defaults_comparison.md). Existing runs retain their saved configurations.


This audit compares `calderon_pca100`, `calderon_filtered_pca100`, and `calderon_constrained_pca100` against Maizels IDs 7, 5, and 6, respectively. The comparison uses seed 0, the Maizels three-time schedule `d3_d3p8_d8`, real elapsed time, and no `MAIZELS_*` environment overrides. It describes defaults, not parameters recovered from a particular completed Maizels run.

The literal Maizels base config defaults to the two-endpoint schedule `d3_d8`; its three-time experiment selects `d3_d3p8_d8`. The 50-step sampling defaults are the same in both schedules.

Sources: [Maizels config](../../py/configs/maizels_pca50.py), [Calderon base config](../../py/configs/calderon_pca100.py), [filtered config](../../py/configs/calderon_filtered_pca100.py), [constrained config](../../py/configs/calderon_constrained_pca100.py), [Calderon launcher](../../py/launchers/calderon_ot_flow_map.py), and [Maizels logging/evaluation](../../py/common/logging.py). The [complete config-key comparison](../../.cache/calderon2022/maizels_defaults_diff_before_alignment.json) includes additions, removals, paths, and inactive fields; the tables below align fields with different names by their actual purpose.

## Why Calderon uses 100 composed steps

Maizels defaults to **50**, in both `logging.maizels.flowmap_n_steps` and `logging.maizels.distribution_eval_flowmap_n_steps`. Calderon's **100** is an explicit evaluation choice, not an inherited Maizels default or a training requirement.

Calderon extracts all W1 snapshots and classifier checks from one uniform, uninterrupted source-to-final rollout. Its W1 times are `0.25, 0.5, 0.75, 1`; its 50 classifier checks are `0.02, 0.04, ..., 1`. A uniform 50-step rollout contains the classifier checks, but its quarter-times would fall at steps 12.5 and 37.5. A 100-step rollout contains both grids exactly. The W1-only snapshots at 0.25 and 0.75 do not increase the number of violation checks.

Maizels W1 evaluation does not require this shared global grid: it uses 50 equal composed steps from each selected source time to each queried target time. With three-time training, omitted-time evaluation starts from the nearest observed left training marginal. Its trajectory-violation evaluation separately follows 50 composed steps across the entire first-to-last interval. Thus the two implementations also differ in where they start and how their step counts apply.

Changing only Calderon's `--flowmap-steps` to 50 currently fails the exact-grid validation for the quarter-time marginals. Matching a 50-step evaluation protocol would require changing the evaluation grid or how W1 trajectories are sampled.

## Data and training protocol differences

| Setting or behavior | Maizels three-time setup | Calderon default |
| --- | --- | --- |
| Feature dimension | PCA50; MLP input/output dimension 50 | Existing all-times HVG3000/PCA100 representation; MLP input/output dimension 100 |
| Training marginals | D3, D3.8, D8 | 4–6, 8–10, 12–14 h |
| Normalized training clock | 0, 0.16, 1 | Window midpoints 5, 9, 13 h mapped to 0, 0.5, 1 |
| Intermediate evaluation marginals | D3.2, D3.4, D3.6, D4, D5, D6, D7 | 6–8 and 10–12 h, at 0.25 and 0.75 |
| Flow cohort | Maizels cells and its lineage labels | All lineage-assigned Calderon cells within 4–14 h, including Yolk; no branch-prefix restriction |
| Flow cell split | 90% training, one 10% held-out pool at each retained time, reused for validation and trajectory evaluation | Approximately 80% training / 10% validation / 10% test at each retained window; validation and test are distinct |
| Split construction and seed | Random cell split, `maizels_holdout_seed=701` plus time-specific offsets | Reuses the classifier's saved splits stratified by window and branch ID; split seed 0 remains fixed when the flow initialization seed changes |
| Scalar network rescale | Adaptive standard deviation of the virtual interval-pair population: equal pair allocations give the three marginal distributions weights 1/4, 1/2, 1/4 | Standard deviation of the unique union of 129,764 training cells; each training cell contributes once |
| Nominal dataset size | `problem.n=500000`, a virtual pair-population size under dynamic OT | `problem.n=129764`, the number of distinct fitting cells; both runs still use the same configured update budget |
| Training classifier artifact | Schedule-specific Maizels `.pt` classifier | Frozen two-member selected-times tissue ensemble, 23 outputs; shared JAX export for filtering and the differentiable loss |
| Evaluation classifier artifact | Separate full-data Maizels `.pt` classifier | Separate three-member all-times tissue ensemble, 24 outputs, aligned to training labels by tissue name |
| Allowable transitions | Maizels lineage with descendant reachability | Supplied tissue relation projected from branch self/descendant transitions; restricted to the training vocabulary for the loss, with no additional closure of the tissue graph |
| Plain-OT infeasibility fallback | `partial` | `error`; this does not normally matter for unmasked plain OT. Filtered and constrained Calderon variants retain Maizels's `partial` fallback |
| Hyperparameter-validation time summaries | D3.4 and D6 designated by `hparam_val_times`, with separate W1 means | No designated intermediate hyperparameter-validation times; both intermediate marginals are reserved for final evaluation |

The independent training/evaluation classifier roles already exist in Maizels. The differences are the dataset, weights, ensemble implementation, and output vocabularies, not the separation of those roles.

The global Calderon feature fit includes all 314,387 lineage-assigned expression profiles, as requested. The flow's scalar rescale uses fitting cells only. Maizels's dynamic-OT scalar rescale also uses fitting pools only, but weights their distributions differently. The older standalone Maizels held-out evaluator has its own rescale-reconstruction code; it should not be substituted for the training code when defining this comparison.

Relative to the literal two-endpoint Maizels config, Calderon additionally enables time-rescaled interpolation and absolute interval bounds in pair labels. Those mechanisms already match the Maizels three-time setup. Both networks are unconditional on cell labels.

## Validation and final evaluation differences

| Setting or behavior | Maizels defaults | Calderon defaults |
| --- | --- | --- |
| Validation pair mode | `same_as_training`: plain for ID 7; filtered for IDs 5/6 | Fixed `ot_plain` for all three variants |
| Validation objective | Training objective, including the lineage penalty for ID 6 | Plain-OT LSD with the lineage penalty disabled, including for the constrained variant |
| Validation parameters | Instantaneous flow parameters | EMA 0.999 |
| Validation batching | One fixed 1,024-pair loss evaluation | Eight deterministic 128-pair evaluations, averaged; also 1,024 pairs in total |
| Final sampling parameters | Instantaneous parameters: `logging.visual_ema_factor=None` | EMA 0.999 |
| Composed sampling steps | 50; W1 uses 50 steps from the relevant source to each target | 100 across the complete source-to-final interval, with snapshots extracted from this single trajectory |
| Violation source sample | At most 512 held-out first-time cells | Every first-window test cell: 5,275 cells |
| W1 source population | All cells at the relevant observed source marginal | The same 5,275 first-window test sources used for violation tracking |
| W1 reference population | Entire queried marginal | Entire intermediate marginal; test cells only at the middle and final training times |
| W1 population cap | Uncapped (`distribution_eval_points_per_time=0`) | At most 2,048 generated and 2,048 reference cells per comparison |
| Violation confidence threshold | Probability 0; margin 0 | Both raw argmax violations and an additional probability ≥0.9 report; margin 0 |
| Evaluation classifier inference batch | 8,192 | 4,096, with a separate PyTorch worker configured for four threads |
| Flow evaluation batching | Whole-population composed evaluation | 256 source trajectories per batch |
| Default diagnostics | Direct map, composed map, Euler, and linear baselines; W1 and RBF MMD; trajectory plots and observed-interval evaluation | Composed flow-map W1 at four marginals, raw/confidence-filtered trajectory violations, and real-cell classifier diagnostics |

The standalone `eval_maizels_heldout.py` defaults to EMA 0.9999; that differs from Maizels's automatic training/final evaluation and is not the baseline used in this table. The standalone observed-W1 evaluator defaults to instantaneous parameters and 50 steps per queried interval.

## Logging, artifacts, and inactive settings

| Setting | Maizels defaults | Calderon defaults |
| --- | --- | --- |
| Scalar logging frequency | Every step | Every 100 steps, plus the first step and validation checks |
| In-training visual diagnostics | Every 5,000 steps | Dedicated launcher does final evaluation rather than Maizels's periodic plotting suite |
| W&B | `learn.py` initializes W&B; normal behavior depends on the user's W&B environment | Dedicated launcher defaults to `--wandb disabled`; offline/online modes are explicit options |
| Checkpoint/artifact layout | Generic Maizels launcher and dataset-specific run names | Dedicated Calderon run directories, `best.msgpack`, `last.msgpack`, split/provenance manifests, filter summaries, and saved marginal/trajectory evaluations |
| Weight-2000 variant | ID 6 defaults to weight 1000 | `calderon_constrained_w2000_pca100` explicitly changes weight to 2000 and uses a distinct output name |

The raw diff also lists Maizels fields omitted by the dedicated Calderon launcher. Disabled constraints fields in the plain/filtered config and disabled FID fields do not create behavior differences. The global-OT cache, mass-tolerance, orphan-dropping, and rejection-pool options are inactive under these dynamic-minibatch-OT defaults. Dataset paths, classifier paths, output directories, and run names naturally differ. Calderon's explicit interpolant chunk size of 50 performs the same full 50-point check as its filtered default and Maizels's full-grid check.

## Settings verified to match

A programmatic comparison found no differences in the **training** or **optimization** config sections for the three matching variants. The **network** sections differ only in input/output dimension. The complete active **constraints** section for the weight-1000 constrained variant matches Maizels ID 6 exactly.

These shared settings include:

- LSD, convex stopgrad, no PSD term, unconditional flow, and seed 0.
- Batch size 128, diagonal fraction 0.75, RAdam with learning rate 0.0003, gradient clipping 10, square-root schedule, and 50,000 maximum updates/decay steps.
- MLP `n_hidden=2`, width 1,024, GELU, and identical residual/precision settings.
- EMA tracking for both 0.999 and 0.9999; the choice of which parameters to evaluate differs as described above.
- Early-stopping patience 10, checks every 100 steps, zero warmup/minimum improvement, and checkpoint saving every 5,000 steps.
- Equal training pair counts per adjacent interval, minibatch squared-Euclidean OT, minibatch-size cap 128, and at most 20 resampling attempts.
- For filtering: 50 interior checks, probability/margin thresholds 0, and training-classifier batch size 8,192.
- For the native `loss_points_nll` constraint: weight 1000, transition coefficient 1, start/final coefficients 0, temperature 1, entropy weight 0, and the remaining Maizels ID-6 settings.

No flow configuration values were changed by this audit.
