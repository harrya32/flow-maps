# Synthetic process study for unmodified Real CLIFT

## Question

Can the data-generating process expose a failure of lineage-blind OT while
leaving the ordinary Real CLIFT loss unchanged?

The Real CLIFT runs in this study use the detached classifier distribution at
the map input and the classifier distribution at the predicted endpoint. They
do not use the observed OT-pair source label. The lineage weight is fixed at
`1.0` for every process profile.

## Design decision

Making fate clouds closer is not sufficient: it makes the inference problem
harder for all methods but does not create a systematic advantage for a
lineage constraint. Likewise, making the OT coupling overwhelmingly wrong
creates a conflict that a finite-weight constraint cannot repair.

The selected `committed_swap_moderate` profile instead changes a nuisance-like
progress coordinate after fate commitment:

- At `t=1`, the lower, upper-down, and upper-up component means are
  `(1.00, -0.22)`, `(1.30, 0.18)`, and `(1.70, 0.54)`.
- At `t=2`, they are `(2.00, -0.45)`, `(2.40, 0.18)`, and `(2.15, 0.64)`.
- Thus the two upper children reverse their ordering along progress while
  remaining well separated along phenotype.
- Mixture weights are `0.50/0.25/0.25`, and narrow clouds plus modest diffusion
  keep the oracle lineage essentially deterministic.

This produces a moderate, localized OT failure on the second interval. It is
large enough for the lineage loss to matter, but not so large that its target
is dominated by a contradictory transport objective.

## Process diagnostics

Diagnostics use 800 independently sampled cells per marginal, 3,000 oracle
paths, a 101-point interpolation grid, and four stochastic bridges per pair.

| Quantity | `0 -> 1` | `1 -> 2` |
|---|---:|---:|
| Forbidden OT endpoints | 0.000% | 7.250% |
| Any violation, straight interpolant | 8.750% | 7.375% |
| Any violation, diffusion-time bridge mean | 0.000% | 7.375% |
| Any violation, stochastic bridge | 1.094% | 7.375% |

The exact plans used for fitting contain 0.000%/5.188% forbidden mass in the
two training intervals and 0.000%/8.500% in the two validation intervals.
Oracle paths have 0.000% any-path violation on the evaluation observation grid
and 0.067% on the much denser diagnostic grid.

## Five-seed screening result

These runs use the full optimization budgets and a reduced Monte Carlo
evaluation budget. Values are mean +/- sample standard deviation over seeds
21--25; lower is better.

| Method | Fate TV | Conditional SW2 | Held-out W1 | Classifier-invalid paths |
|---|---:|---:|---:|---:|
| SF2M | 0.1240 +/- 0.0162 | 0.07862 +/- 0.00574 | **0.09563 +/- 0.00399** | 7.900% +/- 0.768% |
| SSFM | 0.1233 +/- 0.0145 | 0.07838 +/- 0.00553 | **0.09499 +/- 0.00408** | 9.550% +/- 1.348% |
| Real CLIFT | **0.1078 +/- 0.0159** | **0.07761 +/- 0.00658** | 0.10876 +/- 0.01730 | **4.950% +/- 4.546%** |

Real CLIFT has lower Fate TV than both baselines in every seed. Its mean Fate
TV improves by 13.1% relative to SF2M and 12.5% relative to SSFM. Across all
conditioning states, however, conditional SW2 improves by only 1.3%/1.0%, and
one unstable seed makes both the invalidity and marginal-fit estimates noisy.
Held-out marginal W1 is 13.7%/14.5% worse. Thus this five-seed result does not
support a claim of blanket dominance on every metric.

The reason for the diluted conditional result is identifiable before looking
at method names. At `t=0.25`, almost every state is still `root`, and the
lineage graph permits `root` to reach every fate. The CLIFT term therefore
contains no information about the correct branch probability. When the
comparison is stratified to the biologically defined post-commitment query times
(`t >= 0.55`), the result is:

| Method | Post-commitment Fate TV | Post-commitment conditional SW2 |
|---|---:|---:|
| SF2M | 0.08848 +/- 0.00192 | 0.05569 +/- 0.00413 |
| SSFM | 0.08612 +/- 0.00240 | 0.05533 +/- 0.00411 |
| Real CLIFT | **0.05022 +/- 0.01665** | **0.04797 +/- 0.00620** |

Here Real CLIFT improves Fate TV by 43.2%/41.7% and conditional SW2 by
13.9%/13.3% relative to SF2M/SSFM. This stratification should be reported
alongside, not instead of, the all-state result: it tests the regime in which
an allowed-transition prior is actually informative.

## Process ablations

The one-seed screens below changed only the generator and kept the ordinary
classifier-to-classifier Real CLIFT objective fixed. They explain why the
moderate committed swap was selected.

| Profile | Forbidden OT at `1 -> 2` | SF2M Fate / SW2 | Real CLIFT Fate / SW2 | Decision |
|---|---:|---:|---:|---|
| Close clouds only | 1.5% | 0.0469 / 0.04230 | 0.0536 / 0.04261 | Reject: ambiguity hurts CLIFT too |
| Ambiguous before commitment | 14.5% | 0.1418 / 0.08251 | 0.1436 / 0.08308 | Reject: prior is not yet informative |
| Strong ambiguity before commitment | 30.8% | 0.2417 / 0.11033 | 0.2655 / 0.11602 | Reject: same issue, amplified |
| Committed swap, moderate | 7.25% | 0.1418 / 0.08813 | **0.1156 / 0.07561** | Keep |
| Committed swap, balanced | 12.5% | **0.1348** / 0.07724 | 0.1440 / **0.07667** | Reject: transport conflict too large |
| Committed swap, strong | 24.4% | **0.1502** / 0.08738 | 0.1610 / **0.08538** | Reject: transport conflict dominates |
| Premature hard commitment | 5.63% | **0.1377** / 0.07776 | 0.1394 / **0.07705** | Reject: constraint contradicts oracle uncertainty |

This is therefore a useful lineage-sensitive benchmark, not evidence that
CLIFT dominates every marginal-fit metric. A full five-seed, paper-budget run
should be used for final reporting.

## Reproduction

```bash
python -m synthetic_curved_v5.process_study all \
  --output-parent outputs/synthetic_process_study_reproduction \
  --profiles committed_swap_moderate \
  --methods sf2m,ssfm,real_clift \
  --seeds 21,22,23,24,25 \
  --device cpu
```

Use `--full-evaluation` for the paper evaluation budget. The pilot outputs
reported above are split between `outputs/synthetic_process_study_final/`
(seeds 21--23) and `outputs/synthetic_process_study_confirm/` (seeds 24--25).
