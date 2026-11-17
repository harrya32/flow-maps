# Time points for an unconstrained flow pilot

This document records the earlier two-marginal pilot. The newer [three-timepoint design audit](three_timepoint_design.md) compares schedules with a classifier trained only on the selected flow-training windows and a separate all-times reference classifier for final violation tracking.

Start with **4–6 h → 12–14 h**, reserving **8–10 h** for an interpolation comparison. Use the `Ubi` component for the first run: it covers most of the atlas's assigned branches and removes shifts in the relative abundance of the separate Blastoderm and Yolk components. The broader tissue labels are the primary classification target; branch-ID scores on the same trajectories are a secondary diagnostic.

These are inferred developmental-age windows. Use their midpoints, **5, 9, and 13 h**, as the model clock: the held-out reference is halfway between the endpoints. Preserve the original collection windows and continuous ages in the export.

| Role | Inferred-age bin | Assigned `Ubi` cells | Branch IDs | Broad tissue labels |
| --- | --- | ---: | ---: | ---: |
| Flow source | 4–6 h | 46,410 | 14 | 9 |
| Intermediate reference | 8–10 h | 48,630 | 22 | 17 |
| Flow target | 12–14 h | 44,267 | 27 | 16 |

The later endpoint has more branch diversity than 14–16 h, more cells, and better classification on the existing random-cell test (whole assigned population, not a newly evaluated `Ubi` subset). A second run can extend to **14–16 h**: 36,098 `Ubi` cells, 25 branches and 14 tissues.

## Why this interval needs two distinct violation diagnostics

The repository's `path_validity_from_predictions` scans successive confident classifier labels along a trajectory, starting from the known source label. It rejects a trajectory at the first forbidden transition. The generated-flow evaluator does not require a paired real target-cell label. Report this **classifier-based path violation percentage**, with a fixed set of checkpoint times and confidence/coverage statistics.

For the initial endpoint run, classify intermediate predictions at the **6–8, 8–10 and 10–12 h** midpoints (normalized times 0.25, 0.5 and 0.75), and at the generated endpoint. Keep those checks identical across models; increasing their number can increase the chance that a classification error invalidates a whole path. The existing classifier trained without the 8–10 h bin provides the temporal generalization check. Its tissue accuracy is only **53.25%** on that bin, so the path percentage remains a noisy measure of agreement with the reconstructed prior. Report per-checkpoint rejection and confident coverage; paths without confident predictions must not be presented as demonstrated valid paths.

Separately, audit **endpoint compatibility using released annotations**. This isolates graph/snapshot mismatch from classification errors. The table below gives the minimum forbidden mass in any exactly balanced coupling of the observed endpoint label proportions. These percentages are computed support bounds, **not measured flow violation rates**.

| Cohort and interval | Branch-ID minimum forbidden mass | Tissue minimum forbidden mass |
| --- | ---: | ---: |
| `Ubi`, 4–6 → 12–14 h | 50.72% | 0.00% |
| `Ubi`, 4–6 → 14–16 h | 52.88% | 0.26% |
| All assigned, 4–6 → 12–14 h | 46.33% | 3.51% |
| `Ubi`, 6–8 → 10–12 h | 30.18% | 9.51% |
| `Ubi`, 8–10 → 10–12 h | 16.23% | 11.10% |
| `Ubi`, 0–2 → 14–16 h | 0.00% | 0.00% |

The tissue relation is permissive after merging branches, but the selected interval still has nontrivial restrictions: independently pairing its source and target tissue labels would violate **33.42%** of pairs in `Ubi` (47.42% in the full assigned population). A zero feasibility floor therefore does not mean every tissue pairing is allowed. The early `Ubi` population includes a large `Unknown` tissue group (24,404 of the 46,410 source cells); retain that label's defined allowed pairs and report results stratified by source tissue.

The large branch-ID floor reflects prematurely ending branches and incompatible empirical proportions in the candidate prefix hierarchy. It should not be attributed entirely to the unconstrained model. Exact target matching, the released graph and the observed sampling weights jointly determine this bound. A model that misses the target distribution can appear to evade it. Curating the graph, changing weights, or allowing growth/loss changes the benchmark.

**0–2 → 14–16 h within `Ubi` remains useful as an implementation control for whole-path monotonicity.** Its source is the common root, so every observed target branch is allowed from the source. An endpoint-only annotation score would be zero for any coupling and would give no information about which descendants were selected. Whole paths can still switch sibling branches or move backward. Avoid mixing all three components for this root control: their changing proportions alone force 66.17% forbidden endpoint mass under both reconstructed relations.

Avoid **18–20 h**, which has no released branch assignments, and initially prefer **12–14 h** over **16–18 h**, whose assigned branch support is much narrower. An **8–10 → 10–12 h** run is a useful local branch-ID control with fewer orphan sources, but its two-hour span leaves no full intermediate bin for the existing held-out-bin protocol.

## Audit method and artifacts

For source proportions p and target proportions q, maximize the sum of coupling entries on allowed pairs, subject to row sums at most p, column sums at most q, and nonnegative entries. One minus that optimum is the minimum forbidden mass for a fully balanced endpoint coupling. SciPy HiGHS solves this small linear program. The audit uses self/descendant branch pairs and the tissue relation projected from those pairs **before** merging; it does not add a second transitive closure to the merged tissue graph. These are the documented reconstructions, not an exact exported version of the authors' Figure 3C state graph.

[timepoint_support.csv](timepoint_support.csv) contains all 36 forward bin pairs for both populations; probabilities are stored as fractions. [audit_timepoint_support.py](audit_timepoint_support.py) regenerates it from the downloaded metadata and existing allowed-pair CSVs:

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 /opt/anaconda3/envs/maizels2023aa/bin/python docs/calderon2022/audit_timepoint_support.py
```

The [OT flow-map example](flow_map_example.md) implements this selection. Its seed-0 unconstrained run completed 7,000 steps and selected step 6,000 using endpoint validation; [training results and controls](ot_flow_map_results.md) report both marginals and all 4,641 test-source trajectories. It uses an endpoint/reference-atlas protocol: the existing HVG/PCA representation, graph and classifier contain atlas information beyond the two flow-training endpoint distributions, as described in the main feasibility report.
