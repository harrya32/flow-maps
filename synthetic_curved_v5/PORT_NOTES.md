# Port map and deliberate differences

The reference implementation is the MIT-licensed
[`BenjaminisCoding/flow_maps_traj`](https://github.com/BenjaminisCoding/flow_maps_traj)
repository, locally inspected from `flow_maps_benji`.
The source checkout was at commit `b0ec335399bf0fbcd958b8d68e72282dde21a402`.

| This port | Upstream source responsibility |
|---|---|
| `dataset.py` | `curved_synthetic/dataset.py` |
| `classifier.py` | `curved_synthetic/classifier.py` |
| `diffusion.py`, SF2M parts of `models.py` | `curved_synthetic/models.py` and `hierarchical_branching/sf2m.py` |
| strong-map parts of `models.py` | `curved_strong_maps/model.py` |
| `training.py` | `curved_synthetic/models.py`, `curved_strong_maps/training.py`, and `curved_strong_maps/five_seed_study.py` |
| `evaluation.py` | `curved_synthetic/evaluation.py` and `curved_strong_maps/five_seed_evaluation.py` |

The following changes are intentional:

- Parameters that defined the synthetic process as Python constants are JSON
  fields, including every ramp, branch coefficient, width knot, class boundary,
  mixture weight, and ground-truth diffusion parameter.
- Training times and snapshot spacing are configurable; pair banks and held-out
  evaluation times are derived rather than fixed to array indices.
- Exact balanced OT is included because the paper's final five-seed table used
  only `ground_truth_ot` and `sigma_0p2_ot`. The earlier entropic-OT screening
  matrix is not part of this compact port.
- The source's repository-specific artifact firewall, code freezing, runtime
  calibration, and post-hoc audit programs are replaced by one portable CLI,
  a resolved-configuration/source-hash check, checkpoint hashes, and common
  saved reference banks. The final calibrated sampling rates are retained in
  `paper.json`.
- Refined marginal strong-map sampling is executed directly at 4,160 calls per
  unit. It has the same law as the source's conditionally refined Brownian
  increments, but this port does not also create the coarser convergence panel.

These differences make experimentation straightforward, but a changed JSON
configuration is a new benchmark rather than the frozen published run.

## Post-port method

`real_clift` is an additional comparison and is not part of the upstream paper
reproduction. It applies the stochastic Maizels training structure to this
benchmark: end-to-end local Euler--Maruyama regression, off-diagonal
self-consistency with an EMA teacher, and an endpoint lineage NLL on the same
current-student off-diagonal prediction. The transition prior is anchored to
the known class of the coupled source endpoint, matching the main experiment's
label semantics and avoiding accidental over-constraint from reclassifying a
noisy intermediate state. It retains the benchmark's fixed OT
plans, uses a horizon curriculum across every adjacent training interval,
trains for a fixed budget, and evaluates only the instantaneous student.
