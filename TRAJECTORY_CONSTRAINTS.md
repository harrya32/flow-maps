# Constrained Trajectory Inference with Flow Maps

## Executive summary

Short answer: **yes, mostly**. Flow maps make many trajectory-level constraints easier to express and optimize than CNF/ODE-only flow matching, because you can evaluate nonlocal transitions `X_{s,t}` directly, without numerically simulating the whole ODE at training time.

Important caveat: this is a computational and modeling convenience, not a free lunch in identifiability. If constraints conflict with endpoint marginals, or are too weak/ambiguous, training can still fail or yield non-unique trajectories.

## Why flow maps help for constraints

In this repo, the learned object is

`X_{s,t}(x) = x + (t - s) v_{s,t}(x)`.

So you can query:

- local velocity-like objects on the diagonal (`v_{t,t}`),
- nonlocal transitions (`X_{s,t}` for any `s < t`),
- multi-time relations via composition.

Compared to standard CNF-style training (where constraints are typically imposed by integrating `dx/dt = b_t(x)`), this gives:

- no solver-in-the-loop for many constraint evaluations,
- direct losses on intermediate states and cross-time pairs,
- easier implementation of semigroup-style and intra-trajectory constraints.

## Is this "more general" than regular flow matching?

Pragmatically: **often yes for trajectory constraints**, because optimization sees direct map evaluations instead of unrolled numerical integration.

Theoretically: not automatically more expressive than a CNF. The gain is mostly in:

- computational path (simulation-free map evaluation),
- objective design flexibility (direct losses on `X_{s,t}`),
- potentially better gradient signal for long-horizon constraints.

## General objective with constraints

Current training is a diagonal term (flow matching) plus off-diagonal self-distillation (LSD/ESD/PSD). A constrained version is:

`L_total = L_selfdistill + sum_k lambda_k * L_constraint_k`.

Here each `L_constraint_k` can depend on:

- `x_t = X_{0,t}(x_0)`,
- `x_{s->t} = X_{s,t}(x_s)`,
- tuples `(x_{t1}, x_{t2}, ..., x_{tm})`,
- optional labels/metadata.

## Your example constraints

### 1) Intermediate marginal moment constraint (example at `t = 0.5`)

If you need `E[phi(x_{0.5})] = m*`, use:

`L_moment = || E[phi(X_{0,0.5}(x_0))] - m* ||^2`.

Examples:

- mean only: `phi(x)=x`,
- covariance: `phi(x)=vec(xx^T)`,
- domain stats: custom summary features.

Why this is easy with flow maps:

- one map evaluation per sample at `t=0.5`,
- no ODE rollout needed.

### 2) Bias over the whole trajectory (stay close to observed data manifold)

Define a potential or distance-to-data score `U(x)` and penalize occupancy:

`L_path = E[ integral_0^1 U(X_{0,t}(x_0)) dt ]`.

In practice use random-time Monte Carlo or a small fixed time grid:

`L_path ~= (1/M) sum_i (1/K) sum_j U(X_{0,t_j}(x_0^i))`.

Possible `U` choices:

- nearest-neighbor distance to observed samples,
- kernel density surrogate,
- learned critic/energy model.

### 3) Intra-trajectory constraint (link `t=0.25` and `t=0.75`)

Suppose feasible relation set `Gamma` requires `x_{0.75}` to be consistent with `x_{0.25}`.

Use:

`x_025 = X_{0,0.25}(x_0)`,
`x_075 = X_{0.25,0.75}(x_025)` (or `X_{0,0.75}(x_0)`),
`L_pair = E[ d( x_075, Gamma(x_025) )^2 ]`.

This is exactly the kind of cross-time loss that is awkward with solver-based CNFs and natural with map models.

## What must change mathematically for two-terminal trajectory inference

Your setting is a bridge between two observed terminal distributions, not necessarily Gaussian-to-data.

Needed conceptual changes:

1. Replace fixed Gaussian base with observed `rho_0`.
2. Use observed `rho_1` for terminal data.
3. Choose/define coupling for sampled pairs `(x_0, x_1)`:
   - independent coupling (simple baseline),
   - OT or learned coupling (usually better trajectory semantics).
4. Keep self-distillation core (LSD/PSD/ESD), then add constraint penalties.

The flow-map framework itself does not force one terminal to be Gaussian; that is just the current experimental setup in this repo.

## What must change in this codebase (no edits yet)

## 1) Data sampling and endpoint distributions

Files:

- `py/common/datasets.py`
- `py/common/loss_args.py`

Needed:

- add support for sampling from both empirical terminal distributions (`rho_0`, `rho_1`),
- optionally add configurable pairing/coupling policy.

Current code samples `x_0` from `setup_base` (Gaussian) and `x_1` from dataset iterator.

## 2) Add constraint terms to objective

File:

- `py/common/losses.py`

Needed:

- implement `constraint_term_*` functions (moment/path/pairwise),
- combine with existing diagonal + off-diagonal losses:
  - `total_loss = base_loss + lambda_moment * L_moment + lambda_path * L_path + lambda_pair * L_pair`.

## 3) Sample extra times needed by constraints

File:

- `py/common/loss_args.py`

Needed:

- sample explicit constraint times (`t*`, or grids/pairs like `(0.25, 0.75)`),
- pass these extra tensors into `loss_fn_args`.

## 4) Add configuration knobs

Files:

- `py/configs/*.py`

Needed:

- add constraint block, e.g. weights, active constraints, time grids, targets.

## 5) Logging and diagnostics

File:

- `py/common/logging.py`

Needed:

- log each constraint loss term separately,
- log achieved constrained statistics (for example observed intermediate moments).

## Practical guidance and failure modes

1. Start with soft penalties and ramp `lambda` over training.
2. Keep constraint terms on similar scale as base self-distillation loss.
3. Verify feasibility: endpoint marginals plus constraints may be inconsistent.
4. Prefer LSD first for stability; add constraints there before trying ESD.
5. For strong intra-trajectory rules, consider augmented Lagrangian or projection-style penalties if soft constraints are insufficient.

## Bottom line

For your use case, flow maps are a strong fit:

- They naturally support constraints on intermediate marginals, full-path occupancy, and cross-time dependencies.
- They reduce or eliminate solver overhead for many such losses.
- The required changes are mostly objective/data plumbing, not a full architectural rewrite.

The main research question is not "can we express the constraints?" but "which constraints and coupling choices give identifiable, stable, and useful trajectories."
