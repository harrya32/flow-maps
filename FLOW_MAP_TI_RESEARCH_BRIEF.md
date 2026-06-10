# Flow Maps for Trajectory Inference

## Project brief

This project proposes a general framework for **trajectory inference (TI)** based on **flow maps**.  
The key object is a two-time transport operator:

`X_{s,t}(x) = x + (t-s) v_{s,t}(x)`, for `0 <= s <= t <= 1`.

Instead of learning only local dynamics `dx/dt = b_t(x)` as in continuous normalising flows, flow maps directly learn nonlocal transitions between arbitrary times.  
This enables faster sampling at inference time, using larger time step jumps than integrating along the instantaneous velocity. Of particular interest for trajectory inference, flow maps can allow for **simulation-free imposition of useful trajectory constraints** and **path biasing**.

## Motivation

Many TI problems are underdetermined:

- only endpoint distributions are observed,
- intermediate observations are sparse, noisy, or summary-level,
- desired trajectories should satisfy scientific or physical priors.

Existing ODE/SDE TI methods often impose these priors through:

- rollout-dependent penalties (expensive),
- fixed interpolation families (limited flexibility),
- iterative bridge updates that can accumulate approximation error.

Flow maps provide a direct operator-level interface to efficiently impose constraints on:

- intermediate marginals,
- whole-path behavior,
- cross-time dependencies.

## Positioning vs CNFs and ODE-based TI

## CNF / velocity-only flow matching view

Typical model: learn instantaneous velocity `b_t(x)` and recover trajectories by integrating ODEs.  
Strengths: mature, scalable, simulation-free for local velocity regression.  
Limitations for TI constraints:

- cross-time constraints often require rollout/unrolling through solver steps,
- long-horizon supervision can have weak or noisy gradients through integration,
- nonlocal dependencies are indirect because model is local in time.

## Flow map view

Model directly returns jumps `X_{s,t}` for arbitrary `(s,t)`.  
Advantages for constrained TI:

- direct losses on `x_t`, `x_{s->t}`, and multi-time tuples,
- many constraints can be evaluated without ODE simulation in the loss,
- easier expression of relational constraints across distant times.

## Spline/interpolant-based methods

These methods impose smooth paths via fixed interpolation functional forms.  
They are useful but can be restrictive when true dynamics require more expressive models. 

Flow maps can be viewed as a learned, data-adaptive operator family with fewer hard assumptions on trajectory geometry.

## Core training objective with constraints

Start from standard flow-map training (diagonal flow-matching + off-diagonal self-distillation, e.g. using Lagrangian self-distillation):

`L_base = L_diag + L_offdiag`.

Add constraint terms:

`L_total = L_base + sum_k lambda_k * L_k`.

Each `L_k` acts on sampled particles and selected times through `X_{s,t}`.

## Constraint templates for trajectory inference

### 1) Intermediate summary-statistic constraints

When full intermediate marginals are unavailable, enforce moments/features:

`L_moment(t*) = || E[phi(X_{0,t*}(x_0))] - m_{t*} ||^2`.

Use cases:

- known mean/variance trends from assay aggregates,
- expected marker expression summaries at specific developmental stages.

### 2) Path bias via occupancy/energy penalties

Bias trajectories toward plausible regions:

`L_path = E[ integral_0^1 U(X_{0,t}(x_0), t) dt ]`.

Approximate with random-time Monte Carlo or a short grid.  
`U` can encode:

- distance to empirical data manifold,
- physics/biology energy surrogate,
- barrier penalties for forbidden regions.

### 3) Cross-time relational constraints (intra-trajectory)

Impose rules linking states across distant times:

`x_a = X_{0,t_a}(x_0)`, `x_b = X_{t_a,t_b}(x_a)`,
`L_rel = E[d(g(x_a, x_b), 0)^2]`.

Examples:

- if a lineage marker is present at `t_a`, states at `t_b` must lie in a permitted subset,
- monotonic progression constraints in latent potential or pseudotime.

## Why this is relevant for TI settings

Trajectory inference often has:

- sparse snapshot data,
- partial knowledge at intermediate times,
- domain rules that can be used to bias learned trajectories.

Flow-map constraints naturally match this regime because supervision can be placed at the level of:

- selected times,
- selected statistics,
- selected cross-time dependencies,

without requiring full intermediate distributions or full rollout-based adjoints.

## Expected outcomes

1. A unified constrained TI objective based on flow-map operators.
2. Better constraint satisfaction at fixed compute vs velocity-only baselines.
3. Competitive endpoint/marginal fidelity.
4. Faster training-time handling of nonlocal constraints due to reduced simulation burden in loss computation.
5. Adaptable granularity of simulated trajectory at inference time, and faster simulation-free sampling.

## Suggested evaluation axes

- **Quality**: endpoint fit, intermediate fit.
- **Constraint metrics**: moment error, rule violation rate, path-energy score.
- **Efficiency**: wall-clock per step, memory, convergence speed.