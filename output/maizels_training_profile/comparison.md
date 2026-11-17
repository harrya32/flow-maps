The largest training-time differences are pair construction, the number of optimizer updates, and repeated diagnostics outside the gradient update. The extra differentiable lineage loss was small in the measured native update.

This comparison uses the current deterministic Maizels `bio_prior_ot_constrained_flow_map` preset (ID 6) and OTP-FM with precomputed OT coupling, `W2InfPotential`, and the direct solver. Both use D3/D3.8/D8 on the real-time clock 0/0.16/1 and the shared 90/10 cell split with split seed 701. The measurements were collected on 16 November 2026 using the real retained-day PCA50 cells and selected-day classifier weights.

Measurements are synchronized CPU component timings, after the first call, using the installed `flowmaps` and `mfm_env` environments. NumPy BLAS and PyTorch were limited to four threads; JAX used its CPU runtime. The frameworks, networks, and default batch sizes differ. These measurements establish the expensive components in the current code; they are not an end-to-end GPU training comparison. Production training code and existing experiments were not changed.

| Retained day | Training cells | Validation cells |
|---|---:|---:|
| D3 | 2,883 | 320 |
| D3.8 | 3,034 | 337 |
| D8 | 3,380 | 376 |

**Cold full-population biological OT has two substantial costs before training.**

The global implementation enumerates every endpoint-valid pair and classifies its straight-line interpolant at 50 interior times. The training populations contain:

| Interval | All pairs | Endpoint-valid pairs | Classifier point evaluations before interpolant rejection |
|---|---:|---:|---:|
| D3 → D3.8 | 8,747,022 | 8,129,624 | 406,481,200 |
| D3.8 → D8 | 10,254,920 | 8,092,514 | 404,625,700 |
| Total | 19,001,942 | 16,222,138 | 811,106,900 |

These counts precede interpolant filtering. The global path then builds a general HiGHS LP on the accepted edges, with explicit source/target index arrays and marginal constraints. OTP-FM uses a dense geometric cost matrix and POT's specialized transportation solver, without classifier checks. The native global path also uses per-dimension feature standardization; OTP-FM uses one common scale, so these are different coupling objectives.

In a capped 64 × 64 D3 → D3.8 example, native global edge construction took 1.32 ms without the prior and 327.63 ms with interpolant checks. The full biological LP was not run: its multimillion-variable solve and 811 million classifier point evaluations would be a large experiment. The counts above come directly from the real populations and the configured endpoint mask.

A cache hit skips these global edge checks and the LP. Full OT is then sampled from an existing pair pool during training. This cold-start cost should therefore be separated from steady-state training and from subsequent seeds that can reuse the same plan.

Source: [global edge construction](/Users/harryamad/Desktop/drive/2026/flow-maps/py/common/maizels.py:1259), [general LP solver](/Users/harryamad/Desktop/drive/2026/flow-maps/py/common/maizels.py:1400), [cache and plan sampling](/Users/harryamad/Desktop/drive/2026/flow-maps/py/common/maizels.py:1611).

**The default update budgets differ by approximately 21 times.**

| Setting | Native constrained OT flow map | OT-coupled W2Inf OTP-FM |
|---|---:|---:|
| Batch size | 128 pairs | 256 multimarginal tuples |
| Maximum optimizer updates | 50,000 | 2,400 |
| Training budget | 6,400,000 sampled pairs | 576,600 tuples |
| Trainable network parameters | 2,204,722 | 6,129,266 |
| EMA copies updated | 2 | 1 |

OTP-FM's OT dataset contains one tuple per D3 training cell. That gives `ceil(2883 / 256) = 12` updates per epoch and `200 × 12 = 2400` updates. The native preset sets 50,000 updates. Native early stopping can reduce this maximum; a comparison of completed runs must use the actual completed update count. A tuple contains three observed cells, whereas a native example contains one adjacent-interval pair.

The preset differences should be accounted for when comparing whole-job durations. OTP-FM's model has about 2.78 times as many trainable parameters and a larger batch, while its nominal run performs many fewer updates.

Source: [native preset](/Users/harryamad/Desktop/drive/2026/flow-maps/py/configs/maizels_pca50.py:347), [OTP-FM dataset](/Users/harryamad/Desktop/drive/2026/flow-maps/OTP-FM/experiments/citeseq/data.py:196), [OTP-FM preset](/Users/harryamad/Desktop/drive/2026/flow-maps/OTP-FM/configs/citeseq/defaults.json).

**With the current minibatch backend, interpolant filtering is the largest measured extra cost per update.**

The current native configs default to minibatch OT, although global OT remains selectable. Each three-marginal batch has two 64 × 64 OT problems.

| Batch construction | Median time | Measured range |
|---|---:|---:|
| Plain minibatch OT | 1.26 ms | 1.03–1.59 ms |
| Endpoint-masked OT | 1.48 ms | 1.42–1.85 ms |
| Endpoint + interpolant-masked OT | 71.72 ms | 18.14–134.76 ms |

The last mode classifies selected lines at 50 times, removes invalid edges, and resolves the assignment until its selected edges pass. A measured final batch required one refinement for D3 → D3.8 and eight for D3.8 → D8. Endpoint masking itself was inexpensive. The much larger cost came from interpolant checks and repeated refinement.

OTP-FM's usual OT-coupled W2Inf setting does no new endpoint coupling at each update. Its intermediate potential direct solve took 0.185 ms for a 256-cell batch. The actual `W2Potential` is a different variant: its single minibatch OT gradient took 6.82 ms here, before the additional work of an iterative conditional-path solver.

This 71.72 ms batch-construction cost applies to minibatch training. It does not apply to steady-state training with a cached full-population plan.

Source: [minibatch assignment refinement](/Users/harryamad/Desktop/drive/2026/flow-maps/py/common/cite_multi.py:556), [direct W2Inf solver](/Users/harryamad/Desktop/drive/2026/flow-maps/OTP-FM/src/otpfm/solvers.py:139).

**Native per-step diagnostics cost about as much as its optimizer update.**

| Native component | Current median | Jitted comparison |
|---|---:|---:|
| Gradient + optimizer, without lineage loss | 23.05 ms | — |
| Gradient + optimizer, with lineage loss | 22.82 ms | — |
| Lineage diagnostics on the current batch | 24.05 ms | 1.28 ms |
| Gradient-norm logging | 2.10 ms | Not measured |
| Two EMA updates | 10.89 ms | Already jitted |
| Validation objective, 1,024 examples | 115.44 ms | 20.16 ms |

Both native optimizer-update measurements excluded batch construction, EMA, diagnostics, validation, W&B, and checkpoint I/O. The with/without-lineage timings overlap at this measurement precision; the differentiable classifier loss did not produce a substantial measured slowdown.

By default, native scalar logging runs every update. `log_metrics` recomputes the map and classifier diagnostics outside the jitted optimizer step. Jitting that same diagnostic calculation reduced its measured median by about 19 times. Validation is also called outside a single encompassing jit; the experimental jitted comparison was about 5.7 times faster. Validation runs every 100 updates, so its current 115 ms cost amortizes to about 1.15 ms per update and is less important than per-step diagnostics.

The corresponding OTP-FM optimizer + EMA step took 89.42 ms with W2Inf and 80.81 ms without potentials, using its larger model and 256-cell batch. Its measured stages were approximately 57.67 ms forward, 16.88 ms backward, 7.54 ms optimizer, and 1.42 ms EMA. These stage medians come from separate samples and need not sum to the whole-step median. The direct potential calculation was small compared with the neural-network work.

Source: [native loop and timer](/Users/harryamad/Desktop/drive/2026/flow-maps/py/launchers/learn.py:107), [per-step diagnostic calls](/Users/harryamad/Desktop/drive/2026/flow-maps/py/common/logging.py:4883), [validation call](/Users/harryamad/Desktop/drive/2026/flow-maps/py/common/logging.py:4842), [OTP-FM optimizer loop](/Users/harryamad/Desktop/drive/2026/flow-maps/OTP-FM/experiments/common/trainer.py:210).

**The fast default POT startup solve hit its iteration cap on both real training intervals.**

| Interval | Default cap, 100,000 | Converged with cap 1,000,000 | Objective excess at default cap |
|---|---:|---:|---:|
| D3 → D3.8 | 1.09 s, limit reached | 1.91 s, success | 0.62% |
| D3.8 → D8 | 1.43 s, limit reached | 2.25 s, success | 0.39% |

Both capped plans were feasible: marginal residuals were at most about 2.4 × 10⁻¹⁶. Their costs were slightly above the converged optimum. Thus the default call's approximately 2.53-second total is not a completed exact-OT timing. Raising the cap yielded successful training plans in approximately 4.16 seconds. This still leaves a large structural advantage over the native cold global biological-support construction and general LP. Cost-matrix construction took only about 25 ms per training interval in the initial profile; the two small validation solves took about 25 ms together.

The current OTP-FM alignment function does not pass a higher `numItermax` or inspect convergence status. This finding concerns the local real-data profile, not every possible server, library version, or dataset subset.

Source: [OTP-FM alignment call](/Users/harryamad/Desktop/drive/2026/flow-maps/OTP-FM/experiments/citeseq/data.py:155), [convergence measurements](/Users/harryamad/Desktop/drive/2026/flow-maps/output/maizels_training_profile/otpfm_convergence_check.json).

**Recorded runtimes need care.**

Local historical Maizels three-marginal runs contain the same 1,024-wide native network and batch size 128, but older code and configurations. Their final reported `step_time` values are roughly 20–46 ms without OT, 340–379 ms with plain minibatch OT, and 375–942 ms with constrained biological minibatch OT. These support investigating pairing overhead but are not controlled measurements of the current code. The native timer stops before EMA and logging, and does not explicitly synchronize JAX, so it can also under-report asynchronous GPU work. Two older full-plan-cache GPU logs completed 10,000 updates in 47 minutes 53 seconds, using a smaller 289,586-parameter model; those durations are not comparable to the current larger model.

There were no local Maizels OTP-FM run logs to pair with these historical native runs. The live component measurements above therefore provide the main timing evidence. Full plotting, distribution evaluation, W&B I/O, checkpoint writing, and complete GPU runs were not benchmarked. Native plots run every 5,000 steps; OTP-FM diagnostics normally run every 20 epochs, while its sweep can select best-checkpoint-only evaluation. Those settings should be matched or reported separately in a whole-job comparison.

The first optimization priorities supported by this audit are: reuse global biological plans across runs when that backend is intended; reduce the frequency of diagnostic recomputation or jit the diagnostic functions; and compare actual update budgets before attributing job duration to the training objective. If minibatch OT is intended, its 50-point interpolation checks and refinement loop are the primary measured pairing cost. POT convergence should be checked when reporting exact full-OT startup timings.

Raw measurements and reproduction code: [native profile](/Users/harryamad/Desktop/drive/2026/flow-maps/output/maizels_training_profile/native_profile.json), [OTP-FM profile](/Users/harryamad/Desktop/drive/2026/flow-maps/output/maizels_training_profile/otpfm_profile.json), [component profiler](/Users/harryamad/Desktop/drive/2026/flow-maps/output/maizels_training_profile/profile_components.py), [convergence checker](/Users/harryamad/Desktop/drive/2026/flow-maps/output/maizels_training_profile/check_ot_convergence.py).
