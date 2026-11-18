# Calderon stochastic flow-map experiment

The three stochastic counterparts use the existing direct strong stochastic flow-map (SSFM) trainer from `py/launchers/maizels_stochastic.py`. `py/configs/calderon_stochastic.py` builds directly from the corresponding Maizels stochastic config, and `py/launchers/calderon_stochastic.py` provides preparation, training, and saved-checkpoint evaluation.

| Calderon variant | Calderon ID | Maizels stochastic ID | Training pairs | Differentiable constraint |
|---|---:|---:|---|---|
| `ot` | 0 | 5 | Unconstrained minibatch OT | Disabled |
| `filtered` | 1 | 3 | Endpoint tissue-mask minibatch OT | Disabled |
| `constrained` | 2 | 4 | Endpoint tissue-mask minibatch OT | Direct off-diagonal endpoint transition NLL |

Like stochastic Maizels, the biological prior filters the annotated tissue transition between OT endpoints. There are **no straight-line interpolant classifier checks**: the SSFM training interpolants include Gaussian noise. The constrained variant adds the selected-times ensemble's differentiable transition NLL on the current student's direct prediction, using the native stochastic loss and calibration. The frozen ensemble remains differentiable with respect to generated PCA coordinates.

The supplied tissue relation is projected onto the training vocabulary without adding a transitive closure. When a balanced masked OT assignment is infeasible, the existing maximum-valid partial assignment is used; a completely unsupported batch is retried and ultimately raises an error. The tissue mask is never relaxed.

## Data and classifier roles

- Use the existing all-times HVG3000/PCA100 export of the lineage-assigned cells. No additional PCA fit is performed.
- Train on **4–6, 8–10, and 12–14 h**, normalized to 0, 0.5, and 1. Both adjacent intervals receive equal pair counts before the batch is shuffled. Tissue labels and absolute interval bounds enter the objective; the model is unconditional on tissue labels.
- Use the independent Maizels **90/10 train/validation cell split**, with fixed holdout seed 701 and the existing window offsets. The three training windows contain **145,985 training cells** and **16,221 validation cells**. Classifier cell folds remain separate from these flow splits.
- Hold out **6–8 and 10–12 h** for marginal evaluation. Exclude times outside the retained 4–14 h range from flow training and evaluation. The existing global expression preprocessing includes all atlas cells.
- Use the selected-times **23-tissue ensemble** for the training constraint. Use the all-times **24-tissue ensemble** as the violation oracle. Classifier outputs are aligned by tissue name. Endpoint filtering uses the original tissue annotations and the supplied allowable-transition table.
- The loader checks dataset, classifier, and split metadata; run artifacts record their SHA256 hashes. Classifier JAX exports must match the selected manifests.

## Defaults retained from stochastic Maizels

| Setting | Default |
|---|---|
| Optimizer | AdamW, learning rate 3e−4, betas 0.9/0.99, weight decay 1e−4, gradient clip 10 |
| Schedule | Cosine decay; 1,000-step warmup; maximum 100,000 updates |
| Optimizer batch / configured OT batch | 512 / 512, with the optimizer batch divided equally between the two intervals |
| Objective | 75% local stochastic targets, 25% semigroup consistency |
| SSFM | 3 Brownian coefficients, hidden width 512, 3 hidden layers, uncertainty width 64 |
| Interpolant noise / diffusion | Gamma scale 0.2 / diffusion scale 0.2 |
| Local horizon | Fraction 0.02 of the observed interval; the native horizon curriculum is retained |
| EMA | Decay 0.999; final evaluation uses the EMA from the best validation step |
| Constraint | Weight **10**, transition NLL weight 1; start, final-label and entropy weights 0; constraint batch 64 |
| Validation | Fixed **1,024-pair** held-out batch, fixed noise, same objective and pair mode as training |
| Early stopping | Validate every 100 steps, patience 10, minimum improvement 0, no warmup |
| Noise evaluation | **3 independent noise draws**, with metrics averaged across draws and noise variation reported |
| Composed sampling | **50 steps per query/observed interval**; full rollout has **100 steps** across the two intervals |
| Violation evaluation | **512 held-out 4–6 h cells**, **50 full-window checks**, zero confidence thresholds |
| Logging | Scalar/progress every 50 steps; checkpoint and plots every 5,000 steps; same W&B project/entity |

The necessary differences from stochastic Maizels are the dataset and PCA dimension (**100**), time windows, tissue vocabulary/relation, classifier manifests/exports, output names, and the **4,096-cell W1 cap on both source and reference populations**. Calderon evaluation is **composed only**, with MMD and direct evaluation disabled. The optimizer, loss, validation, sampling, plotting and logging implementations are shared. The stochastic defaults include EMA evaluation and constraint weight 10.

Final evaluation computes **composed intermediate W1**, observed-interval W1, full-rollout W1, and lineage violations. **No MMD is computed and no direct samples are generated**, including periodic plots and evaluation of older saved checkpoints. Composed evaluation retains the original noise keys, and the sampler-neutral `ssfm` W1 aliases refer to composed sampling. Each of the three noise draws produces at most 4,096 generated cells, which are compared with at most 4,096 reference cells; the three draws are scored separately rather than pooled into one 12,288-cell OT problem. The cap applies to intermediate, observed-interval, and full-rollout W1. `--eval-max-points 0` restores uncapped population evaluation.

Intermediate evaluations restart from the real preceding training population: 4–6 → 6–8 h and 8–10 → 10–12 h. Observed-interval evaluations similarly restart from real cells at the left endpoint. Full-rollout evaluation starts once at 4–6 h and carries the stochastic prediction through 8–10 to 12–14 h.

The principal composed intermediate metric is `final_eval/flowmap_mean_emd_test_times`. Full-rollout W1 is `final_eval/day4-6_to_day12-14_rollout_flowmap_emd`. The oracle violation rate is **`final_eval/full_data_classifier/flowmap_invalid_trajectory_pct`**; unprefixed validity aliases follow the selected-times classifier, as in stochastic Maizels.

## Commands

Run from the repository. On the current Apple GPU environment, the following runs two seeds of each variant sequentially:

```bash
for seed in 0 1; do
  for variant in ot filtered constrained; do
    ENABLE_PJRT_COMPATIBILITY=1 JAX_PLATFORMS=METAL,cpu \
      /Users/harryamad/Desktop/drive/2026/flow-maps/.venv-flowmaps-metal/bin/python \
      /Users/harryamad/Desktop/drive/2026/flow-maps/py/launchers/calderon_stochastic.py \
      --variant "$variant" --seed "$seed"
  done
done
```

`--slurm-id 0`, `1`, or `2` can replace the named variant. The default output root is `/Users/harryamad/Desktop/drive/2026/flow-maps/outputs/calderon_stochastic`; the trainer creates one variant/seed directory beneath it, following Maizels' layout. Use `--output-folder` to choose another root. Existing trained runs are protected against overwrite.

To use constraint weight 100:

```bash
ENABLE_PJRT_COMPATIBILITY=1 JAX_PLATFORMS=METAL,cpu \
  /Users/harryamad/Desktop/drive/2026/flow-maps/.venv-flowmaps-metal/bin/python \
  /Users/harryamad/Desktop/drive/2026/flow-maps/py/launchers/calderon_stochastic.py \
  --variant constrained --constraint-weight 100 --seed 0
```

`--stage prepare` verifies and records the populations and classifier provenance without training. `--smoke --wandb-mode disabled` runs two optimizer updates with a small network, small validation/evaluation batches, one noise draw, and four trajectory steps; these overrides are confined to the smoke run.

Saved stochastic runs contain `best.msgpack`, `latest.msgpack`, `config.json`, `data_summary.json`, `cell_splits.csv.gz`, and `final_metrics.json`. Checkpoints retain instantaneous and EMA parameters, feature scales, config, and the best validation step/loss.

To re-evaluate a saved checkpoint with a different W1 cap:

```bash
ENABLE_PJRT_COMPATIBILITY=1 JAX_PLATFORMS=METAL,cpu \
  /Users/harryamad/Desktop/drive/2026/flow-maps/.venv-flowmaps-metal/bin/python \
  /Users/harryamad/Desktop/drive/2026/flow-maps/py/launchers/calderon_stochastic.py \
  --stage evaluate \
  --checkpoint /Users/harryamad/Desktop/drive/2026/flow-maps/outputs/calderon_stochastic/calderon_symmetric_4_8_12_pca100_minibatch_ot_ssfm_seed0/best.msgpack \
  --eval-max-points 2048
```

Evaluation reloads the saved config, checks it against the embedded checkpoint config and input hashes, and evaluates the saved EMA. Results go into `evaluations/w1_2048` beneath the run directory unless `--evaluation-output-dir` is supplied. Training overrides are unavailable during checkpoint evaluation.

## Validation

The new adapter and existing Maizels, CITE/Multi, and LARRY stochastic tests passed: **62 tests**. Checks cover default parity, fresh OT coupling, endpoint-mask enforcement, independent validation cells, interval clocks, 4,096-cell caps on both populations, composed full rollouts, manifest-based classifiers, noise draws, and checkpoint restoration. The reduced-protocol checks fail if a direct sampler or MMD calculation is invoked, verify composed-only plots, and check that removing direct evaluation preserves composed noise keys.

All three variants completed two-step **GPU smoke runs on the real Calderon dataset**, including validation, both tissue classifiers, and final marginal/rollout W1. A saved constrained checkpoint was successfully re-evaluated on the GPU with a changed W1 cap. After MMD and direct evaluation were removed, that existing checkpoint was re-evaluated again: all **32 retained metrics matched exactly**, and the result contained no direct or MMD metrics. These are execution checks; full stochastic comparison runs have not been trained.
