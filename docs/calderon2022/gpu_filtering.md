# Calderon GPU filtering and constraint logging

The filtered, constrained and weight-2000 configs now use fused JAX interpolant checks with pending checks combined across the two training intervals. All Calderon configs disable the duplicate post-update constraint metric calculation. The full constrained training and validation losses remain enabled for the constrained variants.

## Filtering implementation

The original filter constructed all interpolant coordinates in NumPy, ran the frozen classifier trunks on the GPU, then returned logits to CPU probability calibration and transition scanning. The new filter transfers endpoint coordinates, interpolates and standardizes them on the device, evaluates the existing frozen ensemble, and checks the tissue transitions there. Only masks and small diagnostics return to the CPU.

For raw-label checking, the ensemble's final positive-temperature calibration preserves the argmax and is omitted. Member temperature calibration and averaging are retained. A path with a top-two score gap at most `1e-4` is checked again by the original decoder. This fallback preserves reference decisions near numerical ties and is recorded per interval. Both probability and margin filtering thresholds remain zero by default; every path evaluates all 50 interior points and the observed final label.

Each adjacent interval retains its own sampling key, NumPy RNG, endpoint mask, assignment/refinement state and partial-OT fallback. Two CPU workers advance those existing loops until their next classifier requests are available. One shared GPU checker handles the combined requests, then returns each interval's results separately. Worker failures release pending requests and restore the original checker objects.

The training classifier, differentiable loss adapter, all-times evaluation classifier, flow splits and evaluation protocol retain their existing roles. The optimization does not fit or replace classifier weights.

## Local verification and timings

The benchmark uses the current 4–6 / 8–10 / 12–14 h dataset, batch size 128, and identical sampling seeds 1000–1011 for reference and optimized pipelines. Classifier buckets and reference fallback kernels are warmed before timing. Timing order alternates across blocks. Measurements were taken locally on an Apple M1 Pro; other experiment activity during development makes these indicative timings.

| Measurement | Reference | Batched JAX |
| --- | ---: | ---: |
| Mean pair-generation time per batch | 0.397 s | 0.326 s |
| Mean time in filtering per batch | 0.364 s | 0.289 s |
| Interval classifier requests / combined GPU check requests over 12 batches | 1,144 | 698 |

Pair generation is **1.22× faster**, reducing its time by about **18%**. Combining pending checks reduces the request count by about **39%**. These figures exclude the optimizer update and additional constraint-metric logging, so they do not quantify total training speedup.

All optimizer input arrays and solver statistics matched exactly across the 12 minibatches. Both implementations inspected 633,300 logical interior points. The fallback rechecked 148 of 12,666 candidate paths (1.17%). An additional 1,024 arbitrary endpoint pairs, covering 51,200 interior points, produced identical acceptance, checkpoint rejection, final-label rejection and confidence masks.

The first fused implementation was slower for the many small requests. Combining requests across intervals produced the measured improvement; putting interpolation on the GPU alone was insufficient.

Verification also passed **52 Calderon CPU tests and 24 Maizels regression tests**, plus **seven Metal tests** covering path decisions, final labels, confidence scanning, checker sharing, unchanged sampling/refinements, and cleanup after errors. A two-update constrained training smoke used all 50 interpolant checks and the full 1,024-pair validation batch. It completed with finite loss and **zero duplicate constraint-metric calls**; final evaluation was skipped in this training smoke.

Artifacts: [benchmark](../../.cache/calderon2022/gpu_filter_batched_benchmark.json), [CPU tests](../../.cache/calderon2022/gpu_filter_tests.log), [Maizels regressions](../../.cache/calderon2022/gpu_filter_maizels_tests.log), [Metal tests](../../.cache/calderon2022/gpu_filter_metal_tests.log), and [training verification](../../.cache/calderon2022/gpu_filter_training_validation.json).

## Defaults and commands

The relevant settings are:

```python
cfg.logging.constraint_metrics_enabled = False
cfg.problem.interpolant_filter_backend = "jax"
cfg.problem.batch_interpolant_checks_across_intervals = True
cfg.problem.gpu_filter_decision_tolerance = 1e-4
```

The last three apply to the filtered/constrained variants. Maizels configs retain their existing logging behavior. Previously saved Calderon configs without the backend field use the reference pipeline; a running process retains the code and config it loaded at launch.

For a new constrained run from the repository root:

```sh
ENABLE_PJRT_COMPATIBILITY=1 JAX_PLATFORMS=METAL,cpu \
  .venv-flowmaps-metal/bin/python py/launchers/calderon_ot_flow_map.py \
  --config configs.calderon_constrained_pca100 --seed 0 \
  --output-dir outputs/calderon_gpu_constrained_seed0 \
  --classifier-python /opt/anaconda3/envs/maizels2023aa/bin/python
```

Add `--interpolant-filter-backend reference` to select the original filter.

For a constrained run, `--constraint-weight 100` overrides the config's weight for both training and validation. The value is saved in `config.json`. Omitting the flag preserves the config default (1000, or 2000 in the dedicated config). Saved-run evaluation reloads its recorded weight.

To reproduce the pair-generation comparison:

```sh
ENABLE_PJRT_COMPATIBILITY=1 JAX_PLATFORMS=METAL,cpu \
  .venv-flowmaps-metal/bin/python scripts/benchmark_calderon_gpu_filter.py \
  --blocks 12 --output .cache/calderon2022/gpu_filter_comparison.json
```

Source: [fused checker and batching](../../py/common/calderon_filter.py), [dataset integration](../../py/common/calderon.py), [logging switch](../../py/common/logging.py), [filtered defaults](../../py/configs/calderon_filtered_pca100.py), and [regression tests](../../tests/test_calderon_gpu_filter.py).
