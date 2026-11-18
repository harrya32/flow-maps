# Calderon W1 evaluation with 4096 cells

Re-evaluated the same four matched seeds (0–3) of each method using the unchanged saved best checkpoints. Each empirical W1 comparison uses 4096 generated cells and 4096 reference cells sampled without replacement. The original 2048-cell results remain intact.

W1 uses uniform cell masses and Euclidean distance in the existing all-times HVG3000/PCA100 representation. Composed flow-map sampling uses 50 steps per intermediate query and observed interval; the 4–6 h to 12–14 h rollout carries the prediction across both intervals (100 total steps). The intermediate evaluations start from real 4–6 h and 8–10 h populations, respectively.

Values below are mean ± sample standard deviation across four training seeds. Intermediate W1 averages the 6–8 h and 10–12 h evaluations within each seed. Observed W1 averages the two training intervals within each seed.

| Method | Intermediate W1, 2048 | Intermediate W1, 4096 | Full rollout W1, 2048 | Full rollout W1, 4096 |
|---|---:|---:|---:|---:|
| OT flow map | 8.974 ± 0.194 | 8.557 ± 0.128 | 15.144 ± 1.613 | 14.992 ± 1.771 |
| Filtered OT flow map | 8.846 ± 0.187 | 8.428 ± 0.182 | 13.541 ± 0.282 | 13.110 ± 0.163 |
| Filtered + constrained (weight 1000) | 9.043 ± 0.215 | 8.634 ± 0.128 | 16.575 ± 0.744 | 16.203 ± 0.627 |

| Method | Observed interval W1, 2048 | Observed interval W1, 4096 | Original violation rate |
|---|---:|---:|---:|
| OT flow map | 10.441 ± 0.179 | 10.096 ± 0.280 | 75.29% ± 1.01 |
| Filtered OT flow map | 10.327 ± 0.107 | 10.048 ± 0.132 | 62.21% ± 2.99 |
| Filtered + constrained (weight 1000) | 11.174 ± 0.336 | 10.939 ± 0.342 | 44.53% ± 6.14 |

The ordering of the method means is the same at both sample sizes. Filtering gives the lowest W1. With 4096 cells, the weight-1000 constraint raises intermediate W1 by 0.206 and full-rollout W1 by 3.093 compared with filtering alone. The intermediate differences remain modest.

The violation rates above are the original 512-source, 50-check oracle measurements; only W1 was recalculated. The 4096-cell source and reference populations are identical across methods within each seed and query. Larger samples are drawn with the original evaluation seed protocol; the 4096-cell samples need not contain the original 2048 samples.

Verified unchanged checkpoint/config/original-metrics hashes, shared dataset provenance, exact 4096×100 sample dimensions, finite generated coordinates, saved W1 values, and composition counts. Reproducing the first checkpoint's saved 2048-source intermediate trajectory gave a maximum coordinate difference of zero.

Machine-readable per-seed results and population paths: [w1_4096_summary.json](/Users/harryamad/Desktop/drive/2026/flow-maps/.cache/calderon2022/w1_4096_summary.json).

Evaluation driver: [reevaluate_w1_4096.py](/Users/harryamad/Desktop/drive/2026/flow-maps/.cache/calderon2022/reevaluate_w1_4096.py).
