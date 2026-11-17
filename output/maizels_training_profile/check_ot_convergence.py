"""Check POT's current iteration cap on the retained Maizels training cells."""

import json
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import ot
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
with np.load(OUT / "retained_cells.npz") as raw:
    populations = [raw[f"train_x{i}"] for i in range(3)]
all_train = np.concatenate(populations)
center = all_train.mean(axis=0, keepdims=True)
scale = float(all_train.std(axis=0).max())
populations = [((x - center) / scale).astype(np.float32) for x in populations]
records = []
with threadpool_limits(limits=4):
    for index in range(2):
        x, y = populations[index:index + 2]
        cost = ot.dist(x, y, metric="sqeuclidean")
        a, b = ot.unif(len(x)), ot.unif(len(y))
        for limit in (100_000, 1_000_000):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                start = perf_counter()
                plan, log = ot.emd(a, b, cost, numItermax=limit, log=True)
                duration = perf_counter() - start
            record = {
                "interval_index": index,
                "shape": list(cost.shape),
                "iteration_limit": limit,
                "solve_s": duration,
                "result_code": int(log["result_code"]),
                "warning": log.get("warning"),
                "warnings": [str(item.message) for item in caught],
                "transported_mass": float(plan.sum()),
                "objective": float(np.sum(plan * cost)),
                "source_max_abs_residual": float(np.max(np.abs(plan.sum(axis=1) - a))),
                "target_max_abs_residual": float(np.max(np.abs(plan.sum(axis=0) - b))),
            }
            records.append(record)
            print(json.dumps(record), flush=True)
            del plan
        del cost
(OUT / "otpfm_convergence_check.json").write_text(
    json.dumps({"records": records}, indent=2) + "\n"
)
