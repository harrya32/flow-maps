"""Shared marginal protocols and summaries for Maizels checkpoint evaluation."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

from . import maizels, wasserstein


@dataclass
class Marginals:
    retained: Tuple[str, ...]
    timepoints: Tuple[str, ...]
    hparam_val_times: Tuple[str, ...]
    test_times: Tuple[str, ...]
    populations: Dict[str, np.ndarray]
    times: Dict[str, float]

    def metadata(self) -> Dict:
        return {
            "retained_timepoints": list(self.retained),
            "timepoints": list(self.timepoints),
            "hparam_val_times": list(self.hparam_val_times),
            "test_times": list(self.test_times),
            "normalized_times": self.times,
            "population_sizes": {
                day: int(len(population))
                for day, population in self.populations.items()
            },
        }


def build_marginals(cfg, data: Dict, hparam_val_times=None) -> Marginals:
    """Keep every cell at every marginal inside the trained time range."""
    retained = maizels.retained_timepoints(cfg)
    first, last = map(maizels.parse_timepoint, (retained[0], retained[-1]))
    timepoints = tuple(
        day for day in maizels.TIMEPOINTS
        if first <= maizels.parse_timepoint(day) <= last
    )
    if any(day not in timepoints for day in retained):
        raise ValueError(f"Unknown retained Maizels timepoints: {retained}.")
    populations = {
        day: np.asarray(data["x"][data["timepoints"] == day], dtype=np.float32)
        for day in timepoints
    }
    empty = [day for day, population in populations.items() if not len(population)]
    if empty:
        raise ValueError(f"Missing Maizels marginal populations: {empty}.")

    if hparam_val_times is None:
        hparam_val_times = getattr(cfg.problem, "hparam_val_times", ("D3.4", "D6"))
    if hparam_val_times is None:
        hparam_val_times = ("D3.4", "D6")
    if isinstance(hparam_val_times, str):
        hparam_val_times = (
            () if hparam_val_times.strip().lower() in ("", "none")
            else [value.strip() for value in hparam_val_times.split(",") if value.strip()]
        )
    by_value = {maizels.parse_timepoint(day): day for day in timepoints}
    requested = set()
    for value in hparam_val_times:
        try:
            day = by_value[float(str(value).strip().upper().removeprefix("D"))]
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Unknown Maizels hparam validation time {value!r}.") from exc
        if day in retained:
            raise ValueError(f"Hparam validation time {day} is a retained observation.")
        requested.add(day)
    val_times = tuple(day for day in timepoints if day in requested)
    test_times = tuple(
        day for day in timepoints if day not in retained and day not in requested
    )
    times = {day: maizels.normalized_time(day, cfg) for day in timepoints}
    if any(times[left] >= times[right] for left, right in zip(timepoints, timepoints[1:])):
        raise ValueError("Maizels marginal model times must be strictly increasing.")
    return Marginals(retained, timepoints, val_times, test_times, populations, times)


def evaluate_marginals(
    marginals: Marginals,
    pushforward: Callable,
    *,
    context: Dict,
    score: Callable = None,
    progress: Callable = None,
) -> List[Dict]:
    """Score interval restarts and a rollout carried only across observations.

    Each target uses the requested sampler from its preceding retained day.
    Evaluating an omitted marginal never changes the carried rollout state.
    """
    if score is None:
        score = wasserstein.exact_emd
    rows = []
    source_day = marginals.retained[0]

    def add_row(protocol, day, prediction, w1):
        observed = day in marginals.retained
        row = {
            **context,
            "protocol": protocol,
            "timepoint": day,
            "partition": "observed" if observed else (
                "hparam_val" if day in marginals.hparam_val_times else "test"
            ),
            "n_timepoints": 1,
            "source_timepoint": source_day if protocol == "rollout" else left,
            "source_n": int(len(prediction)),
            "target_n": int(len(marginals.populations[day])),
            "w1": float(w1),
        }
        rows.append(row)

    left = source_day
    rollout = marginals.populations[source_day]
    for protocol in ("interval", "rollout"):
        add_row(protocol, source_day, rollout, 0.0)

    for left, right in zip(marginals.retained[:-1], marginals.retained[1:]):
        # Hold the state at the observed left endpoint fixed for all targets in
        # this interval; only the observed right endpoint advances the rollout.
        rollout_at_left = rollout
        for index, day in enumerate(marginals.timepoints):
            if not (marginals.times[left] < marginals.times[day] <= marginals.times[right]):
                continue
            kwargs = {
                "start_time": marginals.times[left],
                "end_time": marginals.times[day],
                "target_index": index,
            }
            prediction = pushforward(
                marginals.populations[left], protocol="interval", **kwargs
            )
            if not np.isfinite(prediction).all():
                raise FloatingPointError(f"Non-finite interval predictions at {day}.")
            interval_w1 = score(prediction, marginals.populations[day])
            add_row("interval", day, prediction, interval_w1)
            if left == source_day:
                rollout_prediction, rollout_w1 = prediction, interval_w1
            else:
                rollout_prediction = pushforward(
                    rollout_at_left, protocol="rollout", **kwargs
                )
                if not np.isfinite(rollout_prediction).all():
                    raise FloatingPointError(f"Non-finite rollout predictions at {day}.")
                rollout_w1 = score(rollout_prediction, marginals.populations[day])
            add_row("rollout", day, rollout_prediction, rollout_w1)
            if day == right:
                rollout = rollout_prediction
            if progress is not None:
                progress(day, interval_w1, rollout_w1)

    for protocol in ("interval", "rollout"):
        values = [
            row["w1"] for row in rows
            if row["protocol"] == protocol and row["timepoint"] in marginals.test_times
        ]
        if values:
            rows.append({
                **context,
                "protocol": protocol,
                "timepoint": "test_times",
                "partition": "test",
                "n_timepoints": len(values),
                "source_timepoint": source_day if protocol == "rollout" else "mixed",
                "source_n": None,
                "target_n": None,
                "w1": float(np.mean(values)),
            })
    return rows


def summary_rows(
    rows: Iterable[Dict], group_fields: Tuple[str, ...], count_field: str = "n"
) -> List[Dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(float(row["w1"]))
    summaries = []
    for group, values in sorted(grouped.items()):
        array = np.asarray(values, dtype=np.float64)
        sd = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
        summaries.append({
            **dict(zip(group_fields, group)),
            count_field: int(array.size),
            "w1_mean": float(np.mean(array)),
            "w1_sd": sd,
            "w1_se": sd / math.sqrt(array.size),
        })
    return summaries


def metric_records(rows: Iterable[Dict], value_field: str = "w1") -> List[Dict]:
    """Expose training-compatible metric names for each evaluated checkpoint."""
    grouped = {}
    fields = ("method", "seed", "checkpoint", "sampler", "n_steps_per_interval")
    for row in rows:
        identity = tuple(row[field] for field in fields)
        record = grouped.setdefault(identity, {**dict(zip(fields, identity)), "metrics": {}})
        prefix = "final_eval" if row["protocol"] == "interval" else "rollout_eval"
        suffix = (
            "mean_emd_test_times" if row["timepoint"] == "test_times"
            else "emd_" + row["timepoint"].replace(".", "p")
        )
        record["metrics"][f"{prefix}/{row['sampler']}_{suffix}"] = float(row[value_field])
    return list(grouped.values())


def write_csv(path: Path, rows: List[Dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
