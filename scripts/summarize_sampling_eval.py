#!/usr/bin/env python3
"""Write one method's test-time EMD at sampling steps 1, 2, 4, 8, 16, 32, 64.

Read interval/test_times rows from summary.csv in the supplied directory or
its step-count subdirectories. Values are the evaluator's w1_mean, including
its averaging over noise draws and model seeds. No training libraries are needed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import sys


STEPS = (1, 2, 4, 8, 16, 32, 64)
COLUMNS = ("method", "1 sample", *(f"{step} samples" for step in STEPS[1:]))
REQUIRED_FIELDS = {
    "method",
    "sampler",
    "protocol",
    "timepoint",
    "n_steps_per_interval",
    "w1_mean",
}


@dataclass(frozen=True)
class Evaluation:
    method: str
    sampler: str
    steps: int
    emd: float
    source: Path


def sampling_row(folder, *, method=None, sampler=None):
    """Collect one summary per requested step count, leaving missing cells empty."""
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(
            f"Sampling evaluation directory does not exist: {folder}"
        )
    paths = sorted(folder.rglob("summary.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No evaluation summary.csv files found under {folder}."
        )

    evaluations = []
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_FIELDS - set(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"{path} is missing evaluation columns {sorted(missing)}. "
                    "Use results from the updated sampling evaluator."
                )
            for line, row in enumerate(reader, start=2):
                if row["protocol"] != "interval" or row["timepoint"] != "test_times":
                    continue
                if sampler is not None and row["sampler"] != sampler:
                    continue
                try:
                    steps = int(row["n_steps_per_interval"])
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"Invalid step count in {path}:{line}.") from exc
                if steps not in STEPS:
                    continue
                try:
                    value = float(row["w1_mean"])
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"Invalid mean EMD in {path}:{line}.") from exc
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite mean EMD in {path}:{line}.")
                saved_method = (row["method"] or "").strip()
                saved_sampler = (row["sampler"] or "").strip()
                if not saved_method or not saved_sampler:
                    raise ValueError(f"Missing method or sampler in {path}:{line}.")
                evaluations.append(
                    Evaluation(saved_method, saved_sampler, steps, value, path)
                )

    if not evaluations:
        raise ValueError(
            "No interval/test_times summaries found for sampling steps "
            f"{', '.join(map(str, STEPS))}"
            + (f" with sampler {sampler!r}." if sampler else ".")
        )
    methods = {item.method for item in evaluations}
    if len(methods) != 1:
        raise ValueError(
            f"Found multiple methods {sorted(methods)}; point to one method's evaluation folder."
        )
    samplers = {item.sampler for item in evaluations}
    if len(samplers) != 1:
        raise ValueError(
            f"Found multiple samplers {sorted(samplers)}; select one with --sampler."
        )
    label = next(iter(methods)) if method is None else method.strip()
    if not label:
        raise ValueError("The method label must not be empty.")

    by_step = {}
    for item in evaluations:
        if item.steps in by_step:
            previous = by_step[item.steps]
            raise ValueError(
                f"Found multiple results for {item.steps} steps in "
                f"{previous.source} and {item.source}. "
                "Point to one evaluation per step count, or first evaluate the seeds together."
            )
        by_step[item.steps] = item
    return {
        "method": label,
        **{
            column: by_step[step].emd if step in by_step else None
            for step, column in zip(STEPS, COLUMNS[1:])
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sampling_eval", type=Path, help="One method's sampling evaluation directory."
    )
    parser.add_argument(
        "--method",
        help="Override the output row's method label; defaults to the saved label.",
    )
    parser.add_argument(
        "--sampler",
        choices=("flowmap", "euler", "euler_maruyama"),
        help="Select a sampler when the folder contains results for more than one.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV; defaults to <sampling_eval>/sampling_summary.csv.",
    )
    args = parser.parse_args(argv)
    folder = args.sampling_eval.expanduser().resolve()
    output = (args.output or folder / "sampling_summary.csv").expanduser().resolve()
    try:
        row = sampling_row(folder, method=args.method, sampler=args.sampler)
        if output in folder.rglob("summary.csv"):
            raise ValueError(
                "Output would overwrite an evaluation summary.csv; choose another filename."
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerow(row)
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))

    missing_steps = [
        step for step, column in zip(STEPS, COLUMNS[1:]) if row[column] is None
    ]
    if missing_steps:
        print(
            f"Missing step counts {missing_steps}; those CSV fields are empty.",
            file=sys.stderr,
        )
    print(f"Saved {row['method']} sampling summary to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
