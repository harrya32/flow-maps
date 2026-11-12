"""All-cell, interval-only checkpoint evaluation for CITE/Multi."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from . import cite_multi
from .maizels_checkpoint_eval import (
    Marginals,
    evaluate_marginals,
    metric_records,
    summary_rows,
    write_csv,
)


def heldout_day(value) -> str:
    day = str(int(cite_multi.parse_timepoint(str(value))))
    if day not in ("3", "4") or cite_multi.parse_timepoint(str(value)) != float(day):
        raise ValueError("The held-out CITE/Multi day must be 3 or 4.")
    return day


def build_marginals(cfg, data) -> Marginals:
    """Include every day; the omitted observation is the test-time partition."""
    omitted = heldout_day(cfg.problem.heldout_timepoint)
    retained = cite_multi.retained_timepoints(cfg)
    saved_retained = getattr(cfg.problem, "retained_timepoints", retained)
    if tuple(map(str, saved_retained)) != retained:
        raise ValueError("Retained CITE/Multi days disagree with the held-out day.")
    populations = {
        day: np.asarray(data["x"][data["timepoints"] == day], dtype=np.float32)
        for day in cite_multi.TIMEPOINTS
    }
    empty = [day for day, population in populations.items() if not len(population)]
    if empty:
        raise ValueError(f"Missing CITE/Multi marginal populations: {empty}.")
    if any(not np.isfinite(x).all() for x in populations.values()):
        raise ValueError(
            "CITE/Multi marginal populations contain non-finite coordinates."
        )
    times = {day: cite_multi.normalized_time(day, cfg) for day in cite_multi.TIMEPOINTS}
    if any(not np.isfinite(t) or not 0.0 <= t <= 1.0 for t in times.values()) or any(
        times[left] >= times[right]
        for left, right in zip(cite_multi.TIMEPOINTS, cite_multi.TIMEPOINTS[1:])
    ):
        raise ValueError("CITE/Multi model times must increase strictly within [0, 1].")
    return Marginals(
        retained, cite_multi.TIMEPOINTS, (), (omitted,), populations, times
    )


def _saved_heldout(config):
    for key in ("heldout_day", "heldout_timepoint"):
        value = config.get(key)
        if value is not None and str(value).lower() not in ("", "none"):
            return heldout_day(value)
    index = config.get("t_exclude_current")
    if index is None:
        excluded = config.get("t_exclude")
        if isinstance(excluded, (list, tuple)) and len(excluded) == 1:
            index = excluded[0]
        elif excluded is not None and not isinstance(excluded, (list, tuple)):
            index = excluded
    if index is not None:
        if (
            isinstance(index, bool)
            or int(index) not in (1, 2)
            or float(index) != int(index)
        ):
            raise ValueError(
                "MFM t_exclude must be observation index 1 (day 3) or 2 (day 4)."
            )
        return cite_multi.TIMEPOINTS[int(index)]
    retained = config.get("retained_days")
    if retained is not None:
        missing = set(cite_multi.TIMEPOINTS) - set(map(str, retained))
        if len(missing) == 1:
            return heldout_day(missing.pop())
    return None


def torch_config(
    config,
    backend,
    *,
    dataset_location="",
    dataset_name=None,
    omitted_day=None,
    time_mode=None,
):
    """Use the saved OTP-FM/MFM task and clock without rebuilding training data."""
    saved_dataset = (
        config.get("dataset_name") if backend == "otpfm" else config.get("data_name")
    )
    if backend == "mfm" and config.get("data_type") not in (None, "scrna"):
        raise ValueError("Expected an MFM CITE/Multi scrna training configuration.")
    if dataset_name is None and saved_dataset is None:
        raise ValueError(
            "The checkpoint config has no CITE/Multi dataset; pass --dataset_name."
        )
    name = cite_multi.canonical_dataset_name(dataset_name or saved_dataset)
    if (
        saved_dataset is not None
        and cite_multi.canonical_dataset_name(saved_dataset) != name
    ):
        raise ValueError("--dataset_name disagrees with the saved training dataset.")
    saved_omitted = _saved_heldout(config)
    if omitted_day is None and saved_omitted is None:
        raise ValueError(
            "The saved config does not identify one fold; pass --heldout_day 3 or 4."
        )
    omitted = heldout_day(omitted_day if omitted_day is not None else saved_omitted)
    if saved_omitted is not None and omitted != saved_omitted:
        raise ValueError("--heldout_day disagrees with the saved training fold.")
    retained = tuple(day for day in cite_multi.TIMEPOINTS if day != omitted)
    if (
        config.get("retained_days") is not None
        and tuple(map(str, config["retained_days"])) != retained
    ):
        raise ValueError(
            "Saved retained days disagree with the CITE/Multi training fold."
        )
    clock = cite_multi.canonical_time_mode(
        time_mode or config.get("cite_multi_time_mode") or config.get("time_mode")
    )
    if config.get("time_map") is not None and time_mode is None:
        values = [float(config["time_map"][day]) for day in cite_multi.TIMEPOINTS]
    elif backend == "mfm":
        # TemporalDataModule stores its clock in float32; keep the same values.
        days = np.asarray(cite_multi.TIMEPOINTS, dtype=np.float32)
        values = (
            (
                np.linspace(0.0, 1.0, len(days), dtype=np.float32)
                if clock == "equal_time"
                else (days - days[0]) / (days[-1] - days[0])
            )
            .astype(float)
            .tolist()
        )
    else:
        values = list(cite_multi.normalized_time_map(clock).values())
    location = (
        dataset_location
        or config.get("dataset_path")
        or config.get("data_path")
        or config.get("dataset_location")
        or "."
    )
    return SimpleNamespace(
        problem=SimpleNamespace(
            dataset_name=name,
            dataset_location=str(cite_multi.resolve_dataset_path(location, name)),
            heldout_timepoint=omitted,
            retained_timepoints=retained,
            cite_multi_time_mode=clock,
            timepoint_order=cite_multi.TIMEPOINTS,
            timepoint_values=values,
        )
    )
