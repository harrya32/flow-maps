"""Evaluate deterministic CITE/Multi flow-map, OTP-FM, and MFM checkpoints.

Every target starts from the real population at its preceding retained day.
Report only composed maps, or Euler when selected; MFM always uses Euler.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
for directory in (REPO_ROOT, REPO_ROOT / "py"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from common import cite_multi
from common import cite_multi_checkpoint_eval as checkpoint_eval


def _checkpoint_spec(value):
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "checkpoint must have the form METHOD:SEED:PATH"
        )
    method, seed_text, path_text = parts
    try:
        seed = int(seed_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("checkpoint seed must be an integer") from exc
    if seed < 0:
        raise argparse.ArgumentTypeError("checkpoint seed must be non-negative")
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"checkpoint does not exist: {path}")
    return method, seed, path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("flowmaps", "otpfm", "mfm"), required=True
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        type=_checkpoint_spec,
        metavar="METHOD:SEED:PATH",
    )
    parser.add_argument(
        "--dataset_location",
        default="",
        help="Dataset directory or .h5ad; overrides the saved path.",
    )
    parser.add_argument(
        "--dataset_name",
        choices=("cite", "multi"),
        help="Defaults to the saved training dataset.",
    )
    parser.add_argument(
        "--heldout_day",
        choices=("3", "4"),
        help="Required when the saved config does not identify one fold.",
    )
    parser.add_argument(
        "--cite_multi_time_mode",
        choices=cite_multi.TIME_MODES,
        help="Defaults to the training clock.",
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        help="Saved full native config, OTP-FM experiment.json, or MFM training YAML.",
    )
    parser.add_argument(
        "--cfg_path",
        default="configs.cite_multi_pca100",
        help="Native training config module, used without --config_path.",
    )
    parser.add_argument(
        "--slurm_id",
        type=int,
        help="Native training preset ID, used without --config_path.",
    )
    parser.add_argument(
        "--n_steps_per_interval",
        type=int,
        nargs="+",
        default=[50],
        metavar="N",
        help="Step counts; each checkpoint is restored once for all counts.",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Evaluate only the held-out day and its test_times aggregate.",
    )
    parser.add_argument(
        "--euler",
        action="store_true",
        help="Select Euler for native flow matching or OTP-FM; MFM always uses Euler.",
    )
    parser.add_argument(
        "--ema",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="OTP-FM/MFM EMA override; otherwise match training evaluation.",
    )
    parser.add_argument(
        "--ema_fac",
        type=float,
        help="Native EMA decay; native evaluation otherwise uses instantaneous parameters.",
    )
    parser.add_argument(
        "--rescale",
        type=float,
        help="Native training scale override; otherwise reproduce the training-pool/pair scale.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda or cuda:INDEX; Torch also supports mps.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Inference batch size; every source and target cell is evaluated.",
    )
    parser.add_argument("--out_dir", default="outputs/cite_multi_checkpoint_eval")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or any(steps <= 0 for steps in args.n_steps_per_interval):
        parser.error("batch_size and every n_steps_per_interval value must be positive")
    if args.rescale is not None and (
        not np.isfinite(args.rescale) or args.rescale <= 0
    ):
        parser.error("rescale must be finite and positive")
    if args.ema_fac is not None and not 0.0 <= args.ema_fac <= 1.0:
        parser.error("ema_fac must lie within [0, 1]")
    if args.backend == "flowmaps":
        if args.ema is not None:
            parser.error(
                "Use --ema_fac for native checkpoints; --ema/--no-ema apply to OTP-FM/MFM"
            )
        if args.config_path is None and any(
            value is None
            for value in (args.slurm_id, args.dataset_name, args.heldout_day)
        ):
            parser.error(
                "Native evaluation needs --slurm_id, --dataset_name and --heldout_day, or a full --config_path"
            )
    elif (
        args.ema_fac is not None
        or args.rescale is not None
        or args.slurm_id is not None
    ):
        parser.error("ema_fac, rescale and slurm_id apply to the flowmaps backend")
    args.n_steps_per_interval = list(dict.fromkeys(args.n_steps_per_interval))
    return args


def _whitened_sampler(model_sample, scaler):
    def sample(source, **kwargs):
        prediction = model_sample(scaler.transform(source), **kwargs)
        return scaler.inverse_transform(prediction).astype(np.float32)

    return sample


def main(argv=None):
    args = parse_args(argv)
    # Keep the framework imports separate so Torch evaluation needs no JAX,
    # and native evaluation needs no OTP-FM/MFM installation.
    if args.backend == "flowmaps":
        from common import cite_multi_flowmaps_checkpoint as loader
    else:
        from common import maizels_torch_checkpoint as loader
    device = loader.resolve_device(args.device)
    sampler = "euler" if args.backend == "mfm" or args.euler else "flowmap"
    rows = []
    checkpoint_metadata = []
    expected_marginals = None
    expected_task = None
    dataset_locations = set()
    for method, seed, path in args.checkpoint:
        print(f"Restoring {args.backend} {method} seed={seed}: {path}", flush=True)
        if args.backend == "flowmaps":
            cfg, config_source = loader.training_config(
                seed=seed,
                dataset_location=args.dataset_location,
                dataset_name=args.dataset_name,
                omitted_day=args.heldout_day,
                time_mode=args.cite_multi_time_mode,
                config_path=args.config_path,
                cfg_path=args.cfg_path,
                slurm_id=args.slurm_id,
            )
            data = cite_multi.all_timepoint_data(
                cfg.problem.dataset_location, cfg.problem.dataset_name
            )
            marginals = checkpoint_eval.build_marginals(cfg, data)
            loaded = loader.load_checkpoint(
                path,
                cfg,
                device=device,
                batch_size=args.batch_size,
                ema_fac=args.ema_fac,
                rescale=args.rescale,
            )
            sample = loaded.sample
            loaded.metadata["config_path"] = config_source
        else:
            loaded = loader.load_checkpoint(
                args.backend,
                path,
                device=device,
                batch_size=args.batch_size,
                config_path=args.config_path,
                ema=args.ema,
            )
            cfg = checkpoint_eval.torch_config(
                loaded.config,
                args.backend,
                dataset_location=args.dataset_location,
                dataset_name=args.dataset_name,
                omitted_day=args.heldout_day,
                time_mode=args.cite_multi_time_mode,
            )
            data = cite_multi.all_timepoint_data(
                cfg.problem.dataset_location, cfg.problem.dataset_name
            )
            marginals = checkpoint_eval.build_marginals(cfg, data)
            sample = loaded.sampler.sample
            if args.backend == "mfm":
                whitening = bool(loaded.config.get("whiten", True))
                loaded.metadata["whiten"] = whitening
                if whitening:
                    from sklearn.preprocessing import StandardScaler

                    # TemporalDataModule fits on all PCA rows before splitting.
                    scaler = StandardScaler().fit(data["x"])
                    sample = _whitened_sampler(sample, scaler)
                    loaded.metadata["whitening_mean"] = scaler.mean_.tolist()
                    loaded.metadata["whitening_scale"] = scaler.scale_.tolist()
        if loaded.dimension != 100 or data["x"].shape[1] != loaded.dimension:
            raise ValueError(
                f"CITE/Multi evaluation requires a PCA100 checkpoint; found dimension {loaded.dimension}."
            )
        task = {
            "dataset": cfg.problem.dataset_name,
            "heldout_day": str(cfg.problem.heldout_timepoint),
            "time_mode": cite_multi.time_mode_from_config(cfg),
        }
        if expected_marginals is None:
            expected_marginals, expected_task = marginals.metadata(), task
        elif marginals.metadata() != expected_marginals or task != expected_task:
            raise ValueError(
                "One invocation must contain checkpoints for the same dataset, held-out day and clock."
            )
        dataset_locations.add(
            str(Path(cfg.problem.dataset_location).expanduser().resolve())
        )
        checkpoint_metadata.append(
            {"method": method, "seed": seed, "path": str(path), **loaded.metadata}
        )
        for n_steps in args.n_steps_per_interval:
            print(
                f"  Evaluating {sampler} with {n_steps} steps per source-to-target interval",
                flush=True,
            )

            def pushforward(source, *, start_time, end_time, target_index):
                return sample(
                    source,
                    start_time=start_time,
                    end_time=end_time,
                    n_steps=n_steps,
                    sampler=sampler,
                )

            def progress(day, w1):
                print(f"    day {day}: interval W1={w1:.6f}", flush=True)

            rows.extend(
                checkpoint_eval.evaluate_marginals(
                    marginals,
                    pushforward,
                    context={
                        **task,
                        "method": method,
                        "seed": seed,
                        "checkpoint": str(path),
                        "backend": args.backend,
                        "sampler": sampler,
                        "n_steps_per_interval": n_steps,
                    },
                    progress=progress,
                    test_only=args.test_only,
                )
            )
        del loaded, sample

    summary = checkpoint_eval.summary_rows(
        rows,
        (
            "dataset",
            "heldout_day",
            "time_mode",
            "backend",
            "method",
            "sampler",
            "n_steps_per_interval",
            "protocol",
            "timepoint",
        ),
        count_field="n_checkpoints",
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_eval.write_csv(out_dir / "per_checkpoint.csv", rows)
    checkpoint_eval.write_csv(out_dir / "summary.csv", summary)
    records = checkpoint_eval.metric_records(rows)
    for record in records:
        record.update(expected_task)
        record["backend"] = args.backend
    (out_dir / "metrics.json").write_text(json.dumps(records, indent=2) + "\n")
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                **expected_task,
                **expected_marginals,
                "backend": args.backend,
                "sampler": sampler,
                "device": str(device),
                "metric_space": "raw PCA100",
                "n_steps_per_interval": args.n_steps_per_interval,
                "test_only": args.test_only,
                "evaluated_timepoints": (
                    expected_marginals["test_times"]
                    if args.test_only
                    else expected_marginals["timepoints"]
                ),
                "dataset_locations": sorted(dataset_locations),
                "protocols": {
                    "interval": "start from the real population at the preceding observed day"
                },
                "checkpoints": checkpoint_metadata,
            },
            indent=2,
        )
        + "\n"
    )
    for row in summary:
        if row["timepoint"] == "test_times":
            print(
                f"{row['method']} ({row['n_steps_per_interval']} steps): final_eval/{sampler}_mean_emd_test_times={row['w1_mean']:.6f} +/- {row['w1_se']:.6f}",
                flush=True,
            )
    print(f"Saved results to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
