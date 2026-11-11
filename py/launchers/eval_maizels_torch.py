"""Evaluate pretrained Maizels OTP-FM or MFM models at every marginal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
for directory in (REPO_ROOT, REPO_ROOT / "py"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from common import maizels
from common import maizels_checkpoint_eval as checkpoint_eval
from common import maizels_torch_checkpoint as torch_checkpoint


def _checkpoint_spec(value):
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("checkpoint must have the form METHOD:SEED:PATH")
    method, seed_text, path_text = parts
    try:
        seed = int(seed_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("checkpoint seed must be an integer") from exc
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"checkpoint does not exist: {path}")
    return method, seed, path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("otpfm", "mfm"), required=True)
    parser.add_argument(
        "--checkpoint", action="append", required=True, type=_checkpoint_spec,
        metavar="METHOD:SEED:PATH",
    )
    parser.add_argument(
        "--config_path", type=Path,
        help="Saved experiment JSON or MFM training YAML; otherwise discover it beside the checkpoint.",
    )
    parser.add_argument(
        "--dataset_location", default="",
        help="Dataset directory or PCA50 CSV; overrides the saved dataset path.",
    )
    parser.add_argument(
        "--n_steps_per_interval", type=int, nargs="+", default=[50], metavar="N",
        help="One or more step counts, e.g. 1 2 4 8 16 32 64. Each model is loaded once.",
    )
    parser.add_argument(
        "--euler", action="store_true",
        help="Select Euler for OTP-FM; MFM always uses Euler. OTP-FM defaults to composed maps.",
    )
    parser.add_argument(
        "--ema", action=argparse.BooleanOptionalAction, default=None,
        help="Override EMA selection; defaults to the weights used by training evaluation.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:INDEX, or mps.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Inference batch size; every source cell is evaluated.")
    parser.add_argument("--maizels_schedule", choices=("d3_d3p8_d8", "d3_d8"))
    parser.add_argument("--maizels_time_mode", choices=("real_time", "equal_time"))
    parser.add_argument(
        "--hparam_val_times", default=None,
        help="Validation days excluded from test_times; defaults to saved settings or D3.4,D6.",
    )
    parser.add_argument("--out_dir", default="outputs/maizels_torch_eval")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or any(steps <= 0 for steps in args.n_steps_per_interval):
        parser.error("batch_size and every n_steps_per_interval value must be positive")
    args.n_steps_per_interval = list(dict.fromkeys(args.n_steps_per_interval))
    return args


def main(argv=None):
    args = parse_args(argv)
    device = torch_checkpoint.resolve_device(args.device)
    sampler = "euler" if args.backend == "mfm" or args.euler else "flowmap"
    rows = []
    checkpoint_metadata = []
    expected_marginals = None
    dataset_locations = set()
    for method, seed, path in args.checkpoint:
        print(f"Restoring {args.backend} {method} seed={seed}: {path}", flush=True)
        loaded = torch_checkpoint.load_checkpoint(
            args.backend, path, device=device, batch_size=args.batch_size,
            config_path=args.config_path, ema=args.ema,
        )
        cfg = torch_checkpoint.maizels_config(
            loaded.config, args.backend, dataset_location=args.dataset_location,
            schedule=args.maizels_schedule, time_mode=args.maizels_time_mode,
            hparam_val_times=args.hparam_val_times,
        )
        data = maizels.all_timepoint_data(cfg.problem.dataset_location)
        if data["x"].shape[1] != loaded.dimension:
            raise ValueError(
                f"Checkpoint dimension {loaded.dimension} does not match "
                f"dataset dimension {data['x'].shape[1]}."
            )
        marginals = checkpoint_eval.build_marginals(cfg, data)
        if expected_marginals is None:
            expected_marginals = marginals.metadata()
        elif expected_marginals != marginals.metadata():
            raise ValueError("Checkpoints use different marginal populations, clocks, or validation days.")
        dataset_locations.add(str(Path(cfg.problem.dataset_location).expanduser().resolve()))
        checkpoint_metadata.append({
            "method": method, "seed": seed, "path": str(path),
            **loaded.metadata,
            "maizels_schedule": cfg.problem.maizels_schedule,
            "time_mode": cfg.problem.maizels_time_mode,
        })
        for n_steps in args.n_steps_per_interval:
            print(f"  Evaluating {sampler} with {n_steps} steps per source-to-target interval", flush=True)

            def pushforward(source, *, start_time, end_time, target_index):
                return loaded.sampler.sample(
                    source, start_time=start_time, end_time=end_time,
                    n_steps=n_steps, sampler=sampler,
                )

            def progress(day, w1):
                print(f"    {day}: interval W1={w1:.6f}", flush=True)

            rows.extend(checkpoint_eval.evaluate_marginals(
                marginals, pushforward,
                context={
                    "method": method, "seed": seed, "checkpoint": str(path),
                    "backend": args.backend, "sampler": sampler,
                    "n_steps_per_interval": n_steps,
                },
                progress=progress,
            ))
        del loaded

    summary = checkpoint_eval.summary_rows(
        rows,
        ("backend", "method", "sampler", "n_steps_per_interval", "protocol", "timepoint"),
        count_field="n_checkpoints",
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_eval.write_csv(out_dir / "per_checkpoint.csv", rows)
    checkpoint_eval.write_csv(out_dir / "summary.csv", summary)
    (out_dir / "metrics.json").write_text(
        json.dumps(checkpoint_eval.metric_records(rows), indent=2) + "\n"
    )
    (out_dir / "metadata.json").write_text(json.dumps({
        **expected_marginals,
        "backend": args.backend, "sampler": sampler, "device": str(device),
        "n_steps_per_interval": args.n_steps_per_interval,
        "dataset_locations": sorted(dataset_locations),
        "protocols": {"interval": "start from the real population at the preceding observed day"},
        "checkpoints": checkpoint_metadata,
    }, indent=2) + "\n")
    for row in summary:
        if row["timepoint"] == "test_times":
            print(
                f"{row['method']} ({row['n_steps_per_interval']} steps): "
                f"final_eval/{sampler}_mean_emd_test_times="
                f"{row['w1_mean']:.6f} +/- {row['w1_se']:.6f}",
                flush=True,
            )
    print(f"Saved results to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
