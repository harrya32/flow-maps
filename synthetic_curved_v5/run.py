"""Command-line entry point for the configurable curved-v5 benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .classifier import sha256, train_classifier
from .config import DEFAULT_CONFIG, load_config, write_config
from .dataset import prepare_dataset
from .evaluation import evaluate_all
from .models import prepare_learning_inputs
from .training import METHODS, train_all


def _csv(value, convert=str):
    if value is None:
        return None
    return [convert(item.strip()) for item in value.split(",") if item.strip()]


def prepare(root, config):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    write_config(root / "resolved_config.json", config)
    package = Path(__file__).resolve().parent
    (root / "provenance.json").write_text(
        json.dumps(
            {
                "upstream_repository": "https://github.com/BenjaminisCoding/flow_maps_traj.git",
                "upstream_commit": "b0ec335399bf0fbcd958b8d68e72282dde21a402",
                "resolved_config_sha256": sha256(root / "resolved_config.json"),
                "port_sources": {
                    path.name: sha256(path) for path in sorted(package.glob("*.py"))
                },
            },
            indent=2,
        )
        + "\n"
    )
    prepare_dataset(root / "dataset", config["dataset"])
    train_classifier(
        root / "dataset" / "training_data.npz",
        root / "classifier",
        config["classifier"],
        int(config["seeds"][0]),
    )
    prepare_learning_inputs(
        root / "dataset" / "training_data.npz",
        root / "classifier",
        root / "inputs",
        config["learning_split"],
        int(config["seeds"][0]),
    )
    (root / "preparation.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "training_times": config["dataset"]["training_times"],
                "methods": list(METHODS),
                "priors": list(config["priors"]),
                "seeds": config["seeds"],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Prepared benchmark at {root}", flush=True)


def _prepared_config(root, supplied, overrides):
    resolved = Path(root) / "resolved_config.json"
    if not resolved.exists():
        raise FileNotFoundError(f"No prepared benchmark at {root}; run the prepare stage first.")
    provenance_file = Path(root) / "provenance.json"
    if not provenance_file.exists():
        raise FileNotFoundError(f"Prepared benchmark is missing {provenance_file}.")
    provenance = json.loads(provenance_file.read_text())
    if sha256(resolved) != provenance["resolved_config_sha256"]:
        raise ValueError("resolved_config.json changed after preparation; prepare a new output root.")
    package = Path(__file__).resolve().parent
    changed_sources = [
        name
        for name, digest in provenance["port_sources"].items()
        if not (package / name).exists() or sha256(package / name) != digest
    ]
    if changed_sources:
        raise ValueError(
            f"Benchmark source changed after preparation: {changed_sources}. Prepare a new output root."
        )
    stored = load_config(resolved)
    if supplied is not None or overrides:
        requested = load_config(DEFAULT_CONFIG if supplied is None else supplied, overrides)
        if requested != stored:
            raise ValueError(
                "Training/evaluation settings differ from resolved_config.json. "
                "Prepare a new output root for a changed configuration."
            )
    return stored


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "evaluate", "all"))
    parser.add_argument("--root", type=Path, required=True, help="New or prepared experiment output root.")
    parser.add_argument("--config", type=Path, help=f"JSON config (default: {DEFAULT_CONFIG}).")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted config key; repeat for multiple changes.",
    )
    parser.add_argument(
        "--methods",
        help="Comma-separated subset: sf2m,ssfm,clift,real_clift.",
    )
    parser.add_argument("--seeds", help="Comma-separated training-seed subset.")
    parser.add_argument("--priors", help="Comma-separated prior-name subset.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    methods = _csv(args.methods)
    seeds = _csv(args.seeds, int)
    priors = _csv(args.priors)
    if args.stage in ("prepare", "all"):
        config = load_config(DEFAULT_CONFIG if args.config is None else args.config, args.overrides)
        prepare(root, config)
    else:
        config = _prepared_config(root, args.config, args.overrides)
    if args.stage in ("train", "all"):
        train_all(
            root,
            config,
            methods=METHODS if methods is None else methods,
            seeds=seeds,
            priors=priors,
            device=args.device,
        )
    if args.stage in ("evaluate", "all"):
        evaluate_all(
            root,
            config,
            methods=METHODS if methods is None else methods,
            seeds=seeds,
            priors=priors,
            device=args.device,
        )


if __name__ == "__main__":
    main()
