"""Cache-aware classifier preparation before Schiebinger flow training."""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, List


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER = REPO_ROOT / "scripts" / "train_schiebinger_celltype_classifiers.py"


def _classifier_python_works(executable: Path) -> bool:
    """Return whether an interpreter can import the trainer dependencies."""
    if not executable.is_file():
        return False
    probe = (
        "import anndata, matplotlib, pandas, sklearn, torch; "
        "assert torch.__version__"
    )
    cache_root = Path(tempfile.gettempdir()) / "flow_maps_numba_cache"
    matplotlib_root = Path(tempfile.gettempdir()) / "flow_maps_matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    matplotlib_root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.setdefault("NUMBA_CACHE_DIR", str(cache_root))
    environment.setdefault("MPLCONFIGDIR", str(matplotlib_root))
    try:
        result = subprocess.run(
            [str(executable), "-c", probe],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def resolve_classifier_python(configured: str | None = None) -> str:
    """Choose a Python interpreter that can run the PyTorch trainer.

    Flow training is commonly launched from the ``flowmaps`` environment,
    which does not contain PyTorch.  An explicit config/environment override
    wins; otherwise use the current interpreter when it has every trainer
    dependency, then look for the repository's usual sibling Conda environments.
    """
    requested = configured or os.getenv("SCHIEBINGER_CLASSIFIER_PYTHON", "")
    if requested:
        path = Path(requested).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                "SCHIEBINGER_CLASSIFIER_PYTHON does not point to an existing "
                f"interpreter: {path}"
            )
        path = path.resolve()
        if not _classifier_python_works(path):
            raise RuntimeError(
                "The configured Schiebinger classifier Python cannot import "
                "torch, anndata, scikit-learn, pandas, and matplotlib: "
                f"{path}"
            )
        return str(path)

    current = Path(sys.executable).resolve()
    if _classifier_python_works(current):
        return str(current)

    env_roots = []
    nested_envs = current.parent.parent.parent
    if nested_envs.name == "envs":
        env_roots.append(nested_envs)
    base_envs = current.parent.parent / "envs"
    if base_envs.is_dir():
        env_roots.append(base_envs)
    candidates = [
        env_root / name / "bin" / "python"
        for env_root in env_roots
        for name in ("mfm_env", "torchcfm", "maizels2023aa")
    ]
    for candidate in candidates:
        if _classifier_python_works(candidate):
            return str(candidate.resolve())

    raise RuntimeError(
        "The active flow-training Python has no PyTorch, and no compatible "
        "sibling Conda environment was found. Set SCHIEBINGER_CLASSIFIER_PYTHON "
        "to a Python containing torch, anndata, scikit-learn, pandas, and "
        "matplotlib."
    )


def checkpoint_pair_complete(pt_path: str | Path) -> bool:
    path = Path(pt_path).expanduser().resolve()
    return path.is_file() and path.with_suffix(".npz").is_file()


def checkpoint_pair_partial(pt_path: str | Path) -> bool:
    path = Path(pt_path).expanduser().resolve()
    present = (path.is_file(), path.with_suffix(".npz").is_file())
    return any(present) and not all(present)


def ensure_schiebinger_classifiers(
    cfg,
    *,
    runner: Callable = subprocess.run,
    python_executable: str | None = None,
) -> None:
    """Train missing all-days and, when needed, selected-days classifiers."""
    if getattr(cfg.problem, "target", None) != "schiebinger":
        return
    if not bool(getattr(cfg.problem, "schiebinger_auto_train_classifiers", True)):
        return

    full_path = Path(
        str(cfg.logging.maizels.full_data_classifier_path)
    ).expanduser().resolve()
    training_required = bool(
        getattr(cfg.problem, "flow_training_requires_classifier", False)
    )
    training_path = Path(
        str(getattr(cfg.problem, "training_classifier_path", cfg.problem.classifier_path))
    ).expanduser().resolve()

    required_paths = [full_path]
    if training_required:
        required_paths.append(training_path)
    missing = [path for path in required_paths if not checkpoint_pair_complete(path)]
    if not missing:
        print("Reusing cached Schiebinger classifier checkpoints.")
        return

    # Different slurm variants commonly share the same selected-day schedule.
    # Serialize this one-time preparation so simultaneous jobs do not train or
    # overwrite the same classifier pair concurrently.
    full_path.parent.mkdir(parents=True, exist_ok=True)
    with (full_path.parent / ".classifier_training.lock").open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            missing = [
                path for path in required_paths if not checkpoint_pair_complete(path)
            ]
            if not missing:
                print("Reusing Schiebinger classifiers produced by another job.")
                return

            for path in required_paths:
                if path.suffix != ".pt":
                    raise ValueError(
                        "Schiebinger classifier destinations must be .pt paths, got "
                        f"{path}."
                    )

            configured_python = str(
                getattr(cfg.problem, "schiebinger_classifier_python", "")
            )
            interpreter = python_executable or resolve_classifier_python(
                configured_python
            )
            command: List[str] = [
                interpreter,
                str(TRAINER),
                "--dataset-location",
                str(cfg.problem.dataset_location),
                "--n-pcs",
                str(int(getattr(cfg.problem, "n_pcs", 5))),
                "--seed",
                str(int(getattr(cfg.training, "seed", 0))),
                "--batch-size",
                str(
                    int(
                        getattr(
                            cfg.problem, "schiebinger_classifier_batch_size", 512
                        )
                    )
                ),
                "--max-epochs",
                str(
                    int(
                        getattr(
                            cfg.problem, "schiebinger_classifier_max_epochs", 100
                        )
                    )
                ),
                "--patience",
                str(
                    int(
                        getattr(cfg.problem, "schiebinger_classifier_patience", 20)
                    )
                ),
                "--device",
                str(getattr(cfg.problem, "schiebinger_classifier_device", "cpu")),
            ]
            if full_path in missing:
                command.extend(
                    ["--all-days", "--all-days-checkpoint", str(full_path)]
                )
            if training_required and training_path in missing:
                command.extend(
                    [
                        "--train-times",
                        ",".join(
                            str(value) for value in cfg.problem.retained_timepoints
                        ),
                        "--training-checkpoint",
                        str(training_path),
                    ]
                )
            if any(checkpoint_pair_partial(path) for path in missing):
                command.append("--overwrite")

            print(
                "Preparing missing Schiebinger classifier checkpoint(s) before flow "
                f"training: {', '.join(str(path) for path in missing)}",
                flush=True,
            )
            try:
                runner(command, check=True, cwd=str(REPO_ROOT))
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    "Schiebinger classifier training failed; flow training was "
                    "not started."
                ) from exc

            still_missing = [
                path for path in required_paths if not checkpoint_pair_complete(path)
            ]
            if still_missing:
                raise RuntimeError(
                    "Schiebinger classifier trainer returned without complete "
                    f".pt/.npz checkpoint pairs: {still_missing}."
                )
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
