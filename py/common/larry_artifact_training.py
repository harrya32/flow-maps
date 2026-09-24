"""Cache-aware PCA and classifier preparation for configurable LARRY runs."""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

from common import larry


REPO_ROOT = Path(__file__).resolve().parents[2]
PCA_BUILDER = REPO_ROOT / "scripts" / "create_larry_hvg_pca.py"
CLASSIFIER_TRAINER = REPO_ROOT / "scripts" / "train_larry_celltype_classifiers.py"


def _runtime_npz_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser().resolve()
    return path if path.suffix == ".npz" else path.with_suffix(".npz")


def checkpoint_ready(path_value: str | Path) -> bool:
    """Return whether the NumPy checkpoint used by flow code exists."""
    return _runtime_npz_path(path_value).is_file()


def checkpoint_partial(path_value: str | Path) -> bool:
    """Return whether a PyTorch checkpoint exists without its NumPy export."""
    path = Path(path_value).expanduser().resolve()
    pt_path = path.with_suffix(".pt") if path.suffix == ".npz" else path
    return pt_path.is_file() and not _runtime_npz_path(path).is_file()


def _python_works(executable: Path, modules: Sequence[str]) -> bool:
    if not executable.is_file():
        return False
    probe = (
        "import importlib; "
        f"[importlib.import_module(name) for name in {tuple(modules)!r}]"
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


def _candidate_interpreters() -> Iterable[Path]:
    current = Path(sys.executable).resolve()
    yield current

    env_roots = []
    nested_envs = current.parent.parent.parent
    if nested_envs.name == "envs":
        env_roots.append(nested_envs)
    base_envs = current.parent.parent / "envs"
    if base_envs.is_dir():
        env_roots.append(base_envs)
    for env_root in env_roots:
        for name in ("mfm_env", "torchcfm", "maizels2023aa", "flowmaps"):
            yield env_root / name / "bin" / "python"


def resolve_artifact_python(
    configured: str | None = None,
    *,
    needs_pca: bool,
    needs_classifiers: bool,
) -> str:
    """Find one interpreter containing the dependencies needed right now."""
    modules = ["anndata", "numpy", "pandas"]
    if needs_pca:
        modules.extend(("scanpy", "scipy"))
    if needs_classifiers:
        modules.extend(("matplotlib", "sklearn", "torch"))

    requested = configured or os.getenv("LARRY_ARTIFACT_PYTHON", "")
    if requested:
        path = Path(requested).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                "LARRY_ARTIFACT_PYTHON does not point to an existing Python "
                f"interpreter: {path}"
            )
        if not _python_works(path, modules):
            raise RuntimeError(
                "The configured LARRY artifact Python cannot import the required "
                f"packages ({', '.join(modules)}): {path}"
            )
        return str(path)

    seen = set()
    for candidate in _candidate_interpreters():
        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if _python_works(candidate, modules):
            return str(candidate)

    raise RuntimeError(
        "No Python interpreter with the LARRY preprocessing dependencies was "
        f"found ({', '.join(modules)}). Set LARRY_ARTIFACT_PYTHON to a Python "
        "containing those packages."
    )


def _validate_pca_dataset(cfg, dataset_path: Path) -> None:
    data = larry.load_dataset(
        dataset_path,
        representation=str(cfg.problem.larry_representation),
        n_pcs=int(cfg.problem.n_pcs),
    )
    missing = int((data["clone_ids"] < 0).sum())
    if bool(getattr(cfg.problem, "larry_clone_labelled_only", False)) and missing:
        raise ValueError(
            f"{dataset_path} contains {missing:,} cells without clone assignments. "
            "It cannot be used with --larry_clone_labelled_only; rebuild the "
            "separate cache with scripts/create_larry_hvg_pca.py "
            "--clone-labelled-only."
        )
    observed_days = set(data["timepoints"])
    missing_days = set(larry.TIMEPOINTS) - observed_days
    if missing_days:
        raise ValueError(
            f"{dataset_path} is missing LARRY PCA-fit day(s): "
            f"{sorted(missing_days)}."
        )


def _canonical_classifier_path(cfg, variant: str, output_dir: Path) -> Path:
    kwargs = {
        "classifier_dir": output_dir,
        "n_pcs": int(cfg.problem.n_pcs),
        "n_hvgs": int(cfg.problem.n_hvgs),
        "representation": str(cfg.problem.larry_representation),
        "clone_labelled_only": bool(cfg.problem.larry_clone_labelled_only),
    }
    if variant == "all_days":
        return larry.classifier_checkpoint_path(all_days=True, **kwargs)
    return larry.classifier_checkpoint_path(
        training_timepoints=("D2", "D6"),
        **kwargs,
    )


def _normalise_pt_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser().resolve()
    return path.with_suffix(".pt") if path.suffix == ".npz" else path


def _mark_artifacts_available(cfg) -> None:
    cfg.problem.training_classifier_available = checkpoint_ready(
        cfg.problem.training_classifier_path
    )
    cfg.problem.full_data_classifier_available = checkpoint_ready(
        cfg.problem.full_data_classifier_path
    )
    cfg.logging.maizels.violation_metrics_available = bool(
        cfg.problem.full_data_classifier_available
    )
    if cfg.problem.full_data_classifier_available:
        cfg.logging.maizels.violation_metrics_unavailable_reason = ""


def ensure_larry_artifacts(
    cfg,
    *,
    runner: Callable = subprocess.run,
    python_executable: str | None = None,
) -> None:
    """Create a requested PCA and both matching classifiers once, then reuse."""
    if not larry.is_larry_target(getattr(cfg.problem, "target", None)):
        return
    if not bool(getattr(cfg.problem, "larry_auto_prepare_artifacts", True)):
        return
    representation = larry.canonical_representation(
        getattr(cfg.problem, "larry_representation", larry.PCA50_REPRESENTATION)
    )
    if representation != larry.PCA50_REPRESENTATION:
        raise ValueError(
            "Automatic LARRY PCA/classifier preparation requires the PCA "
            "representation."
        )

    dataset_path = Path(str(cfg.problem.dataset_location)).expanduser().resolve()
    clone_labelled_only = bool(cfg.problem.larry_clone_labelled_only)
    subset_description = "clone-labelled " if clone_labelled_only else ""
    classifier_paths = {
        "train_days_d2_d6": _normalise_pt_path(
            cfg.problem.training_classifier_path
        ),
        "all_days": _normalise_pt_path(cfg.problem.full_data_classifier_path),
    }
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = dataset_path.parent / f".{dataset_path.stem}.artifacts.lock"

    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            needs_pca = not dataset_path.is_file()
            missing_variants = [
                variant
                for variant, path in classifier_paths.items()
                if not checkpoint_ready(path)
            ]
            if needs_pca or missing_variants:
                configured_python = str(
                    getattr(cfg.problem, "larry_artifact_python", "")
                )
                interpreter = python_executable or resolve_artifact_python(
                    configured_python,
                    needs_pca=needs_pca,
                    needs_classifiers=bool(missing_variants),
                )
            else:
                interpreter = ""

            if needs_pca:
                command: List[str] = [
                    interpreter,
                    str(PCA_BUILDER),
                    "--data-dir",
                    str(Path(str(cfg.problem.larry_raw_data_dir)).expanduser()),
                    "--output",
                    str(dataset_path),
                    "--n-hvgs",
                    str(int(cfg.problem.n_hvgs)),
                    "--n-pcs",
                    str(int(cfg.problem.n_pcs)),
                    "--seed",
                    str(int(getattr(cfg.problem, "larry_artifact_seed", 0))),
                ]
                if clone_labelled_only:
                    command.append("--clone-labelled-only")
                print(
                    f"Preparing the missing {subset_description}LARRY PCA cache "
                    "before "
                    f"flow training: {dataset_path}",
                    flush=True,
                )
                try:
                    runner(command, check=True, cwd=str(REPO_ROOT))
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError(
                        f"{subset_description.capitalize()}LARRY PCA preparation "
                        "failed; flow "
                        "training was not started."
                    ) from exc
                if not dataset_path.is_file():
                    raise RuntimeError(
                        "The LARRY PCA builder returned without creating "
                        f"{dataset_path}."
                    )

            _validate_pca_dataset(cfg, dataset_path)

            # Recheck after taking the lock in case another job prepared them.
            missing_variants = [
                variant
                for variant, path in classifier_paths.items()
                if not checkpoint_ready(path)
            ]
            grouped = defaultdict(list)
            for variant in missing_variants:
                path = classifier_paths[variant]
                if path.suffix != ".pt":
                    raise ValueError(
                        "LARRY classifier destinations must be .pt/.npz paths, "
                        f"got {path}."
                    )
                canonical = _canonical_classifier_path(cfg, variant, path.parent)
                if path != canonical:
                    raise ValueError(
                        "Automatic LARRY classifier training uses canonical names. "
                        f"Expected {canonical}, but the configured missing path is "
                        f"{path}. Train that custom checkpoint explicitly or point "
                        "the config at the canonical cache."
                    )
                grouped[path.parent].append(variant)

            for output_dir, variants in grouped.items():
                output_dir.mkdir(parents=True, exist_ok=True)
                command = [
                    interpreter,
                    str(CLASSIFIER_TRAINER),
                    "--dataset-location",
                    str(dataset_path),
                    "--representation",
                    representation,
                    "--output-dir",
                    str(output_dir),
                    "--n-hvgs",
                    str(int(cfg.problem.n_hvgs)),
                    "--n-pcs",
                    str(int(cfg.problem.n_pcs)),
                    "--seed",
                    str(int(getattr(cfg.problem, "larry_artifact_seed", 0))),
                    "--batch-size",
                    str(
                        int(
                            getattr(
                                cfg.problem, "larry_classifier_batch_size", 512
                            )
                        )
                    ),
                    "--max-epochs",
                    str(
                        int(
                            getattr(
                                cfg.problem, "larry_classifier_max_epochs", 100
                            )
                        )
                    ),
                    "--patience",
                    str(
                        int(getattr(cfg.problem, "larry_classifier_patience", 20))
                    ),
                    "--device",
                    str(getattr(cfg.problem, "larry_classifier_device", "cpu")),
                    "--variants",
                    *variants,
                ]
                if clone_labelled_only:
                    variants_index = command.index("--variants")
                    command.insert(variants_index, "--clone-labelled-only")
                if any(checkpoint_partial(classifier_paths[name]) for name in variants):
                    command.append("--overwrite")
                print(
                    f"Preparing missing {subset_description}LARRY classifier(s) "
                    "before "
                    f"flow training: {', '.join(variants)}",
                    flush=True,
                )
                try:
                    runner(command, check=True, cwd=str(REPO_ROOT))
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError(
                        f"{subset_description.capitalize()}LARRY classifier "
                        "training failed; flow "
                        "training was not started."
                    ) from exc

            still_missing = [
                path for path in classifier_paths.values() if not checkpoint_ready(path)
            ]
            if still_missing:
                raise RuntimeError(
                    "The LARRY classifier trainer returned without usable .npz "
                    f"checkpoint(s): {still_missing}."
                )
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    _mark_artifacts_available(cfg)
    print(
        f"Using cached {subset_description}LARRY PCA and D2/D6 + all-days "
        "classifier checkpoints.",
        flush=True,
    )
