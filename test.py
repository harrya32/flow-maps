import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# Keep caches in project-local writable paths (helps on restrictive environments).
_CACHE_ROOT = Path(".cache")
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT.resolve()))
os.environ.setdefault("MPLCONFIGDIR", str((_CACHE_ROOT / "matplotlib").resolve()))
os.environ.setdefault("NUMBA_CACHE_DIR", str((_CACHE_ROOT / "numba").resolve()))
for _p in [
    _CACHE_ROOT,
    Path(os.environ["MPLCONFIGDIR"]),
    Path(os.environ["NUMBA_CACHE_DIR"]),
]:
    _p.mkdir(parents=True, exist_ok=True)


def _ensure_py_path() -> None:
    repo_root = Path(__file__).resolve().parent
    py_dir = repo_root / "py"
    py_dir_str = str(py_dir)
    if py_dir_str not in sys.path:
        sys.path.append(py_dir_str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a flow-map model with LSD loss on Schiebinger serum data "
            "(PCA embedding), then evaluate held-out intermediate times."
        )
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets/reprogramming_schiebinger.h5ad",
        help=(
            "Base path for Schiebinger h5ad. With --subset_to_serum, this resolves "
            "to *_serum.h5ad."
        ),
    )
    parser.add_argument(
        "--subset_to_serum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the serum subset (default: enabled).",
    )
    parser.add_argument(
        "--embedding_key",
        type=str,
        default="X_pca",
        help="Embedding source: X_pca, X, or another key in adata.obsm.",
    )
    parser.add_argument(
        "--time_key",
        type=str,
        default="day",
        help="Time-column key in adata.obs.",
    )
    parser.add_argument("--n_pcs", type=int, default=5)
    parser.add_argument("--whiten_pca", action="store_true", default=False)
    parser.add_argument("--max_endpoint_train", type=int, default=30000)
    parser.add_argument("--train_steps", type=int, default=6000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--diag_fraction", type=float, default=0.75)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--decay_steps", type=int, default=35000)
    parser.add_argument("--schedule_type", type=str, default="sqrt")
    parser.add_argument("--clip", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--heldout_max_times", type=int, default=12)
    parser.add_argument("--points_per_time_for_plot", type=int, default=2000)
    parser.add_argument("--eval_ema", type=float, default=0.9999)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/schiebinger_flowmap",
    )
    parser.add_argument(
        "--fig_dir",
        type=str,
        default="figs",
    )
    return parser.parse_args()


def load_schiebinger_dataset(
    path: str = "datasets/reprogramming_schiebinger.h5ad",
    subset_to_serum: bool = True,
):
    try:
        import scanpy as sc
    except ImportError:
        sc = None
    try:
        import anndata as ad
    except ImportError:
        ad = None

    def target_path(base_path: str, serum_subset: bool) -> Path:
        p = Path(base_path)
        return p.with_name(f"{p.stem}_serum{p.suffix}") if serum_subset else p

    def is_hdf5_signature(p: Path) -> bool:
        if not p.exists() or p.stat().st_size < 8:
            return False
        with p.open("rb") as f:
            return f.read(8) == b"\x89HDF\r\n\x1a\n"

    def load_with_urls(dest: Path, urls: List[str]):
        if sc is None:
            raise RuntimeError("scanpy is unavailable for URL fallback downloads.")
        dest.parent.mkdir(parents=True, exist_ok=True)
        for url in urls:
            if dest.exists() and not is_hdf5_signature(dest):
                dest.unlink()
            try:
                adata = sc.read(dest, backup_url=url, sparse=True, cache=False)
                if not is_hdf5_signature(dest):
                    # Downloaded payload is not an h5ad file (often HTML/proxy response).
                    if dest.exists():
                        dest.unlink()
                    continue
                return adata
            except OSError as exc:
                if "file signature not found" in str(exc).lower() and dest.exists():
                    dest.unlink()
                    continue
                raise
        raise RuntimeError(
            f"Failed to download a valid .h5ad file to '{dest}'. "
            "Check network/proxy access to figshare."
        )

    serum_urls = [
        "https://figshare.com/ndownloader/files/35858033",
        "https://ndownloader.figshare.com/files/35858033",
    ]
    full_urls = [
        "https://figshare.com/ndownloader/files/28618734",
        "https://ndownloader.figshare.com/files/28618734",
    ]

    # Fast local path when scanpy is not installed.
    if sc is None:
        if ad is None:
            raise ImportError(
                "Either scanpy or anndata is required for dataset loading."
            )
        serum_path = target_path(path, True)
        full_path = target_path(path, False)
        if subset_to_serum and serum_path.exists():
            return ad.read_h5ad(serum_path)
        if full_path.exists():
            adata = ad.read_h5ad(full_path)
            if subset_to_serum:
                if "serum" not in adata.obs:
                    raise RuntimeError(
                        "Loaded full Schiebinger dataset, but missing 'serum' in adata.obs."
                    )
                return adata[adata.obs["serum"].astype(bool)].copy()
            return adata
        raise RuntimeError(
            "scanpy is not installed and local Schiebinger .h5ad file was not found."
        )

    if subset_to_serum:
        try:
            return load_with_urls(target_path(path, True), serum_urls)
        except RuntimeError:
            # Fallback: full dataset URL then subset locally if serum column exists.
            adata = load_with_urls(target_path(path, False), full_urls)
            if "serum" in adata.obs:
                return adata[adata.obs["serum"].astype(bool)].copy()
            raise RuntimeError(
                "Loaded full Schiebinger dataset, but missing 'serum' in adata.obs."
            )

    return load_with_urls(target_path(path, False), full_urls)


def get_embedding(
    adata,
    embedding_key: str = "X_pca",
    n_pcs: int = 5,
    pca_random_state: int = 0,
    whiten_pca: bool = False,
) -> np.ndarray:
    try:
        import scanpy as sc
    except ImportError:
        sc = None
    try:
        import scipy.sparse as sp
    except ImportError:
        sp = None

    def to_dense_float32(x):
        if sp is not None and sp.issparse(x):
            x = x.toarray()
        return np.asarray(x, dtype=np.float32)

    def run_pca_fallback(x: np.ndarray, n_comp: int, seed: int) -> np.ndarray:
        try:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=n_comp, random_state=seed)
            return pca.fit_transform(x).astype(np.float32)
        except ImportError:
            x_centered = x - x.mean(axis=0, keepdims=True)
            u, s, _ = np.linalg.svd(x_centered, full_matrices=False)
            return (u[:, :n_comp] * s[:n_comp]).astype(np.float32)

    if embedding_key == "X_pca":
        rep = adata.obsm.get("X_pca")
        if rep is None or rep.shape[1] < n_pcs:
            if sc is not None:
                sc.pp.pca(adata, n_comps=n_pcs, random_state=pca_random_state)
                rep = adata.obsm["X_pca"]
            else:
                x_raw = to_dense_float32(adata.X)
                rep = run_pca_fallback(x_raw, n_pcs, pca_random_state)
        if rep.shape[1] > n_pcs:
            rep = rep[:, :n_pcs]
    elif embedding_key == "X":
        rep = adata.X
    else:
        rep = adata.obsm.get(embedding_key)
        if rep is None:
            raise ValueError(f"embedding_key '{embedding_key}' not found in adata.obsm")

    rep = to_dense_float32(rep)
    if whiten_pca:
        rep = rep - rep.mean(axis=0, keepdims=True)
        std = rep.std(axis=0, ddof=0, keepdims=True)
        std = np.where(std > 0, std, 1.0)
        rep = rep / std
    return rep


def extract_numeric_times(adata, time_key: str = "day") -> np.ndarray:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for reading time labels") from exc

    if time_key not in adata.obs:
        raise KeyError(f"'{time_key}' not found in adata.obs")

    times = pd.to_numeric(adata.obs[time_key], errors="coerce").to_numpy(dtype=float)
    return times


def random_subset(
    x: np.ndarray,
    n: int,
    rng: np.random.Generator,
    replace_if_needed: bool = True,
) -> np.ndarray:
    if x.shape[0] == 0:
        raise ValueError("Cannot sample from an empty array.")
    replace = replace_if_needed and n > x.shape[0]
    idx = rng.choice(x.shape[0], size=n, replace=replace)
    return x[idx]


def choose_heldout_times(unique_times: np.ndarray, max_times: int = 12) -> np.ndarray:
    mid = unique_times[1:-1]
    if mid.size == 0:
        return np.array([], dtype=float)
    if max_times <= 0:
        return mid
    k = min(max_times, mid.size)
    idx = np.linspace(0, mid.size - 1, num=k, dtype=int)
    return np.unique(mid[idx])


def _sample_diag_offdiag_times(
    batch_size: int,
    diag_fraction: float,
    rng: np.random.Generator,
    tmin: float = 0.0,
    tmax: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    diag_bs = max(1, int(batch_size * diag_fraction))
    offdiag_bs = batch_size - diag_bs

    s_diag = rng.uniform(tmin, tmax, size=diag_bs).astype(np.float32)
    t_diag = s_diag.copy()

    if offdiag_bs <= 0:
        return s_diag, t_diag

    a = rng.uniform(tmin, tmax, size=offdiag_bs).astype(np.float32)
    b = rng.uniform(tmin, tmax, size=offdiag_bs).astype(np.float32)
    s_off = np.minimum(a, b)
    t_off = np.maximum(a, b)
    s_batch = np.concatenate([s_diag, s_off], axis=0)
    t_batch = np.concatenate([t_diag, t_off], axis=0)
    return s_batch.astype(np.float32), t_batch.astype(np.float32)


def build_flowmap_config(
    *,
    dim: int,
    seed: int,
    learning_rate: float,
    batch_size: int,
    train_steps: int,
    diag_fraction: float,
    decay_steps: int,
    schedule_type: str,
    clip: float,
    rescale: float,
):
    try:
        import jax
        import ml_collections
    except ImportError as exc:
        raise ImportError(
            "jax and ml_collections are required for flow-map training"
        ) from exc

    cfg = ml_collections.ConfigDict()

    cfg.training = ml_collections.ConfigDict()
    cfg.training.shuffle = True
    cfg.training.conditional = False
    cfg.training.class_dropout = 0.0
    cfg.training.stopgrad_type = "convex"
    cfg.training.psd_type = None
    cfg.training.loss_type = "lsd"
    cfg.training.tmin = 0.0
    cfg.training.tmax = 1.0
    cfg.training.seed = int(seed)
    cfg.training.ema_facs = [0.999, 0.9999]
    cfg.training.ndevices = jax.device_count()
    cfg.training.teacher_ema_factor = None

    cfg.problem = ml_collections.ConfigDict()
    cfg.problem.n = None
    cfg.problem.d = int(dim)
    cfg.problem.image_dims = None
    cfg.problem.num_classes = None
    cfg.problem.target = "schiebinger_pca"
    cfg.problem.dataset_location = None
    cfg.problem.interp_type = "linear"
    cfg.problem.base = "empirical_first_timepoint"
    cfg.problem.gaussian_scale = "adaptive"

    cfg.optimization = ml_collections.ConfigDict()
    cfg.optimization.bs = int(batch_size)
    cfg.optimization.diag_fraction = float(diag_fraction)
    cfg.optimization.learning_rate = float(learning_rate)
    cfg.optimization.clip = float(clip)
    cfg.optimization.total_steps = int(train_steps)
    cfg.optimization.total_samples = int(train_steps * batch_size)
    cfg.optimization.decay_steps = int(decay_steps)
    cfg.optimization.schedule_type = str(schedule_type)

    cfg.logging = ml_collections.ConfigDict()
    cfg.logging.plot_bs = 2000
    cfg.logging.visual_freq = 500
    cfg.logging.save_freq = 1000
    cfg.logging.wandb_project = "self-distill-flow-maps"
    cfg.logging.wandb_name = "schiebinger_flowmap_lsd_local"
    cfg.logging.wandb_entity = os.getenv("WANDB_ENTITY", "your-username")
    cfg.logging.output_folder = ""
    cfg.logging.output_name = "schiebinger_flowmap_lsd_local"
    cfg.logging.fid_freq = 0
    cfg.logging.fid_stats_path = None
    cfg.logging.fid_n_samples = None
    cfg.logging.fid_batch_size = None
    cfg.logging.fid_n_steps_flow = None
    cfg.logging.fid_ema_factor = None
    cfg.logging.visual_ema_factor = None

    # Keep low-dimensional architecture close to paper checker defaults.
    cfg.network = ml_collections.ConfigDict()
    cfg.network.network_type = "mlp"
    cfg.network.n_hidden = 4
    cfg.network.n_neurons = 512
    cfg.network.output_dim = int(dim)
    cfg.network.act = "gelu"
    cfg.network.use_residual = False
    cfg.network.use_weight = False
    cfg.network.use_bfloat16 = False
    cfg.network.rescale = float(rescale)
    cfg.network.load_path = ""
    cfg.network.input_dims = (int(dim),)
    cfg.network.load_ema_fac = None
    cfg.network.img_resolution = None
    cfg.network.img_channels = None
    cfg.network.label_dim = None
    cfg.network.logvar_channels = None
    cfg.network.reset_optimizer = True
    cfg.network.unet_kwargs = None

    return cfg


def _grad_norm(grads) -> float:
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree

    flat = ravel_pytree(grads)[0]
    return float(jnp.linalg.norm(flat))


def train_lsd_flowmap(
    x_start: np.ndarray,
    x_end: np.ndarray,
    *,
    train_steps: int,
    batch_size: int,
    learning_rate: float,
    diag_fraction: float,
    decay_steps: int,
    schedule_type: str,
    clip: float,
    seed: int,
    log_every: int,
    eval_ema: Optional[float],
):
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError("jax is required for flow-map training") from exc

    _ensure_py_path()
    import common.dist_utils as dist_utils
    import common.flow_map as flow_map
    import common.interpolant as interpolant
    import common.losses as losses
    import common.state_utils as state_utils
    import common.updates as updates

    dim = int(x_start.shape[1])
    data_rescale = float(np.std(np.concatenate([x_start, x_end], axis=0)))
    if not np.isfinite(data_rescale) or data_rescale <= 1e-6:
        data_rescale = 1.0

    cfg = build_flowmap_config(
        dim=dim,
        seed=seed,
        learning_rate=learning_rate,
        batch_size=batch_size,
        train_steps=train_steps,
        diag_fraction=diag_fraction,
        decay_steps=decay_steps,
        schedule_type=schedule_type,
        clip=clip,
        rescale=data_rescale,
    )

    prng_key = jax.random.PRNGKey(seed)
    ex_input = jnp.asarray(x_start[0], dtype=jnp.float32)

    train_state, net, _, _ = state_utils.setup_training_state(cfg, ex_input, prng_key)
    interp = interpolant.setup_interpolant(cfg)
    loss_fn = losses.setup_loss(cfg, net, interp)
    train_step = updates.setup_train_step(cfg)
    update_ema_params = updates.setup_ema_update(cfg)

    train_state = dist_utils.safe_replicate(cfg, train_state)

    rng = np.random.default_rng(seed)
    train_losses: List[float] = []
    grad_norms: List[float] = []

    key_seq = jax.random.PRNGKey(seed + 1)
    for step in range(1, train_steps + 1):
        x0_batch = random_subset(
            x_start, cfg.optimization.bs, rng, replace_if_needed=True
        ).astype(np.float32)
        x1_batch = random_subset(
            x_end, cfg.optimization.bs, rng, replace_if_needed=True
        ).astype(np.float32)
        s_batch, t_batch = _sample_diag_offdiag_times(
            cfg.optimization.bs,
            cfg.optimization.diag_fraction,
            rng,
            tmin=cfg.training.tmin,
            tmax=cfg.training.tmax,
        )

        # Unused by LSD path, but provided to match the generic loss signature.
        u_batch = s_batch.copy()
        h_batch = np.zeros((cfg.optimization.bs,), dtype=np.float32)

        key_seq, dropout_key = jax.random.split(key_seq)
        dropout_keys = jax.random.split(dropout_key, num=cfg.optimization.bs).reshape(
            (cfg.optimization.bs, -1)
        )

        loss_fn_args = (
            jnp.asarray(x0_batch, dtype=jnp.float32),
            jnp.asarray(x1_batch, dtype=jnp.float32),
            None,  # labels unused for this unconditional setup
            jnp.asarray(s_batch, dtype=jnp.float32),
            jnp.asarray(t_batch, dtype=jnp.float32),
            jnp.asarray(u_batch, dtype=jnp.float32),
            jnp.asarray(h_batch, dtype=jnp.float32),
            dropout_keys,
            jnp.ones((cfg.optimization.bs,), dtype=jnp.float32),  # constraint scale
            jnp.zeros((cfg.optimization.bs,), dtype=jnp.float32),  # stage2 scale
        )
        loss_fn_args = dist_utils.replicate_loss_fn_args(cfg, loss_fn_args)

        teacher_params = train_state.params
        train_state, loss_value, grads = train_step(
            train_state,
            loss_fn,
            (teacher_params, *loss_fn_args),
        )
        train_state = update_ema_params(train_state)

        loss_scalar = float(
            np.asarray(jax.device_get(dist_utils.safe_index(cfg, loss_value)))
        )
        grads_unrep = dist_utils.safe_unreplicate(cfg, grads)
        grad_norm = _grad_norm(grads_unrep)

        train_losses.append(loss_scalar)
        grad_norms.append(grad_norm)

        if step % log_every == 0 or step == 1 or step == train_steps:
            print(
                f"step {step:5d}/{train_steps}  "
                f"lsd_loss={loss_scalar:.6f}  grad_norm={grad_norm:.6f}"
            )

    # Pick EMA params for evaluation if requested and available.
    eval_params = None
    if eval_ema is not None and eval_ema in train_state.ema_params:
        eval_params = train_state.ema_params[eval_ema]
    if eval_params is None:
        eval_params = train_state.params

    eval_params = dist_utils.safe_unreplicate(cfg, eval_params)
    train_state_unrep = dist_utils.safe_unreplicate(cfg, train_state)

    return {
        "cfg": cfg,
        "net": net,
        "apply_fn": net.apply,
        "eval_params": eval_params,
        "train_state": train_state_unrep,
        "loss_history": np.asarray(train_losses, dtype=np.float32),
        "grad_norm_history": np.asarray(grad_norms, dtype=np.float32),
    }


def pushforward_flowmap(
    apply_fn,
    params,
    x0: np.ndarray,
    tau: float,
) -> np.ndarray:
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError("jax is required for flow-map sampling") from exc

    tau = float(tau)
    tau_jnp = jnp.asarray(tau, dtype=jnp.float32)
    x0_jnp = jnp.asarray(x0, dtype=jnp.float32)

    xt = jax.vmap(
        lambda x: apply_fn(
            params,
            0.0,
            tau_jnp,
            x,
            label=None,
            train=False,
            calc_weight=False,
            return_X_and_phi=False,
        )
    )(x0_jnp)
    return np.asarray(xt, dtype=np.float32)


def covariance_fro_error(x_pred: np.ndarray, x_true: np.ndarray) -> float:
    cov_pred = np.cov(x_pred, rowvar=False)
    cov_true = np.cov(x_true, rowvar=False)
    return float(np.linalg.norm(cov_pred - cov_true, ord="fro"))


def evaluate_heldout_times(
    *,
    embedding: np.ndarray,
    times: np.ndarray,
    x_start_all: np.ndarray,
    t_start: float,
    t_end: float,
    heldout_times: np.ndarray,
    points_per_time: int,
    apply_fn,
    params,
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    rng = np.random.default_rng(seed)
    results: List[Dict] = []
    metrics: List[Dict] = []

    for day in heldout_times:
        actual_all = embedding[times == day]
        if actual_all.shape[0] == 0:
            continue

        n_compare = min(points_per_time, actual_all.shape[0])
        actual = random_subset(
            actual_all,
            n_compare,
            rng,
            replace_if_needed=False,
        ).astype(np.float32)
        x0_for_gen = random_subset(
            x_start_all,
            n_compare,
            rng,
            replace_if_needed=True,
        ).astype(np.float32)

        tau = (float(day) - t_start) / (t_end - t_start)
        tau = float(np.clip(tau, 0.0, 1.0))

        pred = pushforward_flowmap(apply_fn, params, x0_for_gen, tau=tau)

        mean_mse = float(np.mean((pred.mean(axis=0) - actual.mean(axis=0)) ** 2))
        cov_err = covariance_fro_error(pred, actual)

        results.append(
            {
                "day": float(day),
                "tau": tau,
                "actual": actual,
                "pred": pred,
            }
        )
        metrics.append(
            {
                "day": float(day),
                "tau": tau,
                "n": int(n_compare),
                "mean_mse": mean_mse,
                "cov_fro_error": cov_err,
            }
        )
    return results, metrics


def plot_training_curve(
    loss_history: np.ndarray,
    grad_norm_history: np.ndarray,
    out_path: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting") from exc

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    steps = np.arange(1, len(loss_history) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(steps, loss_history, lw=1.5)
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("LSD Loss")
    axes[0].set_title("Flow-Map LSD Training Curve")
    axes[0].grid(alpha=0.25)

    axes[1].plot(steps, grad_norm_history, lw=1.5, color="tab:orange")
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("Gradient Norm")
    axes[1].set_title("Gradient Norm")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_heldout_comparison(results: List[Dict], out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting") from exc

    if not results:
        raise ValueError("No held-out results to plot.")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    all_xy = np.concatenate(
        [np.concatenate([r["actual"][:, :2], r["pred"][:, :2]], axis=0) for r in results],
        axis=0,
    )
    x_min, y_min = np.percentile(all_xy, 1, axis=0)
    x_max, y_max = np.percentile(all_xy, 99, axis=0)

    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 4 * n_rows), squeeze=False)

    for row, r in enumerate(results):
        day = r["day"]
        tau = r["tau"]
        actual = r["actual"]
        pred = r["pred"]

        ax_a = axes[row, 0]
        ax_p = axes[row, 1]

        ax_a.scatter(actual[:, 0], actual[:, 1], s=4, alpha=0.5, c="#1f77b4", linewidths=0)
        ax_p.scatter(pred[:, 0], pred[:, 1], s=4, alpha=0.5, c="#ff7f0e", linewidths=0)

        ax_a.set_title(f"Actual samples at day={day:g}")
        ax_p.set_title(f"Flow-map estimate at day={day:g} (tau={tau:.3f})")

        for ax in [ax_a, ax_p]:
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def write_metrics_csv(metrics: List[Dict], out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["day", "tau", "n", "mean_mse", "cov_fro_error"]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)


def save_training_artifacts(
    train_output: Dict,
    output_dir: str,
) -> Dict[str, str]:
    try:
        from flax.serialization import to_bytes
    except ImportError as exc:
        raise ImportError("flax is required for checkpoint serialization") from exc

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / "schiebinger_flowmap_lsd_state.pkl"
    with ckpt_path.open("wb") as f:
        f.write(to_bytes(train_output["train_state"]))

    loss_npy = out_dir / "schiebinger_flowmap_lsd_loss.npy"
    grad_npy = out_dir / "schiebinger_flowmap_lsd_grad_norm.npy"
    np.save(loss_npy, train_output["loss_history"])
    np.save(grad_npy, train_output["grad_norm_history"])

    return {
        "checkpoint": str(ckpt_path),
        "loss_npy": str(loss_npy),
        "grad_npy": str(grad_npy),
    }


def main():
    args = parse_args()

    print("Loading Schiebinger dataset...")
    adata = load_schiebinger_dataset(
        path=args.dataset_path,
        subset_to_serum=args.subset_to_serum,
    )

    print(
        f"Building {args.n_pcs}D embedding from '{args.embedding_key}' "
        f"(whiten_pca={args.whiten_pca})..."
    )
    embedding = get_embedding(
        adata,
        embedding_key=args.embedding_key,
        n_pcs=args.n_pcs,
        whiten_pca=args.whiten_pca,
        pca_random_state=args.seed,
    )
    times = extract_numeric_times(adata, time_key=args.time_key)

    valid = np.isfinite(times)
    embedding = embedding[valid]
    times = times[valid]

    unique_times = np.sort(np.unique(times))
    if unique_times.size < 3:
        raise RuntimeError("Need at least 3 unique time points (first, held-out, last).")

    t_start = float(unique_times[0])
    t_end = float(unique_times[-1])
    if not t_end > t_start:
        raise RuntimeError("Invalid time range: max time must be greater than min time.")

    x_start_all = embedding[times == t_start]
    x_end_all = embedding[times == t_end]
    if x_start_all.shape[0] == 0 or x_end_all.shape[0] == 0:
        raise RuntimeError("Could not find samples for first or last time point.")

    rng = np.random.default_rng(args.seed)
    n_start_train = min(args.max_endpoint_train, x_start_all.shape[0])
    n_end_train = min(args.max_endpoint_train, x_end_all.shape[0])
    x_start_train = random_subset(
        x_start_all,
        n_start_train,
        rng,
        replace_if_needed=False,
    ).astype(np.float32)
    x_end_train = random_subset(
        x_end_all,
        n_end_train,
        rng,
        replace_if_needed=False,
    ).astype(np.float32)

    print(
        f"First time={t_start:g}: {x_start_all.shape[0]} cells | "
        f"Last time={t_end:g}: {x_end_all.shape[0]} cells"
    )
    print(
        f"Training endpoints: start={x_start_train.shape[0]} "
        f"end={x_end_train.shape[0]}  dim={x_start_train.shape[1]}"
    )
    print("Training flow-map model with LSD loss (constraints disabled)...")
    print(
        "Settings: "
        f"steps={args.train_steps}, batch_size={args.batch_size}, "
        f"diag_fraction={args.diag_fraction}, lr={args.learning_rate}, "
        f"schedule={args.schedule_type}, decay_steps={args.decay_steps}, clip={args.clip}"
    )

    train_output = train_lsd_flowmap(
        x_start_train,
        x_end_train,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        diag_fraction=args.diag_fraction,
        decay_steps=args.decay_steps,
        schedule_type=args.schedule_type,
        clip=args.clip,
        seed=args.seed,
        log_every=args.log_every,
        eval_ema=args.eval_ema,
    )

    heldout_times = choose_heldout_times(
        unique_times,
        max_times=args.heldout_max_times,
    )
    if heldout_times.size == 0:
        raise RuntimeError("No held-out times found between first and last time points.")

    print("Evaluating held-out intermediate times...")
    results, metrics = evaluate_heldout_times(
        embedding=embedding,
        times=times,
        x_start_all=x_start_all,
        t_start=t_start,
        t_end=t_end,
        heldout_times=heldout_times,
        points_per_time=args.points_per_time_for_plot,
        apply_fn=train_output["apply_fn"],
        params=train_output["eval_params"],
        seed=args.seed + 123,
    )
    if not results:
        raise RuntimeError("No held-out results were generated.")

    for row in metrics:
        print(
            f"  day={row['day']:>6.2f}  tau={row['tau']:.3f}  n={row['n']:>4d}  "
            f"mean_mse={row['mean_mse']:.6f}  cov_fro={row['cov_fro_error']:.6f}"
        )

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    loss_plot = fig_dir / "flowmap_lsd_schiebinger_loss.png"
    compare_plot = fig_dir / "flowmap_lsd_schiebinger_heldout_vs_estimated.png"

    plot_training_curve(
        train_output["loss_history"],
        train_output["grad_norm_history"],
        str(loss_plot),
    )
    plot_heldout_comparison(results, str(compare_plot))

    output_paths = save_training_artifacts(train_output, args.output_dir)
    metrics_csv = Path(args.output_dir) / "flowmap_lsd_schiebinger_heldout_metrics.csv"
    write_metrics_csv(metrics, str(metrics_csv))

    print("Done.")
    print(f"checkpoint: {output_paths['checkpoint']}")
    print(f"loss history: {output_paths['loss_npy']}")
    print(f"grad norm history: {output_paths['grad_npy']}")
    print(f"heldout metrics: {metrics_csv}")
    print(f"loss plot: {loss_plot}")
    print(f"heldout comparison plot: {compare_plot}")


if __name__ == "__main__":
    main()
