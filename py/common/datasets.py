"""
Nicholas M. Boffi
10/5/25

Code for initializing common datasets.
"""

import functools
from pathlib import Path
from typing import Callable, Dict

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from ml_collections import config_dict

_SCHIEBINGER_SERUM_URLS = [
    "https://figshare.com/ndownloader/files/35858033",
    "https://ndownloader.figshare.com/files/35858033",
]
_SCHIEBINGER_FULL_URLS = [
    "https://figshare.com/ndownloader/files/28618734",
    "https://ndownloader.figshare.com/files/28618734",
]
_SCHIEBINGER_CACHE = {}


def unnormalize_image(image: jnp.ndarray):
    """Unnormalize an image from [-1, 1] to [0, 1] by scaling and clipping."""
    image = (image + 1) / 2
    image = jnp.clip(image, 0.0, 1.0)
    return image


def normalize_image_tf(image: tf.Tensor):
    """Normalize an image to have pixel values in the range [-1, 1]."""
    return (2 * (image / 255)) - 1


def preprocess_celeb_a(image: tf.Tensor) -> tf.Tensor:
    """Crop an image to 140x140, then resize to 64x64 pixels."""
    image = normalize_image_tf(image)
    crop = tf.image.resize_with_crop_or_pad(image, 140, 140)
    crop64 = tf.image.resize(crop, [64, 64], method="area", antialias=True)
    return crop64


def preprocess_image(cfg, x: Dict) -> Dict:
    """Preprocess the image for TensorFlow datasets."""
    image = x["image"]

    if cfg.problem.target == "celeb_a":
        # celeb_a doesn't have labels; artificially pad them all to 1
        label = 1.0
    else:
        label = x["label"]

    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)

    if cfg.problem.target == "cifar10" or "afhq" in cfg.problem.target:
        image = normalize_image_tf(image)
    elif cfg.problem.target == "celeb_a":
        image = preprocess_celeb_a(image)
    else:
        raise ValueError("Unknown dataset type.")

    # ensure (N, C, H, W)
    image = tf.transpose(image, [2, 0, 1])

    return {"image": image, "label": label}


def get_image_dataset(cfg: config_dict.ConfigDict):
    """Assemble a TensorFlow dataset for the specified problem target."""
    small_image_datasets = ["cifar10", "celeb_a"]
    is_small_image_dataset = cfg.problem.target in small_image_datasets
    is_afhq = "afhq" in cfg.problem.target

    if is_small_image_dataset:
        if cfg.problem.target == "cifar10":
            ds = tfds.load(
                "cifar10",
                split="train",
                shuffle_files=True,
                data_dir=cfg.problem.dataset_location,
            )
        elif cfg.problem.target == "celeb_a":
            ds = tfds.load(
                "celeb_a",
                split="train",
                shuffle_files=True,
                data_dir=cfg.problem.dataset_location,
            )
    elif is_afhq:
        load_str = f"{cfg.problem.dataset_location}/{cfg.problem.target}"
        ds = tf.data.experimental.load(load_str)

    ds = (
        ds.shuffle(10_000, reshuffle_each_iteration=True)
        .map(
            lambda x: preprocess_image(cfg, x),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .repeat()
        .batch(cfg.optimization.bs)
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator()
    )

    return ds


def sample_checkerboard(
    n_samples: int, key: jnp.ndarray, *, n_squares: int
) -> np.ndarray:
    """
    Samples the checkerboard dataset on [-1,1] x [-1,1]
    with alternating squares removed.
    """
    del key
    total_samples = 0
    samples = np.array([]).reshape((0, 2))

    while total_samples < n_samples:
        # Generate uniform samples on unit square
        curr_samples = np.random.rand(
            n_samples * 2, 2
        )  # Generate extra to account for filtering

        # Determine which square each point falls into
        x_idx = (curr_samples[:, 0] * n_squares).astype(int)
        y_idx = (curr_samples[:, 1] * n_squares).astype(int)

        # Keep points that fall in "white squares" of checkerboard
        mask = (x_idx + y_idx) % 2 == 0
        curr_samples = curr_samples[mask]

        # Take only what we need
        samples = np.concatenate((samples, curr_samples))
        total_samples = samples.shape[0]

    return 2 * samples[:n_samples] - 1


def sample_facing_moons(
    n_samples: int,
    key: jnp.ndarray,
    *,
    side: str,
    noise_std: float,
    gap: float,
) -> jnp.ndarray:
    """Sample one moon side from sklearn's make_moons dataset.

    Falls back to a local NumPy implementation if sklearn is unavailable.
    """

    if side not in ("left", "right"):
        raise ValueError(f"Unknown moon side: {side}")

    # Derive a reproducible sklearn seed from the JAX key.
    seed = int(jax.random.randint(key, shape=(), minval=0, maxval=2**31 - 1))

    try:
        from sklearn.datasets import make_moons

        x_all, y_all = make_moons(
            n_samples=(n_samples, n_samples),
            noise=float(noise_std),
            random_state=seed,
        )
    except ModuleNotFoundError:
        # Local fallback approximating sklearn.datasets.make_moons.
        rng = np.random.RandomState(seed)
        t0 = rng.rand(n_samples) * np.pi
        t1 = rng.rand(n_samples) * np.pi

        outer = np.stack([np.cos(t0), np.sin(t0)], axis=1)
        inner = np.stack([1.0 - np.cos(t1), -np.sin(t1) + 0.5], axis=1)

        x_all = np.concatenate([outer, inner], axis=0)
        y_all = np.concatenate(
            [
                np.zeros(n_samples, dtype=np.int32),
                np.ones(n_samples, dtype=np.int32),
            ]
        )

        if noise_std > 0:
            x_all = x_all + float(noise_std) * rng.randn(*x_all.shape)

        perm = rng.permutation(x_all.shape[0])
        x_all = x_all[perm]
        y_all = y_all[perm]

    mean_x_by_class = [x_all[y_all == kk, 0].mean() for kk in (0, 1)]
    left_label = int(np.argmin(mean_x_by_class))
    right_label = 1 - left_label
    label = left_label if side == "left" else right_label

    samples = x_all[y_all == label].astype(np.float32)

    # Optional horizontal separation for easier visualization/training.
    if gap != 0:
        shift = 0.5 * float(gap)
        samples[:, 0] = samples[:, 0] + (-shift if side == "left" else shift)

    return jnp.asarray(samples)


def _resolve_schiebinger_base_path(cfg: config_dict.ConfigDict) -> Path:
    dataset_location = getattr(cfg.problem, "dataset_location", "")
    default_filename = getattr(
        cfg.problem, "schiebinger_filename", "reprogramming_schiebinger.h5ad"
    )

    if dataset_location in ("", None):
        return Path("datasets") / default_filename

    location_path = Path(str(dataset_location))
    if location_path.suffix.lower() == ".h5ad":
        return location_path

    return location_path / default_filename


def _schiebinger_subset_path(base_path: Path, subset_to_serum: bool) -> Path:
    if not subset_to_serum:
        return base_path
    if base_path.stem.endswith("_serum"):
        return base_path
    return base_path.with_name(f"{base_path.stem}_serum{base_path.suffix}")


def _is_hdf5_signature(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 8:
        return False
    with path.open("rb") as f:
        return f.read(8) == b"\x89HDF\r\n\x1a\n"


def _load_schiebinger_with_urls(dest: Path, urls):
    """Load via scanpy backup URLs and guard against HTML/proxy payloads."""
    import scanpy as sc

    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        if dest.exists() and not _is_hdf5_signature(dest):
            dest.unlink()
        try:
            adata = sc.read(dest, backup_url=url, sparse=True, cache=False)
            if not _is_hdf5_signature(dest):
                if dest.exists():
                    dest.unlink()
                continue
            return adata
        except OSError as exc:
            if "file signature not found" in str(exc).lower() and dest.exists():
                dest.unlink()
                continue
            raise

    raise RuntimeError(f"Failed to fetch a valid Schiebinger .h5ad file at {dest}.")


def _load_schiebinger_anndata(cfg: config_dict.ConfigDict):
    """Load Schiebinger h5ad with local-first behavior and optional URL fallback."""
    subset_to_serum = bool(getattr(cfg.problem, "subset_to_serum", True))
    base_path = _resolve_schiebinger_base_path(cfg)
    subset_path = _schiebinger_subset_path(base_path, subset_to_serum)
    cache_key = (str(subset_path.resolve()), subset_to_serum)
    if cache_key in _SCHIEBINGER_CACHE:
        return _SCHIEBINGER_CACHE[cache_key]

    try:
        import scanpy as sc
    except ModuleNotFoundError:
        sc = None
    try:
        import anndata as ad
    except ModuleNotFoundError:
        ad = None

    if sc is None and ad is None:
        raise ImportError("Either scanpy or anndata is required for Schiebinger loading.")

    # Prefer explicit serum file if requested and available.
    if subset_to_serum and subset_path.exists():
        adata = sc.read(subset_path) if sc is not None else ad.read_h5ad(subset_path)
        _SCHIEBINGER_CACHE[cache_key] = adata
        return adata

    # Fall back to full local file and subset in-memory if needed.
    if base_path.exists():
        adata = sc.read(base_path) if sc is not None else ad.read_h5ad(base_path)
        if subset_to_serum:
            if "serum" not in adata.obs:
                raise RuntimeError(
                    "Loaded full Schiebinger dataset but missing `serum` in adata.obs."
                )
            adata = adata[adata.obs["serum"].astype(bool)].copy()
        _SCHIEBINGER_CACHE[cache_key] = adata
        return adata

    # Optional URL fallback requires scanpy backup_url support.
    if sc is None:
        raise RuntimeError(
            "Could not find local Schiebinger .h5ad and scanpy is unavailable for URL fallback."
        )

    if subset_to_serum:
        try:
            adata = _load_schiebinger_with_urls(subset_path, _SCHIEBINGER_SERUM_URLS)
        except RuntimeError:
            adata = _load_schiebinger_with_urls(base_path, _SCHIEBINGER_FULL_URLS)
            if "serum" not in adata.obs:
                raise RuntimeError(
                    "Loaded full Schiebinger dataset but missing `serum` in adata.obs."
                )
            adata = adata[adata.obs["serum"].astype(bool)].copy()
    else:
        adata = _load_schiebinger_with_urls(base_path, _SCHIEBINGER_FULL_URLS)

    _SCHIEBINGER_CACHE[cache_key] = adata
    return adata


def _to_dense_float32(x):
    try:
        import scipy.sparse as sp

        if sp.issparse(x):
            x = x.toarray()
    except ModuleNotFoundError:
        pass
    return np.asarray(x, dtype=np.float32)


def _run_pca_fallback(x: np.ndarray, n_comp: int, seed: int) -> np.ndarray:
    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_comp, random_state=seed)
        return pca.fit_transform(x).astype(np.float32)
    except ModuleNotFoundError:
        x_centered = x - x.mean(axis=0, keepdims=True)
        u, s, _ = np.linalg.svd(x_centered, full_matrices=False)
        return (u[:, :n_comp] * s[:n_comp]).astype(np.float32)


def load_schiebinger_embedding(
    cfg: config_dict.ConfigDict,
):
    """Load Schiebinger embedding/times according to config fields."""
    adata = _load_schiebinger_anndata(cfg)
    embedding_key = getattr(cfg.problem, "embedding_key", "X_pca")
    n_pcs = int(getattr(cfg.problem, "n_pcs", 5))
    pca_random_state = int(
        getattr(cfg.problem, "pca_random_state", getattr(cfg.training, "seed", 0))
    )
    whiten_pca = bool(getattr(cfg.problem, "whiten_pca", False))
    time_key = getattr(cfg.problem, "time_key", "day")

    if embedding_key == "X_pca":
        rep = adata.obsm.get("X_pca")
        if rep is None or rep.shape[1] < n_pcs:
            try:
                import scanpy as sc
            except ModuleNotFoundError:
                sc = None
            if sc is not None:
                sc.pp.pca(adata, n_comps=n_pcs, random_state=pca_random_state)
                rep = adata.obsm["X_pca"]
            else:
                rep = _run_pca_fallback(_to_dense_float32(adata.X), n_pcs, pca_random_state)
        if rep.shape[1] > n_pcs:
            rep = rep[:, :n_pcs]
    elif embedding_key == "X":
        rep = adata.X
    else:
        rep = adata.obsm.get(embedding_key)
        if rep is None:
            raise ValueError(f"Embedding key '{embedding_key}' not found in adata.obsm.")

    rep = _to_dense_float32(rep)
    if whiten_pca:
        rep = rep - rep.mean(axis=0, keepdims=True)
        std = rep.std(axis=0, ddof=0, keepdims=True)
        std = np.where(std > 0, std, 1.0)
        rep = rep / std

    if time_key not in adata.obs:
        raise KeyError(f"Time key '{time_key}' not found in adata.obs.")

    try:
        import pandas as pd

        times = pd.to_numeric(adata.obs[time_key], errors="coerce").to_numpy(dtype=float)
    except ModuleNotFoundError:
        raw = np.asarray(adata.obs[time_key])
        times = np.full(raw.shape[0], np.nan, dtype=float)
        for ii, value in enumerate(raw):
            try:
                times[ii] = float(value)
            except (TypeError, ValueError):
                continue

    valid = np.isfinite(times)
    rep = rep[valid]
    times = np.asarray(times[valid], dtype=np.float32)
    if rep.shape[0] == 0:
        raise RuntimeError("No finite Schiebinger time points remained after filtering.")

    return rep, times


def load_schiebinger_splits(
    cfg: config_dict.ConfigDict,
    *,
    subsample_endpoints: bool = True,
):
    """Load Schiebinger endpoints and optional endpoint-train subsamples."""
    embedding, times = load_schiebinger_embedding(cfg)
    unique_times = np.sort(np.unique(times))
    if unique_times.size < 2:
        raise RuntimeError("Schiebinger data needs at least 2 unique time points.")

    t_start = float(unique_times[0])
    t_end = float(unique_times[-1])
    x0_all = embedding[times == t_start]
    x1_all = embedding[times == t_end]
    if x0_all.shape[0] == 0 or x1_all.shape[0] == 0:
        raise RuntimeError("Could not find Schiebinger samples at first/last time points.")

    max_endpoint_train = int(getattr(cfg.problem, "max_endpoint_train", 0))
    if not subsample_endpoints:
        max_endpoint_train = 0
    seed = int(getattr(cfg.training, "seed", 0))

    def maybe_subsample(x: np.ndarray, max_n: int, seed_offset: int) -> np.ndarray:
        if max_n <= 0 or x.shape[0] <= max_n:
            return x
        rng = np.random.default_rng(seed + seed_offset)
        idx = rng.choice(x.shape[0], size=max_n, replace=False)
        return x[idx]

    x0_train = maybe_subsample(x0_all, max_endpoint_train, seed_offset=17).astype(np.float32)
    x1_train = maybe_subsample(x1_all, max_endpoint_train, seed_offset=29).astype(np.float32)

    return {
        "embedding": embedding,
        "times": times,
        "unique_times": unique_times,
        "t_start": t_start,
        "t_end": t_end,
        "x0_all": x0_all.astype(np.float32),
        "x1_all": x1_all.astype(np.float32),
        "x0_train": x0_train,
        "x1_train": x1_train,
    }


def setup_base(cfg: config_dict.ConfigDict, ex_input: jnp.ndarray) -> Callable:
    """Set up the base density for the system."""
    if cfg.problem.base == "gaussian":

        @functools.partial(jax.jit, static_argnums=(0,))
        def sample_rho0(bs: int, key: jnp.ndarray):
            return cfg.network.rescale * jax.random.normal(
                key, shape=(bs, *ex_input.shape)
            )
    elif cfg.problem.base == "left_moon":
        # Precompute a source pool outside JIT, then draw mini-batches via JAX indexing.
        # This keeps sklearn-compatible moon geometry while remaining tracer-safe.
        pool_size = max(
            int(getattr(cfg.problem, "base_pool_size", 20_000)),
            int(cfg.optimization.bs),
        )
        pool_key = jax.random.PRNGKey(cfg.training.seed + 17)
        source_pool = sample_facing_moons(
            pool_size,
            pool_key,
            side="left",
            noise_std=cfg.problem.moons_noise,
            gap=cfg.problem.moons_gap,
        )
        source_pool = jnp.asarray(source_pool, dtype=jnp.float32)

        @functools.partial(jax.jit, static_argnums=(0,))
        def sample_rho0(bs: int, key: jnp.ndarray):
            idx = jax.random.randint(
                key,
                shape=(bs,),
                minval=0,
                maxval=source_pool.shape[0],
            )
            return source_pool[idx]

    elif cfg.problem.base == "schiebinger_first_timepoint":
        split_data = load_schiebinger_splits(cfg, subsample_endpoints=True)
        source_pool = jnp.asarray(split_data["x0_train"], dtype=jnp.float32)
        if source_pool.shape[0] == 0:
            raise RuntimeError("Schiebinger source pool is empty.")

        @functools.partial(jax.jit, static_argnums=(0,))
        def sample_rho0(bs: int, key: jnp.ndarray):
            idx = jax.random.randint(
                key,
                shape=(bs,),
                minval=0,
                maxval=source_pool.shape[0],
            )
            return source_pool[idx]

    else:
        raise ValueError("Specified base density is not implemented.")

    return sample_rho0


def np_to_tfds(cfg: config_dict.ConfigDict, x1s: np.ndarray) -> tf.data.Dataset:
    """Given a NumPy array, convert to a TensorFlow dataset with batching and shuffling."""
    return (
        tf.data.Dataset.from_tensor_slices(x1s)
        .shuffle(50_000, reshuffle_each_iteration=True)
        .repeat()
        .batch(cfg.optimization.bs)
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator()
    )


def setup_target(cfg: config_dict.ConfigDict, prng_key: jnp.ndarray):
    """Set up the target density for the system."""
    if cfg.problem.target == "checker":
        assert cfg.problem.d == 2, "Checkerboard only implemented for d=2."

        @functools.partial(jax.jit, static_argnums=(0,))
        def sample_rho1(num_samples: int, key: jnp.ndarray) -> jnp.ndarray:
            return sample_checkerboard(num_samples, key, n_squares=4)

        n_samples = cfg.problem.n
        key, prng_key = jax.random.split(prng_key)
        x1s = sample_rho1(n_samples, key)
        rescale_value = float(np.std(x1s))
        ds = np_to_tfds(cfg, x1s)
    elif cfg.problem.target == "twomoons":
        assert cfg.problem.d == 2, "Two-moons target only implemented for d=2."

        n_samples = cfg.problem.n
        key, prng_key = jax.random.split(prng_key)
        x1s = sample_facing_moons(
            n_samples,
            key,
            side="right",
            noise_std=cfg.problem.moons_noise,
            gap=cfg.problem.moons_gap,
        )
        x1s = np.asarray(x1s)
        rescale_value = float(np.std(x1s))
        ds = np_to_tfds(cfg, x1s)
    elif cfg.problem.target == "schiebinger":
        split_data = load_schiebinger_splits(cfg, subsample_endpoints=True)
        x0s = split_data["x0_train"]
        x1s = split_data["x1_train"]
        cfg.problem.n = int(x1s.shape[0])
        cfg.problem.t_start = float(split_data["t_start"])
        cfg.problem.t_end = float(split_data["t_end"])
        rescale_value = float(np.std(np.concatenate([x0s, x1s], axis=0)))
        ds = np_to_tfds(cfg, x1s)
        print(
            "Loaded Schiebinger endpoints: "
            f"t_start={cfg.problem.t_start:g} (n={x0s.shape[0]}), "
            f"t_end={cfg.problem.t_end:g} (n={x1s.shape[0]}), "
            f"dim={x1s.shape[1]}"
        )

    elif (
        cfg.problem.target == "cifar10"
        or cfg.problem.target == "celeb_a"
        or "afhq" in cfg.problem.target
    ):
        ds = get_image_dataset(cfg)
        print("Loaded image dataset.")

    else:
        raise ValueError("Specified target density is not implemented.")

    # compute standard deviation of the dataset
    if cfg.problem.gaussian_scale == "adaptive":
        # hard code
        if (
            cfg.problem.target == "cifar10"
            or cfg.problem.target == "celeb_a"
            or "afhq" in cfg.problem.target
        ):
            rescale_value = 0.5

        # for generated datasets, it's computed above
        cfg.network.rescale = rescale_value
    else:
        cfg.network.rescale = 1.0

    return cfg, ds, prng_key
