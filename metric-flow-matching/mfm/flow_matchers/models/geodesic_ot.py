"""Heat-kernel Geodesic Sinkhorn coupling for the SF2M cell-data baseline."""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path

import numpy as np
import ot as pot
import torch
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import ArpackNoConvergence, eigsh
from sklearn.neighbors import NearestNeighbors


def _row_key(row: np.ndarray) -> bytes:
    return np.ascontiguousarray(row, dtype=np.float32).tobytes()


def _unique_rows(values: np.ndarray) -> np.ndarray:
    """Deduplicate rows without changing their first-occurrence order."""
    keep = []
    seen = set()
    for index, row in enumerate(values):
        key = _row_key(row)
        if key not in seen:
            seen.add(key)
            keep.append(index)
    return values[np.asarray(keep, dtype=np.int64)]


def _subsample_rows(values: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or values.shape[0] <= max_points:
        return values
    rng = np.random.default_rng(seed)
    indices = np.sort(
        rng.choice(values.shape[0], size=int(max_points), replace=False)
    )
    return values[indices]


class HeatKernelGeodesicCost:
    r"""Approximate the paper's ``c_geo^2 = -log(H_t)`` ground cost.

    The reference cells define an adaptive-weighted k-nearest-neighbour graph.
    A density-corrected symmetric graph Laplacian approximates the
    Laplace--Beltrami operator. Its lowest eigenpairs provide a reusable
    low-rank approximation to ``H_t = exp(-t L)``.
    """

    def __init__(
        self,
        populations,
        *,
        n_neighbors: int,
        heat_time: float,
        n_eigenvectors: int,
        max_points: int,
        seed: int,
        cache_dir: str | Path | None,
        heat_epsilon: float,
    ):
        arrays = []
        for population in populations:
            array = np.asarray(population, dtype=np.float32)
            if array.ndim != 2:
                raise ValueError("SF2M geodesic reference populations must be 2D.")
            if array.shape[0] > 0:
                arrays.append(array)
        if len(arrays) < 2:
            raise ValueError(
                "Geodesic SF2M needs at least two non-empty retained populations."
            )
        dimensions = {array.shape[1] for array in arrays}
        if len(dimensions) != 1:
            raise ValueError(
                "SF2M geodesic reference populations must share one dimension."
            )

        all_reference = np.concatenate(arrays, axis=0)
        reference = _subsample_rows(all_reference, int(max_points), int(seed))
        self.reference = _unique_rows(reference)
        if self.reference.shape[0] < 3:
            raise ValueError("SF2M geodesic graph needs at least three unique cells.")

        self.n_neighbors = min(int(n_neighbors), self.reference.shape[0] - 1)
        if self.n_neighbors < 2:
            raise ValueError("sf2m_geodesic_knn must be at least 2.")
        self.heat_time = float(heat_time)
        if self.heat_time <= 0.0:
            raise ValueError("sf2m_geodesic_heat_time must be positive.")
        self.n_eigenvectors = min(
            int(n_eigenvectors), self.reference.shape[0] - 1
        )
        if self.n_eigenvectors < 2:
            raise ValueError("sf2m_geodesic_eigenvectors must be at least 2.")
        self.heat_epsilon = float(heat_epsilon)
        self.seed = int(seed)
        if not 0.0 < self.heat_epsilon < 1.0:
            raise ValueError("sf2m_geodesic_heat_epsilon must lie in (0, 1).")

        self._row_to_node = {}
        self._nearest_model = None
        self._register_reference_rows()
        self._register_known_rows(all_reference)

        cache_path = self._cache_path(cache_dir)
        cached = self._load_cache(cache_path)
        if cached is None:
            eigenvalues, eigenvectors, component_count = self._build_spectrum()
            self._save_cache(
                cache_path,
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                component_count=component_count,
            )
            cache_hit = False
        else:
            eigenvalues, eigenvectors, component_count = cached
            cache_hit = True

        heat_weights = np.exp(-self.heat_time * eigenvalues)
        self.heat_features = (
            eigenvectors * np.sqrt(heat_weights)[None, :]
        ).astype(np.float64, copy=False)
        self.summary = {
            "definition": "c_geo^2=-log(H_t)",
            "graph": "adaptive_rbf_knn",
            "laplacian": "density_corrected_symmetric_normalized",
            "reference_cells": int(self.reference.shape[0]),
            "reference_dimensions": int(self.reference.shape[1]),
            "knn": int(self.n_neighbors),
            "connected_components": int(component_count),
            "heat_time": float(self.heat_time),
            "eigenvectors": int(eigenvalues.shape[0]),
            "cache_hit": bool(cache_hit),
            "cache_path": "" if cache_path is None else str(cache_path),
        }

    def _cache_key(self) -> str:
        digest = hashlib.sha256()
        digest.update(str(self.reference.shape).encode("utf-8"))
        digest.update(self.reference.tobytes(order="C"))
        digest.update(
            json.dumps(
                {
                    "version": 1,
                    "knn": self.n_neighbors,
                    "eigenvectors": self.n_eigenvectors,
                    "seed": self.seed,
                },
                sort_keys=True,
            ).encode("utf-8")
        )
        return digest.hexdigest()[:20]

    def _cache_path(self, cache_dir: str | Path | None) -> Path | None:
        if not cache_dir:
            return None
        directory = Path(cache_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"heat_kernel_geometry_{self._cache_key()}.npz"

    def _load_cache(self, path: Path | None):
        if path is None or not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as payload:
                eigenvalues = np.asarray(payload["eigenvalues"], dtype=np.float64)
                eigenvectors = np.asarray(payload["eigenvectors"], dtype=np.float64)
                component_count = int(payload["component_count"])
            if eigenvectors.shape != (
                self.reference.shape[0],
                eigenvalues.shape[0],
            ):
                raise ValueError("cached eigenvector shape does not match graph")
            return eigenvalues, eigenvectors, component_count
        except (KeyError, OSError, ValueError) as exc:
            warnings.warn(f"Ignoring invalid SF2M geodesic cache {path}: {exc}")
            return None

    @staticmethod
    def _save_cache(
        path: Path | None,
        *,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        component_count: int,
    ) -> None:
        if path is None:
            return
        temporary = path.with_suffix(".tmp.npz")
        try:
            np.savez_compressed(
                temporary,
                eigenvalues=np.asarray(eigenvalues, dtype=np.float64),
                eigenvectors=np.asarray(eigenvectors, dtype=np.float32),
                component_count=np.asarray(component_count, dtype=np.int64),
            )
            os.replace(temporary, path)
        except OSError as exc:
            warnings.warn(f"Could not cache SF2M geodesic geometry at {path}: {exc}")

    def _register_reference_rows(self) -> None:
        for index, row in enumerate(self.reference):
            self._row_to_node[_row_key(row)] = int(index)

    def _ensure_nearest_model(self) -> NearestNeighbors:
        if self._nearest_model is None:
            self._nearest_model = NearestNeighbors(
                n_neighbors=1,
                algorithm="auto",
                metric="euclidean",
                n_jobs=-1,
            ).fit(self.reference)
        return self._nearest_model

    def _register_known_rows(self, values: np.ndarray) -> None:
        missing_rows = []
        missing_keys = []
        for row in values:
            key = _row_key(row)
            if key not in self._row_to_node:
                missing_rows.append(row)
                missing_keys.append(key)
        if not missing_rows:
            return
        nearest = self._ensure_nearest_model().kneighbors(
            np.asarray(missing_rows, dtype=np.float32),
            return_distance=False,
        )[:, 0]
        for key, index in zip(missing_keys, nearest):
            self._row_to_node[key] = int(index)

    def _node_indices(self, values: torch.Tensor) -> np.ndarray:
        array = (
            values.detach()
            .reshape(values.shape[0], -1)
            .to(device="cpu", dtype=torch.float32)
            .numpy()
        )
        indices = np.empty(array.shape[0], dtype=np.int64)
        unknown_rows = []
        unknown_positions = []
        for position, row in enumerate(array):
            index = self._row_to_node.get(_row_key(row))
            if index is None:
                unknown_rows.append(row)
                unknown_positions.append(position)
            else:
                indices[position] = index
        if unknown_rows:
            nearest = self._ensure_nearest_model().kneighbors(
                np.asarray(unknown_rows, dtype=np.float32),
                return_distance=False,
            )[:, 0]
            for position, row, index in zip(
                unknown_positions, unknown_rows, nearest
            ):
                index = int(index)
                indices[position] = index
                self._row_to_node[_row_key(row)] = index
        return indices

    def _knn_graph(self):
        query_neighbors = self.n_neighbors + 1
        if self.reference.shape[0] <= 10_000:
            model = NearestNeighbors(
                n_neighbors=query_neighbors,
                algorithm="auto",
                metric="euclidean",
                n_jobs=-1,
            ).fit(self.reference)
            distances, neighbors = model.kneighbors(self.reference)
        else:
            try:
                from pynndescent import NNDescent

                model = NNDescent(
                    self.reference,
                    n_neighbors=query_neighbors,
                    metric="euclidean",
                    random_state=self.seed,
                    n_jobs=-1,
                    low_memory=True,
                )
                neighbors, distances = model.neighbor_graph
            except (ImportError, RuntimeError) as exc:
                warnings.warn(
                    "Approximate kNN construction was unavailable; falling back "
                    f"to exact neighbours ({exc})."
                )
                model = NearestNeighbors(
                    n_neighbors=query_neighbors,
                    algorithm="auto",
                    metric="euclidean",
                    n_jobs=-1,
                ).fit(self.reference)
                distances, neighbors = model.kneighbors(self.reference)

        row_indices = []
        column_indices = []
        edge_distances = []
        local_scales = np.empty(self.reference.shape[0], dtype=np.float64)
        for row in range(self.reference.shape[0]):
            keep = neighbors[row] != row
            row_neighbors = np.asarray(neighbors[row][keep], dtype=np.int64)[
                : self.n_neighbors
            ]
            row_distances = np.asarray(distances[row][keep], dtype=np.float64)[
                : self.n_neighbors
            ]
            if row_neighbors.shape[0] != self.n_neighbors:
                raise RuntimeError(
                    "kNN graph construction returned too few neighbours."
                )
            row_indices.extend([row] * self.n_neighbors)
            column_indices.extend(row_neighbors.tolist())
            edge_distances.extend(row_distances.tolist())
            local_scales[row] = row_distances[-1]

        positive_scales = local_scales[local_scales > 0.0]
        fallback_scale = (
            float(np.median(positive_scales)) if positive_scales.size else 1.0
        )
        local_scales = np.where(local_scales > 0.0, local_scales, fallback_scale)
        rows = np.asarray(row_indices, dtype=np.int64)
        columns = np.asarray(column_indices, dtype=np.int64)
        distances = np.asarray(edge_distances, dtype=np.float64)
        denominators = np.maximum(
            local_scales[rows] * local_scales[columns],
            np.finfo(np.float64).tiny,
        )
        weights = np.exp(-(distances**2) / denominators)
        graph = sparse.csr_matrix(
            (weights, (rows, columns)),
            shape=(self.reference.shape[0], self.reference.shape[0]),
            dtype=np.float64,
        )
        graph = graph.maximum(graph.T)
        graph.setdiag(0.0)
        graph.eliminate_zeros()
        return graph

    def _build_spectrum(self):
        graph = self._knn_graph()
        component_count, _ = connected_components(graph, directed=False)
        if component_count > 1:
            warnings.warn(
                "The SF2M geometry kNN graph has "
                f"{component_count} connected components. Cross-component heat "
                "kernel entries will receive the maximum finite ground cost."
            )

        # Diffusion-map alpha=1 density correction removes sampling-density
        # bias before forming the symmetric normalized graph Laplacian.
        density = np.asarray(graph.sum(axis=1)).reshape(-1)
        density_inverse = 1.0 / np.maximum(density, np.finfo(np.float64).tiny)
        corrected = (
            sparse.diags(density_inverse)
            @ graph
            @ sparse.diags(density_inverse)
        )
        degree = np.asarray(corrected.sum(axis=1)).reshape(-1)
        degree_inverse_sqrt = 1.0 / np.sqrt(
            np.maximum(degree, np.finfo(np.float64).tiny)
        )
        normalized_affinity = (
            sparse.diags(degree_inverse_sqrt)
            @ corrected
            @ sparse.diags(degree_inverse_sqrt)
        )
        laplacian = sparse.eye(
            graph.shape[0], dtype=np.float64, format="csr"
        ) - normalized_affinity

        try:
            eigenvalues, eigenvectors = eigsh(
                laplacian,
                k=self.n_eigenvectors,
                which="SM",
                tol=1e-5,
            )
        except ArpackNoConvergence as exc:
            if (
                exc.eigenvalues is None
                or exc.eigenvectors is None
                or exc.eigenvalues.shape[0] < 2
            ):
                raise RuntimeError(
                    "The SF2M heat-kernel eigensolver did not converge. Reduce "
                    "sf2m_geodesic_eigenvectors or graph size."
                ) from exc
            warnings.warn(
                "The SF2M heat-kernel eigensolver only partially converged; "
                f"using {exc.eigenvalues.shape[0]} eigenvectors."
            )
            eigenvalues, eigenvectors = exc.eigenvalues, exc.eigenvectors

        order = np.argsort(eigenvalues)
        eigenvalues = np.maximum(
            np.asarray(eigenvalues[order], dtype=np.float64), 0.0
        )
        eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.float64)
        return eigenvalues, eigenvectors, int(component_count)

    def squared_cost(self, x0: torch.Tensor, x1: torch.Tensor) -> np.ndarray:
        """Return ``-log(H_t(x0, x1))``, the squared cost in paper Eq. 12."""
        source = self.heat_features[self._node_indices(x0)]
        target = self.heat_features[self._node_indices(x1)]
        heat_kernel = source @ target.T
        heat_kernel = np.clip(heat_kernel, self.heat_epsilon, 1.0)
        return -np.log(heat_kernel)


class GeodesicSinkhornPlanSampler:
    """Sample minibatch pairs from the paper's Geodesic Sinkhorn plan."""

    def __init__(
        self,
        geometry: HeatKernelGeodesicCost,
        *,
        reg: float,
        max_iterations: int,
        tolerance: float,
    ):
        self.geometry = geometry
        self.reg = float(reg)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        if self.reg <= 0.0:
            raise ValueError("Geodesic Sinkhorn regularization must be positive.")
        if self.max_iterations <= 0:
            raise ValueError("sf2m_geodesic_sinkhorn_max_iter must be positive.")
        if self.tolerance <= 0.0:
            raise ValueError("sf2m_geodesic_sinkhorn_tol must be positive.")

    def get_map(self, x0: torch.Tensor, x1: torch.Tensor) -> np.ndarray:
        cost = self.geometry.squared_cost(x0, x1)
        source_weights = pot.unif(x0.shape[0])
        target_weights = pot.unif(x1.shape[0])
        plan = pot.sinkhorn(
            source_weights,
            target_weights,
            cost,
            reg=self.reg,
            method="sinkhorn_log",
            numItermax=self.max_iterations,
            stopThr=self.tolerance,
            warn=True,
        )
        plan = np.asarray(plan, dtype=np.float64)
        if not np.all(np.isfinite(plan)) or float(plan.sum()) <= 0.0:
            raise RuntimeError("Geodesic Sinkhorn produced an invalid transport plan.")
        plan = np.maximum(plan, 0.0)
        return plan / plan.sum()

    def sample_plan(self, x0: torch.Tensor, x1: torch.Tensor, replace=True):
        plan = self.get_map(x0, x1)
        choices = np.random.choice(
            plan.size,
            p=plan.reshape(-1),
            size=x0.shape[0],
            replace=replace,
        )
        source_indices, target_indices = np.divmod(choices, plan.shape[1])
        source_indices = torch.as_tensor(
            source_indices, dtype=torch.long, device=x0.device
        )
        target_indices = torch.as_tensor(
            target_indices, dtype=torch.long, device=x1.device
        )
        return (
            x0.index_select(0, source_indices),
            x1.index_select(0, target_indices),
        )
