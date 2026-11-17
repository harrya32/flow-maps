"""Audit empirical endpoint support; this does not train or evaluate a flow.

Run in the scientific Python environment used for the classifier experiments.
Allowed pairs come from the reconstructed graph, with reachability projected
before tissue labels are merged. Each snapshot is normalized independently.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def coupling_support(source, target, allowed):
    """Maximize allowed mass with empirical source/target capacity constraints."""
    p = source.value_counts(normalize=True)
    q = target.value_counts(normalize=True)
    source_index = {label: i for i, label in enumerate(p.index)}
    target_index = {label: i + len(p) for i, label in enumerate(q.index)}
    edges = [(u, v) for u in p.index for v in q.index if (u, v) in allowed]
    orphan_mass = sum(p[u] for u in p.index if not any((u, v) in allowed for v in q.index))
    independent_valid_mass = sum(p[u] * q[v] for u, v in edges)
    if not edges:
        return 0.0, float(orphan_mass), float(independent_valid_mass)

    n_edges = len(edges)
    rows = [source_index[u] for u, _ in edges] + [target_index[v] for _, v in edges]
    columns = list(range(n_edges)) * 2
    capacities = coo_matrix(
        (np.ones(2 * n_edges), (rows, columns)),
        shape=(len(p) + len(q), n_edges),
    ).tocsr()
    result = linprog(
        -np.ones(n_edges),
        A_ub=capacities,
        b_ub=np.concatenate([p.values, q.values]),
        bounds=(0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)
    supported = float(np.clip(-result.fun, 0, 1))
    return supported, float(orphan_mass), float(independent_valid_mass)


def main():
    metadata = pd.read_csv(
        ROOT / "datasets/calderon2022/lineage_assigned_metadata.csv.gz",
        usecols=["day", "branch_id", "broad_tissue"],
    )
    allowed = {
        column: set(map(tuple, pd.read_csv(HERE / "graph_data" / f"{stem}_allowed_pairs.csv").values))
        for column, stem in [("branch_id", "branch"), ("broad_tissue", "broad_tissue")]
    }
    bins = sorted(metadata.day.unique(), key=lambda value: int(value.split("-")[0]))
    rows = []
    for cohort in ["all_assigned", "Ubi_only"]:
        subset = metadata if cohort == "all_assigned" else metadata[metadata.branch_id.str.startswith("Ubi_")]
        for i, source_bin in enumerate(bins):
            for target_bin in bins[i + 1:]:
                source = subset[subset.day == source_bin]
                target = subset[subset.day == target_bin]
                row = {
                    "cohort": cohort,
                    "source": source_bin,
                    "target": target_bin,
                    "source_cells": len(source),
                    "target_cells": len(target),
                    "source_branches": source.branch_id.nunique(),
                    "target_branches": target.branch_id.nunique(),
                    "source_tissues": source.broad_tissue.nunique(),
                    "target_tissues": target.broad_tissue.nunique(),
                }
                for column in ["branch_id", "broad_tissue"]:
                    supported, orphan, independent = coupling_support(source[column], target[column], allowed[column])
                    row[f"{column}_max_supported_mass"] = supported
                    row[f"{column}_forced_violation_fraction"] = 1 - supported
                    row[f"{column}_orphan_source_mass"] = orphan
                    row[f"{column}_independent_violation_fraction"] = 1 - independent
                rows.append(row)
    result = pd.DataFrame(rows)
    path = HERE / "timepoint_support.csv"
    result.to_csv(path, index=False)
    print(f"Saved {len(result)} interval/cohort audits to {path}")
    print(result[(result.source == "4-6") & (result.target == "12-14")].to_string(index=False))


if __name__ == "__main__":
    main()
