"""Reproduce the preliminary Calderon RNA lineage-comparison figures.

The authors' complete weighted RNA state-edge export has not been located.
All graphs here are explicit reconstructions from the released lineage IDs,
annotations, inferred-time windows, and lineage decoder. They are not an
independent biological validation or an exact reproduction of Figure 3C.

Dependencies: numpy, openpyxl, networkx, matplotlib, and either the Graphviz
dot executable or the Codex bundled Node/@viz-js/viz runtime.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import textwrap
from collections import Counter, defaultdict

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/calderon2022-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Patch, PathPatch, Polygon
from matplotlib.path import Path as MplPath
import networkx as nx
import numpy as np
import openpyxl


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FIGURES = HERE / "figures"
GRAPH_DATA = HERE / "graph_data"
PDF_OUTPUT = ROOT / "output" / "pdf" / "calderon2022_lineage_graphs.pdf"

INK = "#172B3A"
MUTED = "#526574"
RED = "#B64650"
PALETTE = {
    "Unknown": "#DCE2E7",
    "maternal": "#DCE2E7",
    "unknown": "#DCE2E7",
    "Blastoderm": "#E5DFF1",
    "blastoderm": "#E5DFF1",
    "Ectoderm": "#F8E6AE",
    "ectoderm": "#F8E6AE",
    "Endoderm": "#DCEBCC",
    "endoderm": "#DCEBCC",
    "Mesoderm": "#F0D4D6",
    "mesoderm": "#F0D4D6",
    "Neuroectoderm": "#D4E5F4",
    "neuroectoderm": "#D4E5F4",
    "ExE embryo": "#F8DDC2",
    "extraembryonic": "#F8DDC2",
}


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reachability_pairs(graph, label=lambda node: node):
    return {
        (label(source), label(target))
        for source in graph
        for target in nx.descendants(graph, source)
        if label(source) != label(target)
    }


def project_graph(graph, label):
    projected = nx.DiGraph()
    projected.add_nodes_from(label(node) for node in graph)
    projected.add_edges_from(
        (label(source), label(target))
        for source, target in graph.edges
        if label(source) != label(target)
    )
    return projected


def graph_metrics(graph):
    cyclic = sorted(
        [sorted(component) for component in nx.strongly_connected_components(graph)
         if len(component) > 1],
        key=lambda component: (-len(component), component),
    )
    dag = nx.is_directed_acyclic_graph(graph)
    condensed = nx.condensation(graph)
    return {
        "nodes": len(graph),
        "nonself_edges": graph.number_of_edges(),
        "weak_components": nx.number_weakly_connected_components(graph),
        "roots": sum(graph.in_degree(node) == 0 for node in graph),
        "leaves": sum(graph.out_degree(node) == 0 for node in graph),
        "nodes_with_multiple_children": sum(graph.out_degree(node) > 1 for node in graph),
        "nodes_with_multiple_parents": sum(graph.in_degree(node) > 1 for node in graph),
        "max_children": max(dict(graph.out_degree()).values(), default=0),
        "is_dag": dag,
        "longest_dag_path_edges": nx.dag_longest_path_length(graph) if dag else None,
        "longest_dag_path": nx.dag_longest_path(graph) if dag else None,
        "cyclic_groups": cyclic,
        "cyclic_nodes": sum(map(len, cyclic)),
        "condensation_nodes": len(condensed),
        "condensation_longest_path_edges": nx.dag_longest_path_length(condensed),
    }


def add_support_metrics(metrics, graph, allowed):
    closure = reachability_pairs(graph)
    metrics.update({
        "projected_allowed_nonself_pairs": len(allowed),
        "projected_allowed_pairs_with_self": len(allowed) + len(graph),
        "projected_nonself_support_fraction": len(allowed) / (len(graph) * (len(graph) - 1)),
        "closure_of_merged_edges_nonself_pairs": len(closure),
        "extra_pairs_created_by_second_closure": len(closure - allowed),
        "extra_pair_examples": [list(pair) for pair in sorted(closure - allowed)[:25]],
    })


def load_graphs():
    inspection = json.loads((HERE / "inspection.json").read_text())
    hierarchy = inspection["candidate_prefix_hierarchy"]
    branches = nx.DiGraph()
    branches.add_nodes_from(hierarchy["nodes"])
    branches.add_edges_from(map(tuple, hierarchy["edges"]))

    decoder = {}
    decoder_duplicates = []
    for line in (HERE / "celltype_list_rna.txt").read_text().splitlines():
        branch, germ, tissue = line.split("\t")
        if branch in decoder:
            assert decoder[branch] == {"germ": germ, "tissue": tissue}
            decoder_duplicates.append(branch)
        decoder[branch] = {"germ": germ, "tissue": tissue}
    assert set(branches) <= set(decoder)

    rows = list(openpyxl.load_workbook(
        HERE / "table_s1_withLineage.xlsx", data_only=True
    ).active.values)[1:]
    states = nx.DiGraph()
    by_time = defaultdict(list)
    all_types = set()
    for row in rows:
        window, cluster, label, germ, comment, branch = (
            row[0], row[1], str(row[6]).strip(), str(row[7]).strip(), row[9], row[11]
        )
        if window == "18-20":
            continue
        all_types.add(label)
        if branch in (None, "NA"):
            continue
        state_id = f"{window}:{cluster}"
        assert state_id not in states
        time = int(window.split("-")[0])
        states.add_node(state_id, window=window, time=time, cluster=int(cluster),
                        branch=branch, label=label, germ=germ, comment=comment or "")
        by_time[time].append(state_id)

    depths = {node: max([len(nx.shortest_path(branches, root, node)) - 1
                        for root in nx.ancestors(branches, node)] + [0])
              for node in branches}
    ambiguous = []
    bridges = []
    later_roots = []
    for child, attributes in list(states.nodes(data=True)):
        time = attributes["time"]
        ancestors = nx.ancestors(branches, attributes["branch"]) | {attributes["branch"]}
        parents = [node for node in by_time[time - 2]
                   if states.nodes[node]["branch"] in ancestors]
        gap = False
        if not parents and time > 0:
            earlier = [node for node, data in states.nodes(data=True)
                       if data["time"] < time and data["branch"] in ancestors]
            if earlier:
                latest = max(states.nodes[node]["time"] for node in earlier)
                parents = [node for node in earlier if states.nodes[node]["time"] == latest]
                gap = True
        if not parents:
            if time > 0:
                later_roots.append(child)
            continue
        closest = max(depths[states.nodes[node]["branch"]] for node in parents)
        parents = [node for node in parents if depths[states.nodes[node]["branch"]] == closest]
        if len(parents) > 1:
            ambiguous.append({"child": child, "candidate_parents": sorted(parents)})
        for parent in parents:
            rule = "closest_ancestor_adjacent_window" if not gap else "closest_ancestor_gap_bridge"
            states.add_edge(parent, child, rule=rule, ambiguous_parent=len(parents) > 1)
            if gap:
                bridges.append({"source": parent, "target": child,
                                "time_gap_hours": time - states.nodes[parent]["time"]})
    assert nx.is_directed_acyclic_graph(states)

    projected_branches = project_graph(states, lambda node: states.nodes[node]["branch"])
    assert set(projected_branches.edges) == set(branches.edges), (
        "The reconstructed temporal edges no longer reproduce the branch-prefix links."
    )
    types = project_graph(states, lambda node: states.nodes[node]["label"])
    broad = project_graph(branches, lambda branch: decoder[branch]["tissue"])

    branch_support = reachability_pairs(branches)
    type_support = reachability_pairs(states, lambda node: states.nodes[node]["label"])
    broad_support = reachability_pairs(branches, lambda branch: decoder[branch]["tissue"])
    state_supported_branches = reachability_pairs(states, lambda node: states.nodes[node]["branch"])

    metrics = {"branches": graph_metrics(branches), "cell_states": graph_metrics(types),
               "broad_tissues": graph_metrics(broad), "temporal_source": graph_metrics(states)}
    for key, graph, support in [("branches", branches, branch_support),
                                ("cell_states", types, type_support),
                                ("broad_tissues", broad, broad_support)]:
        add_support_metrics(metrics[key], graph, support)
    metrics["branches"]["temporally_witnessed_nonself_branch_pairs"] = len(state_supported_branches)
    metrics["branches"]["prefix_pairs_not_witnessed_in_temporal_reconstruction"] = len(
        branch_support - state_supported_branches
    )

    biological_labels = defaultdict(set)
    type_germs = defaultdict(Counter)
    for _, attributes in states.nodes(data=True):
        biological_labels[attributes["branch"]].add(attributes["label"])
        type_germs[attributes["label"]][attributes["germ"]] += 1
    mixed_annotations = [node for node, attributes in states.nodes(data=True)
                         if "mixed" in attributes["comment"].lower()]
    scope = {
        "modality": "RNA",
        "inferred_age_hours": [0, 18],
        "annotated_temporal_states": len(states),
        "assigned_cells": sum(row["n_cells"] for row in inspection["lineage_support"]),
        "all_atlas_annotation_strings": len(all_types),
        "assigned_subset_annotation_strings": len(types),
        "annotation_strings_without_assigned_branch": sorted(all_types - set(types)),
        "branches_containing_multiple_annotation_strings": sum(len(labels) > 1 for labels in biological_labels.values()),
        "duplicate_decoder_lines": decoder_duplicates,
        "ambiguous_parent_assignments": ambiguous,
        "gap_bridges": bridges,
        "unconnected_states_after_initial_window": later_roots,
        "states_with_mixed_annotation_comments": mixed_annotations,
    }
    strict_states = states.copy()
    strict_states.remove_edges_from((source,target) for source,target,data in states.edges(data=True)
                                    if data["ambiguous_parent"] or "gap_bridge" in data["rule"])
    strict_types = project_graph(strict_states, lambda node: strict_states.nodes[node]["label"])
    strict_support = reachability_pairs(strict_states, lambda node: strict_states.nodes[node]["label"])
    scope["uncertain_link_sensitivity"] = {
        "rule": "Remove every ambiguous-parent edge and the inferred gap bridge; retain all nodes.",
        "temporal_edges": strict_states.number_of_edges(),
        "cell_state_edges": strict_types.number_of_edges(),
        "cell_state_cyclic_groups": [sorted(group) for group in nx.strongly_connected_components(strict_types) if len(group)>1],
        "projected_allowed_nonself_pairs": len(strict_support),
    }
    return branches, states, types, broad, decoder, type_germs, metrics, scope, type_support, broad_support


def extract_repo_graph(path, classes_key, edges_key):
    constants = {}
    for statement in ast.parse(path.read_text()).body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id in (classes_key, edges_key):
                    constants[target.id] = ast.literal_eval(statement.value)
    graph = nx.DiGraph()
    graph.add_nodes_from(constants[classes_key])
    graph.add_edges_from(constants[edges_key])
    return graph


def dot_quote(value):
    return json.dumps(str(value), ensure_ascii=False)


def make_dot(graph, labels, colors, cyclic_groups=()):
    lines = ['digraph G {', 'graph [rankdir=LR, newrank=true, nodesep=0.22, ranksep=0.52, '
             'splines=spline, outputorder=edgesfirst, bgcolor="white", pad=0.1];',
             'node [shape=box, style="rounded,filled", fontname="DejaVu Sans", '
             'fontsize=11, margin="0.10,0.065", color="#A8B4BE", penwidth=0.7];',
             'edge [color="#91A1AD", penwidth=1, arrowsize=0.55];']
    cycle_nodes = set().union(*map(set, cyclic_groups)) if cyclic_groups else set()
    for node in graph:
        label = labels[node]
        lines.append(f'{dot_quote(node)} [label={dot_quote(label)}, '
                     f'fillcolor={dot_quote(colors[node])}, '
                     f'color={dot_quote(RED if node in cycle_nodes else "#A8B4BE")}];')
    if cyclic_groups:
        for index, group in enumerate(cyclic_groups, 1):
            lines.append(f'subgraph cluster_cycle_{index} {{label="Cycle group {index}: {len(group)} labels"; '
                         'style="rounded"; color="#D59DA2"; fontsize=11; fontname="DejaVu Sans"; '
                         + '; '.join(map(dot_quote, group)) + ';}')
        condensed = nx.condensation(graph)
        layer = {}
        for node in nx.topological_sort(condensed):
            layer[node] = max([layer[parent] + 1 for parent in condensed.predecessors(node)] + [0])
        for value in sorted(set(layer.values())):
            members = sorted(member for node in condensed if layer[node] == value
                             for member in condensed.nodes[node]["members"])
            lines.append('{rank=same; ' + '; '.join(map(dot_quote, members)) + ';}')
    for source, target in graph.edges:
        internal_cycle = any(source in group and target in group for group in cyclic_groups)
        color = RED if internal_cycle else "#91A1AD"
        lines.append(f'{dot_quote(source)} -> {dot_quote(target)} [color={dot_quote(color)}];')
    lines.append('}')
    return '\n'.join(lines)


def layout_dot(dot_path, json_path, node_path, viz_module):
    if shutil.which("dot"):
        with json_path.open("w") as output:
            subprocess.run(["dot", "-Tjson", str(dot_path)], stdout=output, check=True)
    else:
        subprocess.run([str(node_path), str(HERE / "render_graphviz.cjs"),
                        str(viz_module), str(dot_path), str(json_path)], check=True)
    return json.loads(json_path.read_text())


def curve_path(points):
    points = np.asarray(points, dtype=float)
    return MplPath(points, [MplPath.MOVETO] + [MplPath.CURVE4] * (len(points) - 1))


def draw_layout(ax, layout, labels=True, fonts=9.5):
    ax.set_axis_off()
    xmin, ymin, xmax, ymax = map(float, layout["bb"].split(','))
    for obj in layout.get("objects", []):
        if obj.get("name", "").startswith("cluster_") and "bb" in obj:
            x0, y0, x1, y1 = map(float, obj["bb"].split(','))
            ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                       boxstyle="round,pad=0,rounding_size=7",
                                       facecolor="#FCF6F6", edgecolor="#D59DA2", linewidth=0.7, zorder=0))
            if labels:
                ax.text((x0+x1)/2, y1-9, obj.get("label", ""), ha="center", va="top",
                        fontsize=fonts * 0.8, color=RED)
    for edge in layout.get("edges", []):
        color = edge.get("color", "#91A1AD")
        for op in edge.get("_draw_", []):
            if op["op"] == "b":
                ax.add_patch(PathPatch(curve_path(op["points"]), facecolor="none", edgecolor=color,
                                       linewidth=0.7 if labels else 0.6, alpha=0.9, zorder=1))
        for op in edge.get("_hdraw_", []):
            if op["op"] == "P":
                ax.add_patch(Polygon(op["points"], closed=True, facecolor=color,
                                     edgecolor=color, linewidth=0.2, zorder=1))
    for obj in layout.get("objects", []):
        if "pos" not in obj:
            continue
        x, y = map(float, obj["pos"].split(','))
        w, h = float(obj["width"]) * 72, float(obj["height"]) * 72
        ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                                   boxstyle="round,pad=0,rounding_size=5",
                                   facecolor=obj.get("fillcolor", "#DCE2E7"),
                                   edgecolor=obj.get("color", "#A8B4BE"), linewidth=0.75, zorder=2))
        if labels:
            label = obj.get("label", obj["name"]).replace('\\n', '\n')
            ax.text(x, y, label, ha="center", va="center",
                    fontsize=fonts, color=INK, linespacing=1.13, zorder=3)
    ax.set_xlim(xmin - 8, xmax + 8)
    ax.set_ylim(ymin - 8, ymax + 8)
    ax.set_aspect('equal')


def save_figure(fig, stem, pdf):
    fig.savefig(FIGURES / f"{stem}.png", dpi=190, facecolor="white")
    fig.savefig(FIGURES / f"{stem}.svg", facecolor="white")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def graph_page(layout, title, subtitle, footer, metrics):
    _, _, width, height = map(float, layout["bb"].split(','))
    aspect = width / height
    figure_width = max(15.5, min(23, 17 * aspect))
    figure_height = max(12.5, min(23, figure_width / aspect + 2.4))
    fig = plt.figure(figsize=(figure_width, figure_height), facecolor="white")
    fig.text(0.045, 0.967, title, fontsize=20, weight="bold", color=INK, va="top")
    fig.text(0.045, 0.931, subtitle, fontsize=11.5, color=MUTED, va="top")
    depth = str(metrics["longest_dag_path_edges"]) if metrics["is_dag"] else "cyclic"
    summary = (f'{metrics["nodes"]} nodes   |   {metrics["nonself_edges"]} non-self edges   |   '
               f'{metrics["nodes_with_multiple_children"]} nodes with 2+ children   |   longest DAG path: {depth}')
    fig.text(0.045, 0.901, summary, fontsize=11.5, color=INK, va="top")
    legend_keys = ["Unknown", "Blastoderm", "Ectoderm", "Endoderm", "Mesoderm", "Neuroectoderm", "ExE embryo"]
    fig.legend([Patch(facecolor=PALETTE[key], edgecolor="#A8B4BE") for key in legend_keys],
               legend_keys, loc="upper left", bbox_to_anchor=(0.041,0.877), ncol=7,
               frameon=False, fontsize=8.5, handlelength=1.1, columnspacing=1.2)
    ax = fig.add_axes([0.045, 0.09, 0.91, 0.755])
    draw_layout(ax, layout, fonts=10.2)
    fig.text(0.045, 0.057, footer, fontsize=9, color=MUTED, va="top", linespacing=1.35)
    return fig


def overview(layouts, metrics, scope):
    fig = plt.figure(figsize=(18, 12.5), facecolor="white")
    fig.text(0.04, 0.966, "Calderon RNA: branch identity versus cell-state labels", fontsize=23,
             weight="bold", color=INK, va="top")
    fig.text(0.04, 0.93, f'Preliminary reconstruction | 0-18 h | {scope["annotated_temporal_states"]} assigned temporal states | '
             f'{scope["assigned_cells"]:,} cells', fontsize=12, color=MUTED, va="top")
    cards = [("Branch IDs", "branches", 0.04), ("Finer cell-state labels", "cell_states", 0.36),
             ("Broad tissue decoder", "broad_tissues", 0.68)]
    for title, key, left in cards:
        met = metrics[key]
        color = "#FDF5F5" if not met["is_dag"] else "#F3F7FA"
        box = FancyBboxPatch((left, 0.73), 0.28, 0.17, boxstyle="round,pad=0.012,rounding_size=0.012",
                             transform=fig.transFigure, facecolor=color, edgecolor="none")
        fig.add_artist(box)
        fig.text(left+0.013, 0.882, title, fontsize=14, weight="bold", color=INK, va="top")
        fig.text(left+0.013, 0.848, f'{met["nodes"]} nodes / {met["nonself_edges"]} edges',
                 fontsize=22, weight="bold", color=INK, va="top")
        fig.text(left+0.013, 0.807, f'{met["nodes_with_multiple_children"]} nodes with 2+ children',
                 fontsize=12, color=MUTED, va="top")
        status = (f'DAG; graph path {met["longest_dag_path_edges"]} edges' + ('*' if key=="broad_tissues" else '')
                  if met["is_dag"] else f'{len(met["cyclic_groups"])} cycle groups; {met["cyclic_nodes"]} labels involved')
        fig.text(left+0.013, 0.778, status, fontsize=12, color=RED if not met["is_dag"] else MUTED, va="top")
    ax1 = fig.add_axes([0.035, 0.225, 0.445, 0.475])
    ax2 = fig.add_axes([0.52, 0.225, 0.445, 0.475])
    draw_layout(ax1, layouts["branches"], labels=False)
    draw_layout(ax2, layouts["cell_states"], labels=False)
    fig.text(0.04, 0.707, "A  Branch hierarchy", fontsize=15, weight="bold", color=INK)
    fig.text(0.53, 0.707, "B  Cell-state graph (red marks cycles)", fontsize=15, weight="bold", color=INK)
    fig.text(0.04, 0.184, "Merging labels barely reduces node count", fontsize=14, weight="bold", color=INK)
    fig.text(0.04, 0.151, f'{metrics["branches"]["nodes"]} -> {metrics["cell_states"]["nodes"]} labels; '
             f'{scope["branches_containing_multiple_annotation_strings"]}/{metrics["branches"]["nodes"]} branches contain multiple cell-state names.',
             fontsize=12, color=MUTED)
    fig.text(0.53, 0.184, "Merging changes allowable paths", fontsize=14, weight="bold", color=INK)
    fig.text(0.53, 0.151, f'{metrics["cell_states"]["projected_allowed_nonself_pairs"]} witnessed type pairs; '
             f'{metrics["cell_states"]["closure_of_merged_edges_nonself_pairs"]} after closing the merged graph.',
             fontsize=12, color=MUTED)
    fig.text(0.53, 0.123, '*Broad-decoder depth is graph depth; merging can extend paths.',
             fontsize=9.5, color=MUTED)
    fig.text(0.04, 0.101, 'Current repo: Maizels 9 types / 8 edges / depth 4; CITE-Multi 7 types / 6 edges / depth 1.',
             fontsize=11.5, color=INK)
    fig.text(0.04, 0.065, 'Graphs are reconstructed from released IDs and annotations, not the complete weighted Fig. 3C edge export.\n'
             'All arrows shown are direct non-self edges. Full node labels, edge tables and construction rules accompany these figures.',
             fontsize=9.5, color=MUTED, linespacing=1.4)
    return fig


def support_figure(branches, types, branch_support, type_support, metrics):
    fig, axes = plt.subplots(1, 2, figsize=(16, 9.5), gridspec_kw={"wspace": 0.23})
    fig.subplots_adjust(left=0.06, right=0.97, top=0.81, bottom=0.2)
    fig.suptitle("Allowable pairs: original reachability before merging", x=0.06, ha="left", y=0.965,
                 fontsize=22, weight="bold", color=INK)
    fig.text(0.06, 0.91, "Rows are source labels; columns are destination labels. Self pairs are included.", fontsize=12, color=MUTED)
    orders = [sorted(branches), sorted(types)]
    for ax, nodes, pairs, title in zip(axes, orders, [branch_support, type_support],
                                     ["Branch IDs", "Cell-state labels"]):
        matrix = np.array([[(source, target) in pairs or source == target for target in nodes]
                           for source in nodes], dtype=int)
        ax.imshow(matrix, cmap=matplotlib.colors.ListedColormap(["#F0F3F6", "#456C87"]), vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f'{title}: {len(pairs)} non-self allowed pairs', fontsize=14, color=INK, pad=13)
        ax.set_xlabel("Destination label (alphabetical order)", fontsize=11, color=MUTED)
        ax.set_ylabel("Source label (alphabetical order)", fontsize=11, color=MUTED)
        ax.set_xticks(np.arange(0, len(nodes), 10)); ax.set_yticks(np.arange(0, len(nodes), 10))
        for spine in ax.spines.values():spine.set_visible(False)
    fig.text(0.06, 0.106, f'Cell-state support is projected from the temporal reconstruction: {len(type_support)} non-self pairs. '
             f'Taking closure after merging would add {metrics["cell_states"]["extra_pairs_created_by_second_closure"]} more pairs.',
             fontsize=11.5, color=INK)
    fig.text(0.06, 0.06, "This is an inferred transition prior; cycles and mixed annotations require review before hard lineage constraints are used.",
             fontsize=10, color=MUTED)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    runtime = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node"
    parser.add_argument("--node", type=Path, default=runtime / "bin/node")
    parser.add_argument("--viz-module", type=Path, default=runtime / "node_modules/@viz-js/viz")
    args = parser.parse_args()
    for folder in (FIGURES, GRAPH_DATA, PDF_OUTPUT.parent):folder.mkdir(parents=True, exist_ok=True)

    branches, states, types, broad, decoder, type_germs, metrics, scope, type_support, broad_support = load_graphs()
    maizels = extract_repo_graph(ROOT / "py/common/maizels.py", "CLASS_NAMES", "TRANSITION_EDGES")
    metrics["maizels"] = graph_metrics(maizels)
    # CITE/Multi's published benchmark prior is HSC -> the six other types.
    cite = nx.DiGraph(); cite.add_nodes_from(["BP", "EryP", "HSC", "MasP", "MkP", "MoP", "NeuP"])
    cite.add_edges_from(("HSC", name) for name in cite if name != "HSC")
    metrics["cite_multi"] = graph_metrics(cite)
    source_files = [HERE / "table_s1_withLineage.xlsx", HERE / "celltype_list_rna.txt", HERE / "inspection.json"]
    output = {"status": "Preliminary reconstruction, not the complete weighted Figure 3C RNA edge export.",
              "scope": scope, "metrics": metrics,
              "sources": [{"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                          for path in source_files]}
    (GRAPH_DATA / "complexity.json").write_text(json.dumps(output, indent=2) + '\n')
    state_rows = [dict(state_id=node, **data) for node, data in states.nodes(data=True)]
    write_csv(GRAPH_DATA / "temporal_states.csv", state_rows,
              ["state_id", "window", "time", "cluster", "branch", "label", "germ", "comment"])
    write_csv(GRAPH_DATA / "temporal_edges.csv", [dict(source=u, target=v, **data) for u,v,data in states.edges(data=True)],
              ["source", "target", "rule", "ambiguous_parent"])
    for stem, graph in [("branch", branches), ("cell_state", types), ("broad_tissue", broad)]:
        write_csv(GRAPH_DATA / f"{stem}_direct_edges.csv", [dict(source=u,target=v) for u,v in sorted(graph.edges)],
                  ["source", "target"])
    for stem, graph, pairs in [("branch", branches, reachability_pairs(branches)),
                                ("cell_state", types, type_support), ("broad_tissue", broad, broad_support)]:
        all_pairs = sorted(pairs | {(node,node) for node in graph})
        write_csv(GRAPH_DATA / f"{stem}_allowed_pairs.csv", [dict(source=u,target=v) for u,v in all_pairs], ["source", "target"])
    write_csv(GRAPH_DATA / "branch_to_annotations.csv",
              [dict(branch=branch, annotation=label) for branch in branches
               for label in sorted({data["label"] for _,data in states.nodes(data=True) if data["branch"]==branch})],
              ["branch", "annotation"])

    definitions = {
        "branches": (branches, {node:node for node in branches},
                     {node:PALETTE[decoder[node]["germ"]] for node in branches}),
        "cell_states": (types, {node:'\n'.join(textwrap.wrap(node, 26, break_long_words=False)) for node in types},
                        {node:PALETTE.get(type_germs[node].most_common(1)[0][0], "#DCE2E7") for node in types}),
        "broad_tissues": (broad, {node:node.replace(' & ', ' &\n') for node in broad},
                          {node:PALETTE[Counter(decoder[b]["germ"] for b in branches if decoder[b]["tissue"]==node).most_common(1)[0][0]]
                           for node in broad}),
    }
    layouts = {}
    for key,(graph,labels,colors) in definitions.items():
        dot_path, json_path = GRAPH_DATA / f"{key}.dot", GRAPH_DATA / f"{key}.layout.json"
        dot_path.write_text(make_dot(graph, labels, colors, metrics[key]["cyclic_groups"]))
        layouts[key] = layout_dot(dot_path, json_path, args.node, args.viz_module)

    plt.rcParams.update({"font.family":"DejaVu Sans", "svg.fonttype":"none", "pdf.fonttype":42})
    with PdfPages(PDF_OUTPUT) as pdf:
        pdf.infodict().update(Title="Calderon 2022: branch and cell-type graph comparison",
                              Author="Analysis of the authors' public RNA lineage annotations",
                              Subject="Preliminary reconstruction; exact RNA state-edge export not available")
        save_figure(overview(layouts, metrics, scope), "lineage_comparison", pdf)
        save_figure(graph_page(layouts["branches"], "A  Branch-ID lineage hierarchy",
                    "One node per released branch ID; arrows follow the documented prefix-parent reconstruction.",
                    "Three roots are retained; synthetic plotting roots and self loops are omitted. Colours indicate the authors' broad germ-layer decoder.\n"
                    "Source: Calderon et al. (2022), released table_s1_withLineage.xlsx and celltype_list_rna.txt. Preliminary reconstruction.",
                    metrics["branches"]), "branch_id_graph", pdf)
        save_figure(graph_page(layouts["cell_states"], "B  Coarse graph over finer cell-state annotations",
                    "States sharing a manual annotation are merged across time and branches; red edges belong to directed cycles.",
                    "Arrows are direct non-self transitions in the time-ordered reconstruction. Cell-type allowability is computed before merging; do not close this graph again.\n"
                    f'One inferred gap bridge and {len(scope["ambiguous_parent_assignments"])} ambiguous parent assignments are retained and documented in the edge table.\n'
                    "Source: Calderon et al. (2022), released RNA lineage/annotation table. Mixed and ubiquitous states are retained. Preliminary reconstruction.",
                    metrics["cell_states"]), "cell_state_graph", pdf)
        save_figure(graph_page(layouts["broad_tissues"], "C  Additional coarsening to the authors' broad tissue decoder",
                    "Each branch ID is replaced by the decoder's third-column tissue label, giving a coarser 24-label graph.",
                    "These are broad branch-level tissue labels, not the 49 finer per-state annotations. Unknown branches are merged without biological curation.\n"
                    "Arrows are direct non-self projected links. Reachability must be projected from the branch graph before any additional closure.\n"
                    "Its apparent 6-edge depth includes concatenations of different branches and does not establish a 6-decision biological lineage.\n"
                    "Source: Calderon et al. (2022), released celltype_list_rna.txt and RNA lineage table. Preliminary reconstruction.",
                    metrics["broad_tissues"]), "broad_tissue_graph", pdf)
        save_figure(support_figure(branches, types, reachability_pairs(branches), type_support, metrics),
                    "allowable_pair_comparison", pdf)
    print(json.dumps({"pdf":str(PDF_OUTPUT), "figure_directory":str(FIGURES),
                      "scope":scope, "counts":{key:{name:metrics[key][name] for name in
                      ["nodes","nonself_edges","nodes_with_multiple_children","is_dag","longest_dag_path_edges"]}
                      for key in ["branches","cell_states","broad_tissues"]}}, indent=2))


if __name__ == "__main__":
    main()
