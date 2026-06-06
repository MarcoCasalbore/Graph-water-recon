#!/usr/bin/env python3
"""Sweep static graph decoders over a saved DGM soft score matrix.

This script is diagnostic: ground-truth edges are used only to score the
post-processing choices, not to train DGM or build the observable score.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def canonical(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def edge_set_from_edge_index(edge_index: np.ndarray) -> set[tuple[int, int]]:
    src = edge_index[0].astype(int)
    dst = edge_index[1].astype(int)
    return {canonical(int(u), int(v)) for u, v in zip(src, dst) if int(u) != int(v)}


def recovery(gt: set[tuple[int, int]], pred: set[tuple[int, int]]) -> dict[str, float | int]:
    tp = len(gt & pred)
    fp = len(pred - gt)
    fn = len(gt - pred)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def pairwise_dist(xy: np.ndarray) -> np.ndarray:
    return np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=-1))


def geometry_mst_edges(dist: np.ndarray) -> set[tuple[int, int]]:
    graph = nx.Graph()
    n = dist.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            graph.add_edge(i, j, weight=float(dist[i, j]))
    return {canonical(int(u), int(v)) for u, v in nx.minimum_spanning_edges(graph, data=False)}


def select_edges(
    score: np.ndarray,
    dist: np.ndarray,
    budget: int,
    length_gamma: float,
    length_tau: float,
    max_degree: int,
    ensure_connected: bool,
    include_mst: bool,
) -> set[tuple[int, int]]:
    n = score.shape[0]
    tau = max(float(length_tau), 1e-9)
    adjusted = score * np.exp(-float(length_gamma) * dist / tau)
    np.fill_diagonal(adjusted, 0.0)

    selected: set[tuple[int, int]] = set()
    degree = np.zeros(n, dtype=int)

    if include_mst:
        for u, v in geometry_mst_edges(dist):
            selected.add((u, v))
            degree[u] += 1
            degree[v] += 1

    pairs = [
        (float(adjusted[i, j]), i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if (i, j) not in selected
    ]
    pairs.sort(reverse=True)

    def can_add(u: int, v: int, strict_degree: bool = True) -> bool:
        if (u, v) in selected:
            return False
        if strict_degree and max_degree > 0 and (degree[u] >= max_degree or degree[v] >= max_degree):
            return False
        return True

    for _, u, v in pairs:
        if len(selected) >= budget:
            break
        if can_add(u, v, strict_degree=True):
            selected.add((u, v))
            degree[u] += 1
            degree[v] += 1

    if len(selected) < budget:
        for _, u, v in pairs:
            if len(selected) >= budget:
                break
            if can_add(u, v, strict_degree=False):
                selected.add((u, v))
                degree[u] += 1
                degree[v] += 1

    if ensure_connected:
        selected = connect_and_prune(selected, adjusted, dist, budget)

    return selected


def connect_and_prune(
    edges: set[tuple[int, int]],
    adjusted: np.ndarray,
    dist: np.ndarray,
    budget: int,
) -> set[tuple[int, int]]:
    n = adjusted.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(edges)
    selected = set(edges)

    while nx.number_connected_components(graph) > 1:
        components = [set(c) for c in nx.connected_components(graph)]
        best_edge = None
        best_value = -float("inf")
        for a_idx, comp_a in enumerate(components):
            for comp_b in components[a_idx + 1 :]:
                for u in comp_a:
                    for v in comp_b:
                        e = canonical(int(u), int(v))
                        if e in selected:
                            continue
                        value = float(adjusted[e] - 1e-3 * dist[e])
                        if value > best_value:
                            best_value = value
                            best_edge = e
        if best_edge is None:
            break
        selected.add(best_edge)
        graph.add_edge(*best_edge)

    if budget > 0 and graph.number_of_edges() > budget:
        candidates = sorted(
            list(selected),
            key=lambda e: (float(adjusted[e]), -float(dist[e]), e[0], e[1]),
        )
        for edge in candidates:
            if graph.number_of_edges() <= budget:
                break
            graph.remove_edge(*edge)
            if nx.is_connected(graph):
                selected.remove(edge)
            else:
                graph.add_edge(*edge)

    return selected


def edge_tensor(edges: set[tuple[int, int]]) -> torch.Tensor:
    rows = []
    for u, v in sorted(edges):
        rows.append((u, v))
        rows.append((v, u))
    return torch.tensor(rows, dtype=torch.long).T if rows else torch.empty((2, 0), dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep soft graph post-processing parameters.")
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--gammas", default="0,0.25,0.5,1,1.5,2")
    parser.add_argument("--taus", default="0.05,0.1,0.15,0.2,0.3,0.5,1.0")
    parser.add_argument("--max_degrees", default="0,3,4,5")
    parser.add_argument("--include_mst_options", default="0,1")
    parser.add_argument("--connect_options", default="0,1")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    score = np.load(args.score).astype(np.float64)
    data = np.load(args.npz)
    xy = data["node_xy"].astype(np.float64)
    if xy.ndim == 3:
        xy = xy[0]
    xy = xy[:, :2]
    dist = pairwise_dist(xy)
    gt_edges = edge_set_from_edge_index(data["edge_index"])

    gammas = [float(x) for x in args.gammas.split(",") if x]
    taus = [float(x) for x in args.taus.split(",") if x]
    max_degrees = [int(x) for x in args.max_degrees.split(",") if x]
    mst_options = [bool(int(x)) for x in args.include_mst_options.split(",") if x]
    connect_options = [bool(int(x)) for x in args.connect_options.split(",") if x]

    rows = []
    best = None
    best_edges: set[tuple[int, int]] = set()
    for gamma in gammas:
        for tau in taus:
            for max_degree in max_degrees:
                for include_mst in mst_options:
                    for ensure_connected in connect_options:
                        edges = select_edges(
                            score=score,
                            dist=dist,
                            budget=args.budget,
                            length_gamma=gamma,
                            length_tau=tau,
                            max_degree=max_degree,
                            ensure_connected=ensure_connected,
                            include_mst=include_mst,
                        )
                        rec = recovery(gt_edges, edges)
                        graph = nx.Graph()
                        graph.add_nodes_from(range(score.shape[0]))
                        graph.add_edges_from(edges)
                        lengths = [float(dist[u, v]) for u, v in edges]
                        row = {
                            "f1": rec["f1"],
                            "precision": rec["precision"],
                            "recall": rec["recall"],
                            "tp": rec["tp"],
                            "fp": rec["fp"],
                            "fn": rec["fn"],
                            "edges": len(edges),
                            "components": nx.number_connected_components(graph),
                            "length_mean": float(np.mean(lengths)) if lengths else 0.0,
                            "length_max": float(np.max(lengths)) if lengths else 0.0,
                            "gamma": gamma,
                            "tau": tau,
                            "max_degree": max_degree,
                            "include_mst": int(include_mst),
                            "ensure_connected": int(ensure_connected),
                        }
                        rows.append(row)
                        if best is None or row["f1"] > best["f1"]:
                            best = row
                            best_edges = edges

    rows.sort(key=lambda r: (r["f1"], r["tp"], -r["fp"]), reverse=True)
    csv_path = args.out_dir / "soft_postprocess_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best_edge_path = args.out_dir / "best_soft_postprocess_edges.pt"
    torch.save(edge_tensor(best_edges), best_edge_path)
    (args.out_dir / "best_soft_postprocess.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    compare_dir = args.out_dir / "best_graph_comparison"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "water" / "compare_graphs_epanet.py"),
        "--learned",
        str(best_edge_path),
        "--npz",
        str(args.npz),
        "--meta",
        str(args.meta),
        "--outdir",
        str(compare_dir),
    ]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

    print(f"[OK] sweep rows: {len(rows)}")
    print(f"[OK] sweep CSV: {csv_path}")
    print(f"[OK] best: {best}")
    print(f"[OK] best graph: {best_edge_path}")


if __name__ == "__main__":
    main()
