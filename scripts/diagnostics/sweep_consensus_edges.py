"""Sweep static consensus graph sizes from a trained dynamic DGM model."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.run_train_synthetic import (
    build_knn_candidate_mask,
    configure_run,
    _local_undirected_graphs,
)
from src.train_dgm import _flatten_for_dgm, _reshape_ctown_epanet_batch
from src.utils.utils import get_datamodule, get_model, set_params_wrt_dataset


def count_validation_edges(model, loader, params) -> tuple[dict[tuple[int, int], int], int]:
    device = next(model.parameters()).device
    model.eval()
    edge_counts: dict[tuple[int, int], int] = {}
    n_graphs = 0

    with torch.no_grad():
        for batch in loader:
            edge_index = batch.edge_index.to(device)
            edge_weight = batch.edge_attr.to(device) if batch.edge_attr is not None else None
            x, y, batch = _reshape_ctown_epanet_batch(batch, params, device)
            x, y = _flatten_for_dgm(x, y)
            _ = model(x, edge_index, edge_weight)

            for edge_set in _local_undirected_graphs(model.edges, params.num_nodes):
                n_graphs += 1
                for edge in edge_set:
                    edge_counts[edge] = edge_counts.get(edge, 0) + 1

    return edge_counts, n_graphs


class DisjointSet:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def select_top_edges(edge_counts: dict[tuple[int, int], int], top_edges: int) -> list[tuple[int, int]]:
    return [
        edge
        for edge, _ in sorted(edge_counts.items(), key=lambda item: item[1], reverse=True)[:top_edges]
    ]


def select_connected_edges(
    edge_counts: dict[tuple[int, int], int],
    top_edges: int,
    num_nodes: int,
) -> list[tuple[int, int]]:
    ranked = sorted(edge_counts.items(), key=lambda item: item[1], reverse=True)
    dsu = DisjointSet(num_nodes)
    selected: list[tuple[int, int]] = []
    selected_set: set[tuple[int, int]] = set()

    for (u, v), _ in ranked:
        if dsu.union(u, v):
            selected.append((u, v))
            selected_set.add((u, v))
            if len(selected) >= num_nodes - 1:
                break

    for (u, v), _ in ranked:
        if len(selected) >= top_edges:
            break
        if (u, v) in selected_set:
            continue
        selected.append((u, v))
        selected_set.add((u, v))

    return selected[:top_edges]


def save_edges(selected: list[tuple[int, int]], out_path: Path) -> Path:
    rows = []
    for u, v in selected:
        rows.append((u, v))
        rows.append((v, u))
    edge_index = torch.tensor(rows, dtype=torch.long).T if rows else torch.empty((2, 0), dtype=torch.long)
    torch.save(edge_index, out_path)
    return out_path


def load_summary(summary_path: Path) -> dict:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    rec = data["edge_recovery_undirected"]
    learned = data["learned_undirected"]
    return {
        "top_edges": learned["num_edges"],
        "precision": rec["precision"],
        "recall": rec["recall"],
        "f1": rec["f1"],
        "jaccard": rec["jaccard"],
        "components": learned["num_connected_components_undirected"],
        "edge_length_mean": learned["edge_length_mean"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep consensus top-E values from a trained synthetic DGM.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--top_edges", type=int, nargs="+", default=[50, 55, 61, 65, 70, 75, 80, 90, 100])
    parser.add_argument("--connected_consensus", action="store_true")
    parser.add_argument("--data_dir", type=Path, default=PROJECT_ROOT / "data" / "synthetic" / "simple_water_n50")
    parser.add_argument("--num_nodes", type=int, default=50)
    parser.add_argument("--lags", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--lambda_graph", type=float, default=1e-2)
    parser.add_argument("--lambda_stability", type=float, default=0.03)
    parser.add_argument("--use_complementary_loss", action="store_true")
    parser.add_argument("--alpha_forecast", type=float, default=0.4)
    parser.add_argument("--lambda_soft_geo", type=float, default=1.0)
    parser.add_argument("--lambda_soft_degree", type=float, default=0.05)
    parser.add_argument("--lambda_soft_entropy", type=float, default=0.0)
    parser.add_argument("--lambda_soft_connectivity", type=float, default=0.0)
    parser.add_argument("--soft_connectivity_target", type=float, default=0.05)
    parser.add_argument("--lambda_soft_temporal", type=float, default=0.0)
    parser.add_argument("--lambda_soft_triangle", type=float, default=0.0)
    parser.add_argument("--lambda_soft_mst", type=float, default=0.0)
    parser.add_argument("--soft_degree_target", type=float, default=1.0)
    parser.add_argument("--candidate_k", type=int, default=6)
    parser.add_argument("--run_name", default="consensus_sweep")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    args.data_dir = args.data_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    params = configure_run(args)
    train_l, val_l, _, params = get_datamodule(params)
    params = set_params_wrt_dataset(params, train_l)

    model = get_model(params)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    npz = np.load(params.data_npz)
    candidate_mask = build_knn_candidate_mask(npz["node_xy"].astype(np.float32), args.candidate_k)
    for gf in model.graph_f:
        if hasattr(gf, "candidate_mask"):
            gf.candidate_mask = candidate_mask

    edge_counts, n_graphs = count_validation_edges(model, val_l, params)
    counts_csv = args.out_dir / "edge_frequencies.csv"
    with counts_csv.open("w", encoding="utf-8") as f:
        f.write("u,v,count,frequency\n")
        for (u, v), count in sorted(edge_counts.items(), key=lambda item: item[1], reverse=True):
            f.write(f"{u},{v},{count},{count / max(1, n_graphs):.8f}\n")

    rows = []
    for top_e in args.top_edges:
        run_dir = args.out_dir / f"top_{top_e}"
        run_dir.mkdir(parents=True, exist_ok=True)
        if args.connected_consensus:
            selected_edges = select_connected_edges(edge_counts, top_e, params.num_nodes)
        else:
            selected_edges = select_top_edges(edge_counts, top_e)
        learned_path = save_edges(selected_edges, run_dir / "dgm_learned_edges_consensus.pt")
        compare_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "water" / "compare_graphs_epanet.py"),
            "--learned",
            str(learned_path),
            "--npz",
            params.data_npz,
            "--meta",
            params.meta_json,
            "--outdir",
            str(run_dir / "graph_comparison"),
        ]
        subprocess.run(compare_cmd, check=True, cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL)
        row = load_summary(run_dir / "graph_comparison" / "comparison_summary.json")
        rows.append(row)

    rows = sorted(rows, key=lambda row: row["f1"], reverse=True)
    result_json = args.out_dir / "consensus_sweep_results.json"
    result_csv = args.out_dir / "consensus_sweep_results.csv"
    result_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with result_csv.open("w", encoding="utf-8") as f:
        f.write("top_edges,precision,recall,f1,jaccard,components,edge_length_mean\n")
        for row in rows:
            f.write(
                f"{row['top_edges']},{row['precision']:.8f},{row['recall']:.8f},"
                f"{row['f1']:.8f},{row['jaccard']:.8f},{row['components']},"
                f"{row['edge_length_mean']:.8f}\n"
            )

    print(f"[INFO] Counted dynamic edges over {n_graphs} validation graphs.")
    print(f"[INFO] Edge frequencies: {counts_csv}")
    print("\n=== CONSENSUS SWEEP RESULTS ===")
    for row in rows:
        print(
            f"top={row['top_edges']:>3} | F1={row['f1']:.4f} | "
            f"P={row['precision']:.4f} | R={row['recall']:.4f} | "
            f"J={row['jaccard']:.4f} | comp={row['components']}"
        )
    print(f"\n[OK] Wrote {result_csv}")
    print(f"[OK] Wrote {result_json}")


if __name__ == "__main__":
    main()
