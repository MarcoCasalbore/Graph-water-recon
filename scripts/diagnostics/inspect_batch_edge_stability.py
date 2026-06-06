import os
import sys
import json
from collections import Counter, defaultdict

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Parameters
from src.utils.utils import get_model, get_datamodule, set_params_wrt_dataset
from src.train_dgm import _reshape_ctown_epanet_batch, _flatten_for_dgm


def build_knn_candidate_mask(node_xy: np.ndarray, k: int) -> torch.Tensor:
    xy = torch.tensor(node_xy, dtype=torch.float32)
    n = xy.shape[0]

    dist = torch.cdist(xy, xy, p=2)
    eye = torch.eye(n, dtype=torch.bool)
    dist = dist.masked_fill(eye, float("inf"))

    k = min(int(k), n - 1)
    nn_idx = torch.topk(dist, k=k, largest=False, dim=1).indices

    mask = torch.zeros((n, n), dtype=torch.bool)
    rows = torch.arange(n).unsqueeze(1).expand_as(nn_idx)
    mask[rows, nn_idx] = True

    mask = mask | mask.T
    mask.fill_diagonal_(False)
    return mask


def canonical_undirected(u: int, v: int):
    return (u, v) if u <= v else (v, u)


def extract_local_graphs_from_edges(edges: torch.Tensor, num_nodes: int):
    """
    edges: [2, E] with batch-global indexing
    returns:
      list_of_edge_sets, one undirected edge set per logical batch sample
    """
    edges = edges.detach().cpu().numpy().astype(np.int64)
    src = edges[0]
    dst = edges[1]

    batch_ids = src // num_nodes
    B = int(batch_ids.max()) + 1 if src.size > 0 else 0

    out = []
    for b in range(B):
        mask = (src // num_nodes == b)
        src_b = src[mask] - b * num_nodes
        dst_b = dst[mask] - b * num_nodes

        edge_set = set()
        for u, v in zip(src_b.tolist(), dst_b.tolist()):
            if u == v:
                continue
            if not (0 <= u < num_nodes and 0 <= v < num_nodes):
                continue
            edge_set.add(canonical_undirected(int(u), int(v)))
        out.append(edge_set)

    return out


def summarize_edge_stability(edge_sets, num_nodes, out_json):
    n_graphs = len(edge_sets)

    union_edges = set()
    inter_edges = None
    freq = Counter()
    per_node_neighbors = defaultdict(set)

    for es in edge_sets:
        union_edges |= es
        if inter_edges is None:
            inter_edges = set(es)
        else:
            inter_edges &= es

        for e in es:
            freq[e] += 1
            u, v = e
            per_node_neighbors[u].add(v)
            per_node_neighbors[v].add(u)

    if inter_edges is None:
        inter_edges = set()

    def consensus_edges(thr):
        need = int(np.ceil(thr * n_graphs))
        return {e for e, c in freq.items() if c >= need}

    cons_50 = consensus_edges(0.50)
    cons_80 = consensus_edges(0.80)
    cons_100 = consensus_edges(1.00)

    neighbor_variety = [len(per_node_neighbors[i]) for i in range(num_nodes)]

    freq_values = np.array(list(freq.values()), dtype=float) if len(freq) > 0 else np.array([], dtype=float)

    summary = {
        "n_batch_graphs": n_graphs,
        "union_num_edges": len(union_edges),
        "intersection_num_edges": len(inter_edges),
        "consensus_50_num_edges": len(cons_50),
        "consensus_80_num_edges": len(cons_80),
        "consensus_100_num_edges": len(cons_100),
        "edge_frequency_mean": float(freq_values.mean()) if freq_values.size else 0.0,
        "edge_frequency_median": float(np.median(freq_values)) if freq_values.size else 0.0,
        "edge_frequency_min": float(freq_values.min()) if freq_values.size else 0.0,
        "edge_frequency_max": float(freq_values.max()) if freq_values.size else 0.0,
        "neighbor_variety_mean": float(np.mean(neighbor_variety)),
        "neighbor_variety_median": float(np.median(neighbor_variety)),
        "neighbor_variety_min": int(np.min(neighbor_variety)),
        "neighbor_variety_max": int(np.max(neighbor_variety)),
        "top_20_most_frequent_edges": [
            {"u": int(u), "v": int(v), "count": int(c)}
            for (u, v), c in freq.most_common(20)
        ],
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    # Match the run configuration you want to inspect.
    class Args:
        batch_size = 32
        verbose = False

    args = Args()
    run_params = Parameters(args)

    run_params.dataset_name = "ctown_epanet"
    run_params.dataset_path = ""
    run_params.data_npz = os.path.join(PROJECT_ROOT, "data", "ctown_ts", "data.npz")
    run_params.meta_json = os.path.join(PROJECT_ROOT, "data", "ctown_ts", "meta.json")

    run_params.num_nodes = 396
    run_params.lags = 24
    run_params.prediction_window = 1
    run_params.batch_size = 32

    run_params.input_features = 2
    run_params.output_features = 2

    run_params.k = 2
    run_params.k_geo_candidates = 3
    run_params.lambda_geo = 1e-4
    run_params.lambda_smooth = 0.0
    run_params.lambda_graph = 1e-2
    run_params.lambda_knn = 0.0
    run_params.lambda_degree = 0.0
    run_params.max_degree_target = 3
    run_params.pressure_weight = 0.3
    run_params.demand_weight = 2.0

    ckpt_path = os.path.join(PROJECT_ROOT, "models", "dgm_ctown_epanet_best.pt")
    split_to_use = "val"   # "train", "val", or "test"
    max_batches = 20       # inspect first 20 batches
    out_json = os.path.join(PROJECT_ROOT, "batch_edge_stability_summary.json")

    train_l, val_l, test_l, run_params = get_datamodule(run_params)
    run_params = set_params_wrt_dataset(run_params, train_l)

    if split_to_use == "train":
        loader = train_l
    elif split_to_use == "val":
        loader = val_l
    else:
        loader = test_l

    model = get_model(run_params)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    npz = np.load(run_params.data_npz)
    node_xy_np = npz["node_xy"].astype(np.float32)
    candidate_mask = build_knn_candidate_mask(node_xy_np, run_params.k_geo_candidates)

    assigned = 0
    for gf in model.graph_f:
        if hasattr(gf, "candidate_mask"):
            gf.candidate_mask = candidate_mask
            assigned += 1

    print(f"[INFO] Assigned mask to {assigned} graph layers.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_batch_graphs = []

    with torch.no_grad():
        for bidx, batch in enumerate(loader):
            if bidx >= max_batches:
                break

            edge_index = batch.edge_index.to(device)
            edge_weight = batch.edge_attr.to(device) if batch.edge_attr is not None else None

            x, y, batch = _reshape_ctown_epanet_batch(batch, run_params, device)
            x, y = _flatten_for_dgm(x, y)

            _ = model(x, edge_index, edge_weight)

            if not hasattr(model, "edges") or model.edges is None:
                raise RuntimeError("model.edges is missing after forward pass")

            local_graphs = extract_local_graphs_from_edges(model.edges, run_params.num_nodes)
            all_batch_graphs.extend(local_graphs)

            print(f"[BATCH {bidx}] logical_graphs={len(local_graphs)}")

    summary = summarize_edge_stability(
        edge_sets=all_batch_graphs,
        num_nodes=run_params.num_nodes,
        out_json=out_json,
    )

    print("\n=== BATCH EDGE STABILITY SUMMARY ===")
    for k, v in summary.items():
        if k == "top_20_most_frequent_edges":
            continue
        print(f"{k}: {v}")

    print(f"\n[OK] Saved summary to: {out_json}")


if __name__ == "__main__":
    main()
