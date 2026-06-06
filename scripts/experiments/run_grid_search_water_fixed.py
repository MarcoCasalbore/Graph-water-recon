import os
import sys
import json
import random
import itertools
from types import SimpleNamespace

import numpy as np
import torch
import pandas as pd
from lightning import seed_everything
import subprocess
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Parameters
from src.utils.utils import get_model, get_datamodule, set_params_wrt_dataset
from src.train_dgm import train_and_test_dgm


# ============================================================
# USER CONFIG
# Put single-element lists for fixed params, longer lists for grid search
# ============================================================

SEARCH_SPACE = {
    "k": [2],
    "k_geo_candidates": [3, 5],

    "lambda_graph": [0.1, 0.5],
    "lambda_geo": [0.0, 1e-4],
    "lambda_smooth": [0.0],
    "lambda_knn": [0.0],
    "lambda_degree": [0.0, 1e-3],

    "max_degree_target": [3],

    "pressure_weight": [0.3, 1.0],
    "demand_weight": [0.6, 2.0],

    "lags": [24],
    "prediction_window": [1],
    "batch_size": [32],

    "lr": [1e-3],
    "max_epochs": [15],

    "emb_dim": [64],
}
SEEDS = [42]

# Ranking metric
PRIMARY_METRIC = "f1_undirected"

# Output folder
GRID_ROOT = os.path.join(PROJECT_ROOT, "grid_runs_water")


# ============================================================
# Helpers
# ============================================================



def run_graph_comparison(project_root: str, run_dir: str, data_npz: str, meta_json: str):
    """
    Runs the comparison script after training and returns the path of the produced summary JSON.
    """
    compare_script = os.path.join(project_root, "scripts", "water", "compare_graphs_epanet.py")
    learned_pt = os.path.join(run_dir, "dgm_learned_edges.pt")

    if not os.path.exists(compare_script):
        raise FileNotFoundError(f"Missing water graph comparison script at: {compare_script}")
    if not os.path.exists(learned_pt):
        raise FileNotFoundError(f"Missing learned edges file at: {learned_pt}")

    cmd = [
        sys.executable,
        compare_script,
        "--learned", learned_pt,
        "--npz", data_npz,
        "--meta", meta_json,
        "--outdir", run_dir,
    ]

    subprocess.run(cmd, check=True)
    return os.path.join(run_dir, "comparison_summary.json")




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


def assign_candidate_mask_to_model(model, candidate_mask: torch.Tensor):
    assigned = 0
    if hasattr(model, "graph_f"):
        for gf in model.graph_f:
            if hasattr(gf, "candidate_mask"):
                gf.candidate_mask = candidate_mask
                assigned += 1
    return assigned


def make_run_name(cfg: dict, seed: int, idx: int) -> str:
    parts = [
        f"run{idx:04d}",
        f"k{cfg['k']}",
        f"kgeo{cfg['k_geo_candidates']}",
        f"lg{cfg['lambda_graph']}",
        f"ls{cfg['lambda_smooth']}",
        f"lgeo{cfg['lambda_geo']}",
        f"ldeg{cfg['lambda_degree']}",
        f"bs{cfg['batch_size']}",
        f"lags{cfg['lags']}",
        f"pw{cfg['prediction_window']}",
        f"seed{seed}",
    ]
    return "_".join(str(p) for p in parts)


def iter_search_space(search_space: dict):
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def safe_read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_graph_metrics(summary_json: dict):
    out = {}

    if summary_json is None:
        return out

    # Recovery blocks already use the correct keys in comparison_summary.json
    und = summary_json.get("edge_recovery_undirected", {})
    dire = summary_json.get("edge_recovery_directed", {})

    # Graph-stat blocks in compare_graphs_epanet.py are named:
    #   ground_truth_directed / learned_directed
    #   ground_truth_undirected / learned_undirected
    # Keep backward-compatible fallbacks in case older summaries used different names.
    learned_und = summary_json.get("learned_undirected", summary_json.get("learned_graph_undirected", {}))
    learned_dir = summary_json.get("learned_directed", summary_json.get("learned_graph_directed", {}))
    gt_und = summary_json.get("ground_truth_undirected", summary_json.get("ground_truth_graph_undirected", {}))
    gt_dir = summary_json.get("ground_truth_directed", summary_json.get("ground_truth_graph_directed", {}))

    out["tp_undirected"] = und.get("tp")
    out["fp_undirected"] = und.get("fp")
    out["fn_undirected"] = und.get("fn")
    out["precision_undirected"] = und.get("precision")
    out["recall_undirected"] = und.get("recall")
    out["f1_undirected"] = und.get("f1")
    out["jaccard_undirected"] = und.get("jaccard")

    out["tp_directed"] = dire.get("tp")
    out["fp_directed"] = dire.get("fp")
    out["fn_directed"] = dire.get("fn")
    out["precision_directed"] = dire.get("precision")
    out["recall_directed"] = dire.get("recall")
    out["f1_directed"] = dire.get("f1")
    out["jaccard_directed"] = dire.get("jaccard")

    out["learned_num_edges_undirected"] = learned_und.get("num_edges")
    out["learned_degree_mean_undirected"] = learned_und.get("degree_mean")
    out["learned_degree_max_undirected"] = learned_und.get("degree_max")
    out["learned_edge_length_mean_undirected"] = learned_und.get("edge_length_mean")

    out["learned_num_edges_directed"] = learned_dir.get("num_edges")
    out["learned_degree_mean_directed"] = learned_dir.get("degree_mean")
    out["learned_degree_max_directed"] = learned_dir.get("degree_max")
    out["learned_edge_length_mean_directed"] = learned_dir.get("edge_length_mean")

    out["gt_num_edges_undirected"] = gt_und.get("num_edges")
    out["gt_degree_mean_undirected"] = gt_und.get("degree_mean")
    out["gt_degree_max_undirected"] = gt_und.get("degree_max")
    out["gt_edge_length_mean_undirected"] = gt_und.get("edge_length_mean")

    out["gt_num_edges_directed"] = gt_dir.get("num_edges")
    out["gt_degree_mean_directed"] = gt_dir.get("degree_mean")
    out["gt_degree_max_directed"] = gt_dir.get("degree_max")
    out["gt_edge_length_mean_directed"] = gt_dir.get("edge_length_mean")

    return out


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    seed_everything(seed, workers=True)


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(GRID_ROOT, exist_ok=True)

    all_cfgs = list(iter_search_space(SEARCH_SPACE))
    print(f"Total grid configurations: {len(all_cfgs)}")
    print(f"Seeds per configuration: {SEEDS}")
    print(f"Total runs: {len(all_cfgs) * len(SEEDS)}")

    results = []

    data_npz = os.path.join(PROJECT_ROOT, "data", "ctown_ts", "data.npz")
    meta_json = os.path.join(PROJECT_ROOT, "data", "ctown_ts", "meta.json")

    if not os.path.exists(data_npz):
        raise FileNotFoundError(f"Missing data_npz: {data_npz}")
    if not os.path.exists(meta_json):
        raise FileNotFoundError(f"Missing meta_json: {meta_json}")

    node_xy_np = np.load(data_npz)["node_xy"].astype(np.float32)

    run_idx = 0
    for cfg in all_cfgs:
        for seed in SEEDS:
            run_idx += 1
            run_name = make_run_name(cfg, seed, run_idx)
            run_dir = os.path.join(GRID_ROOT, run_name)
            os.makedirs(run_dir, exist_ok=True)

            print("\n" + "=" * 100)
            print(f"[RUN {run_idx}] {run_name}")
            print("=" * 100)

            set_all_seeds(seed)

            # Build params
            args = SimpleNamespace(batch_size=cfg["batch_size"], verbose=False)
            run_params = Parameters(args)

            # Fixed water/EPANET setup
            run_params.dataset_name = "ctown_epanet"
            run_params.dataset_path = ""
            run_params.data_npz = data_npz
            run_params.meta_json = meta_json

            run_params.num_nodes = 396
            run_params.input_features = 2
            run_params.output_features = 2

            # Search-space overrides
            for key, val in cfg.items():
                setattr(run_params, key, val)

            run_params.seed = seed
            run_params.export_dir = run_dir
            run_params.loss_plot_path = os.path.join(run_dir, "loss_curve.png")
            run_params.model = "dgm_gcn"
            run_params.logging = False

            # Data
            train_l, val_l, test_l, run_params = get_datamodule(run_params)
            run_params = set_params_wrt_dataset(run_params, train_l)

            # Model
            model = get_model(run_params)

            # Hard geometric mask
            candidate_mask = build_knn_candidate_mask(
                node_xy=node_xy_np,
                k=run_params.k_geo_candidates
            )
            assigned = assign_candidate_mask_to_model(model, candidate_mask)

            print(f"[INFO] Assigned hard kNN candidate mask to {assigned} graph layer(s).")
            print(f"[INFO] Candidate mask shape: {tuple(candidate_mask.shape)}")
            print(f"[INFO] Allowed undirected edges in mask: {int(candidate_mask.sum().item() // 2)}")

            # Train/test
            out = train_and_test_dgm(model, train_l, val_l, test_l, run_params)

            # Run graph comparison against GT and read the resulting summary
            try:
                summary_path = run_graph_comparison(
                    project_root=PROJECT_ROOT,
                    run_dir=run_dir,
                    data_npz=data_npz,
                    meta_json=meta_json,
                )
                summary_json = safe_read_json(summary_path)
                graph_metrics = extract_graph_metrics(summary_json)
            except Exception as e:
                print(f"[WARN] Graph comparison failed for run {run_name}: {e}")
                summary_json = None
                graph_metrics = {}

            row = {
                "run_idx": run_idx,
                "run_name": run_name,
                "seed": seed,
                **cfg,
                "train_mse": out["train"]["mse"] if out["train"] else None,
                "train_rmse": out["train"]["rmse"] if out["train"] else None,
                "train_mae": out["train"]["mae"] if out["train"] else None,
                "train_p_mse": out["train"].get("pressure_mse") if out["train"] else None,
                "train_d_mse": out["train"].get("demand_mse") if out["train"] else None,
                "val_mse": out["val"]["mse"] if out["val"] else None,
                "val_rmse": out["val"]["rmse"] if out["val"] else None,
                "val_mae": out["val"]["mae"] if out["val"] else None,
                "val_p_mse": out["val"].get("pressure_mse") if out["val"] else None,
                "val_d_mse": out["val"].get("demand_mse") if out["val"] else None,
                "test_mse": out["test"]["mse"] if out["test"] else None,
                "test_rmse": out["test"]["rmse"] if out["test"] else None,
                "test_mae": out["test"]["mae"] if out["test"] else None,
                "test_p_mse": out["test"].get("pressure_mse") if out["test"] else None,
                "test_d_mse": out["test"].get("demand_mse") if out["test"] else None,
                **graph_metrics,
            }

            results.append(row)

            # Save rolling CSV after every run
            df = pd.DataFrame(results)
            df.to_csv(os.path.join(GRID_ROOT, "grid_results_water.csv"), index=False)

            # Print short summary
            if PRIMARY_METRIC in row and row[PRIMARY_METRIC] is not None:
                print(
                    f"[SUMMARY] {PRIMARY_METRIC}={row[PRIMARY_METRIC]:.4f} | "
                    f"prec_u={row.get('precision_undirected', np.nan):.4f} | "
                    f"rec_u={row.get('recall_undirected', np.nan):.4f} | "
                    f"test_mse={row.get('test_mse', np.nan):.4f}"
                )

    df = pd.DataFrame(results)

    # Sort by primary metric descending, then by test_mse ascending
    if PRIMARY_METRIC in df.columns:
        df = df.sort_values(
            by=[PRIMARY_METRIC, "test_mse"],
            ascending=[False, True],
            na_position="last"
        )

    final_csv = os.path.join(GRID_ROOT, "grid_results_water_sorted.csv")
    df.to_csv(final_csv, index=False)

    print("\n" + "=" * 100)
    print("GRID SEARCH COMPLETED")
    print(f"Saved results to: {final_csv}")
    if len(df) > 0:
        print("\nTop 10 runs:")
        show_cols = [
            "run_name",
            PRIMARY_METRIC,
            "precision_undirected",
            "recall_undirected",
            "jaccard_undirected",
            "learned_num_edges_undirected",
            "learned_edge_length_mean_undirected",
            "test_mse",
            "test_p_mse",
            "test_d_mse",
        ]
        show_cols = [c for c in show_cols if c in df.columns]
        print(df[show_cols].head(10).to_string(index=False))
    print("=" * 100)


if __name__ == "__main__":
    main()
