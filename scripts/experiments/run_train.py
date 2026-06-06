"""Train the current DGM configuration on the C-Town EPANET dataset."""

import argparse
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0+PTX"

from src.config import Parameters
from src.utils.utils import get_model, set_params_wrt_dataset, get_datamodule
from src.utils.utils import train_and_test


def build_knn_candidate_mask(node_xy: np.ndarray, k: int) -> torch.Tensor:
    """
    Build an undirected kNN mask from node coordinates.

    Returns:
        mask: torch.BoolTensor of shape [N, N]
              True where an edge is allowed.
    """
    xy = torch.tensor(node_xy, dtype=torch.float32)
    N = xy.shape[0]

    dist = torch.cdist(xy, xy, p=2)

    eye = torch.eye(N, dtype=torch.bool)
    dist = dist.masked_fill(eye, float("inf"))

    k = min(int(k), N - 1)
    nn_idx = torch.topk(dist, k=k, largest=False, dim=1).indices

    mask = torch.zeros((N, N), dtype=torch.bool)
    rows = torch.arange(N).unsqueeze(1).expand_as(nn_idx)
    mask[rows, nn_idx] = True

    mask = mask | mask.T
    mask.fill_diagonal_(False)
    return mask


parser = argparse.ArgumentParser(description="Train DGM on the configured dataset.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size!")
parser.add_argument("--verbose", type=bool, default=True, help="Verbose output!")
args = parser.parse_args()

# Parameters
run_params = Parameters(args)

# ------------------------------------------------------------------
# EPANET / C-Town configuration
# ------------------------------------------------------------------
run_params.dataset_name = "ctown_epanet"
run_params.dataset_path = ""
run_params.data_npz = os.path.join(PROJECT_ROOT, "data", "ctown_ts", "data.npz")
run_params.meta_json = os.path.join(PROJECT_ROOT, "data", "ctown_ts", "meta.json")

run_params.graph_detach_input = False

run_params.k = 2

run_params.num_nodes = 396
run_params.lags = 24
run_params.prediction_window = 1
run_params.batch_size = args.batch_size

run_params.input_features = 2
run_params.output_features = 2

run_params.lambda_geo = 1e-4
run_params.lambda_smooth = 0
run_params.lambda_graph = 1e-2

run_params.lambda_knn = 0
run_params.k_geo_candidates = 3

run_params.lambda_degree = 0
run_params.max_degree_target = 3

run_params.pressure_weight = 0.3
run_params.demand_weight = 2.0

print("Resolved data_npz :", run_params.data_npz)
print("Resolved meta_json:", run_params.meta_json)
print("data_npz exists   :", os.path.exists(run_params.data_npz))
print("meta_json exists  :", os.path.exists(run_params.meta_json))

# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
train_l, val_l, test_l, run_params = get_datamodule(run_params)

# Modify parameters for dataset
run_params = set_params_wrt_dataset(run_params, train_l)

# ------------------------------------------------------------------
# Debug prints
# ------------------------------------------------------------------
print("========== RUN CONFIG ==========")
print(f"dataset_name      : {run_params.dataset_name}")
print(f"data_npz          : {run_params.data_npz}")
print(f"meta_json         : {run_params.meta_json}")
print(f"num_nodes         : {run_params.num_nodes}")
print(f"lags              : {run_params.lags}")
print(f"prediction_window : {run_params.prediction_window}")
print(f"input_features    : {run_params.input_features}")
print(f"output_features   : {run_params.output_features}")
print(f"k                 : {run_params.k}")
print(f"lambda_geo        : {run_params.lambda_geo}")
print(f"lambda_smooth     : {run_params.lambda_smooth}")
print(f"lambda_graph      : {run_params.lambda_graph}")
print(f"lambda_knn        : {run_params.lambda_knn}")
print(f"k_geo_candidates  : {run_params.k_geo_candidates}")
print(f"lambda_degree     : {run_params.lambda_degree}")
print(f"max_degree_target : {run_params.max_degree_target}")
print(f"pressure_weight   : {run_params.pressure_weight}")
print(f"demand_weight     : {run_params.demand_weight}")
print(f"pre_fc[0]         : {run_params.pre_fc[0] if run_params.pre_fc is not None else None}")
print(f"fc_layers[-1]     : {run_params.fc_layers[-1]}")
print("================================")

first_batch = next(iter(train_l))
print("First batch x shape        :", first_batch.x.shape)
print("First batch y shape        :", first_batch.y.shape)
print("First batch edge_index shape:", first_batch.edge_index.shape)
print("First batch node_xy shape  :", first_batch.node_xy.shape)

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
model = get_model(run_params)


# ----------------------------------------------------------
# Hard geometric candidate mask for DGM sampling
# ----------------------------------------------------------
npz = np.load(run_params.data_npz)
node_xy_np = npz["node_xy"].astype(np.float32)

candidate_mask = build_knn_candidate_mask(
    node_xy=node_xy_np,
    k=run_params.k_geo_candidates
)

assigned_masks = 0
for gf in model.graph_f:
    if hasattr(gf, "candidate_mask"):
        gf.candidate_mask = candidate_mask
        assigned_masks += 1

print(f"[INFO] Assigned hard kNN candidate mask to {assigned_masks} graph layer(s).")
print(f"[INFO] Candidate mask shape: {tuple(candidate_mask.shape)}")
print(f"[INFO] Allowed undirected edges in mask: {int(candidate_mask.sum().item() // 2)}")

# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
train_and_test(model, train_l, val_l, test_l, run_params)
