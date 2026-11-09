"""Export aligned test predictions from existing node-only and fixed-graph checkpoints."""
from pathlib import Path
import json
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results_section_v"
CAMPAIGN = RESULTS / "campaign_v2/relational"
DATA = RESULTS / "campaign_v2/datasets/homogeneous_relational_g211_s1211"
SOURCE = CAMPAIGN / "simple_topological_baselines/runs"
OUT = PROJECT / "results_section_v_final/error_analysis/predictions"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts/validation"))

from scripts.experiments.train_forecast_baseline import NodeMLP, make_windows  # noqa: E402
from src.config import Parameters  # noqa: E402
from src.dataset.dataset_epanet import EpanetDataModule  # noqa: E402
from src.train_dgm import _flatten_for_dgm, _reshape_ctown_epanet_batch  # noqa: E402
from run_simple_topology_baselines import FixedDGMForecastPath, batch_edges, directed, edge_set  # noqa: E402

METHODS = [
    "euclidean_mst", "geometric_knn3_mutual", "geometric_knn3_union",
    "correlation_abs_knn3_union", "distance_correlation_hybrid_knn3_union",
    "proposed_final_decoded_graph", "oracle_reference_graph",
    "random_final_decoded_edge_count_control",
]


def fixed_loader():
    class Cli: batch_size = 32; verbose = False
    params = Parameters(Cli())
    params.dataset_name = "synthetic_water"
    params.data_npz = str(DATA / "data.npz"); params.meta_json = str(DATA / "meta.json")
    params.num_nodes = 50; params.lags = 6; params.prediction_window = 12
    params.input_features = 2; params.output_features = 2
    dm = EpanetDataModule(params)
    return params, dm.test_dataloader()


@torch.no_grad()
def fixed_predictions(method, seed, params, loader):
    run = SOURCE / f"{method}_seed{seed}_l6_h12_e300"
    model = FixedDGMForecastPath(12, 24)
    model.load_state_dict(torch.load(run / "best_forecaster.pt", map_location="cpu"))
    model.eval()
    local = directed(edge_set(np.loadtxt(run / "edges.csv", delimiter=",", skiprows=1, dtype=int).T))
    truth, prediction = [], []
    for batch in loader:
        x, y, _ = _reshape_ctown_epanet_batch(batch, params, torch.device("cpu"))
        x, y = _flatten_for_dgm(x, y)
        bsz = x.shape[0] // 50
        pred = model(x, batch_edges(local, bsz, 50, torch.device("cpu")))
        shape = (bsz, 50, 12, 2)
        truth.append(y.reshape(shape).numpy()); prediction.append(pred.reshape(shape).numpy())
    return np.concatenate(truth), np.concatenate(prediction)


def node_predictions(seed):
    archive = np.load(DATA / "data.npz")
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    raw = np.stack([archive["pressure"], archive["demand"]], axis=-1).astype(np.float32)
    train = meta["splits"]["train"]
    chunk = raw[train["start"]:train["end"]]
    mean = chunk.mean(axis=(0,1), keepdims=True).astype(np.float32)
    std = (chunk.std(axis=(0,1), keepdims=True)+1e-8).astype(np.float32)
    series = (raw-mean)/std
    split = meta["splits"]["test"]
    x, y = make_windows(series, split["start"], split["end"], 6, 12)
    model = NodeMLP(6, 2, 64, 12)
    run = CAMPAIGN / "node_only" / f"node_g211_s1211_t{seed}_l6_h12_e300"
    model.load_state_dict(torch.load(run / "node_mlp_controlled.pt", map_location="cpu"))
    model.eval(); parts=[]
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(torch.tensor(x)), batch_size=32):
            parts.append(model(xb).numpy())
    return y.transpose(0, 2, 1, 3), np.concatenate(parts).transpose(0, 2, 1, 3)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    params, loader = fixed_loader()
    reference_truth = None
    for seed in [42,43,44,45,46]:
        truth, pred = node_predictions(seed)
        np.savez_compressed(OUT/f"node_only_seed{seed}.npz", y_true=truth, y_pred=pred,
                            method="node_only", training_seed=seed, domain="train_standardized")
        reference_truth = truth if reference_truth is None else reference_truth
        if not np.allclose(reference_truth, truth): raise RuntimeError("unaligned node-only test targets")
        for method in METHODS:
            truth, pred = fixed_predictions(method, seed, params, loader)
            if not np.allclose(reference_truth, truth): raise RuntimeError(f"unaligned targets: {method} seed={seed}")
            np.savez_compressed(OUT/f"{method}_seed{seed}.npz", y_true=truth, y_pred=pred,
                                method=method, training_seed=seed, domain="train_standardized")
            print(f"[OK] {method} seed={seed}", flush=True)
    print(f"[DONE] {len(list(OUT.glob('*.npz')))} prediction archives")


if __name__ == "__main__": main()
