"""Export representative observed/future forecasts for node-only and joint models."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results_section_v"
ROOT = RESULTS_ROOT / "campaign_v2/relational"
DATA_DIR = RESULTS_ROOT / "campaign_v2/datasets/homogeneous_relational_g211_s1211"
sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(RESULTS_ROOT))

from scripts.plot_style import COLORS, apply_style, save_figure
from scripts.experiments.run_train_synthetic import build_q_dynamic_candidate_mask, configure_run
from scripts.experiments.train_forecast_baseline import NodeMLP
from src.train_dgm import _flatten_for_dgm, _reshape_ctown_epanet_batch
from src.utils.utils import get_datamodule, get_model, set_params_wrt_dataset


def args_for(run_id, out_dir):
    return SimpleNamespace(
        data_dir=DATA_DIR, out_dir=out_dir, run_name=run_id, ffun="mlp", gfun="gcn", verbose=False,
        num_nodes=50, lags=6, stride=1, prediction_window=12, batch_size=32,
        max_epochs=300, check_val_every_n_epoch=5, lr=1e-3, k=2,
        lambda_graph=0.0, lambda_stability=0.0, use_complementary_loss=True,
        alpha_forecast=.8, alpha_forecast_start=None, alpha_forecast_end=None,
        alpha_schedule_epochs=0, alpha_hold_epochs=0, coordinate_embedding_dim=0,
        coordinate_embedding_trainable=False, lambda_soft_pseudo_edge=.5,
        pseudo_geo_tau=1.0, pseudo_temporal_weight=.5, pseudo_score_mode="observable_expert_full",
        pseudo_use_static_target=True, pseudo_max_lag=6, pseudo_partial_ridge=.05,
        pseudo_bridge_topk=1, pseudo_bridge_mass=.15, pseudo_bridge_quantile=.65,
        pseudo_bridge_temperature=.08, pseudo_bridge_min_temporal=.55,
        pseudo_bridge_candidate_k=4, pseudo_candidate_k=6, pseudo_topk=2,
        pseudo_include_mst=True, pseudo_mst_bonus=1.0, select_best_by="observable_val",
        static_top_edges=0, proxy_length_candidate_k=12, proxy_length_quantile=.65,
        proxy_length_weight=1.0, proxy_connectivity_weight=1.0, proxy_frequency_weight=.05,
        soft_degree_target=1.0, task_loss_type="huber", task_huber_delta=1.0,
        task_target="all", task_feature_combine="weighted_mean",
        task_feature_normalization="target_std", pressure_weight=1.0, demand_weight=1.0,
    )


def main():
    main_runs = pd.read_csv(ROOT / "raw_metrics/main_runs.csv")
    record = main_runs.loc[main_runs.training_seed == 42].iloc[0]
    run_dir = PROJECT_ROOT / record.output_dir
    run_id = record.run_id
    params = configure_run(args_for(run_id, run_dir))
    train_loader, val_loader, test_loader, params = get_datamodule(params)
    params = set_params_wrt_dataset(params, train_loader)
    joint = get_model(params)
    mask = build_q_dynamic_candidate_mask(str(DATA_DIR / "data.npz"), "observable_expert_full", 3, .05, 1.0, True)
    for layer in joint.graph_f:
        if hasattr(layer, "candidate_mask"):
            layer.candidate_mask = mask
    ckpt = PROJECT_ROOT / "models" / f"dgm_synthetic_water_{run_id}.pt"
    joint.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)); joint.eval()

    node = NodeMLP(6, 2, 64, 12)
    node_dir = next((ROOT / "node_only").glob("node_g211_s1211_t42_l6_h12_e120"))
    node.load_state_dict(torch.load(node_dir / "node_mlp_controlled.pt", map_location="cpu", weights_only=True)); node.eval()
    all_true, all_joint, all_node, all_history = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            edge_index = batch.edge_index
            x, y, batch = _reshape_ctown_epanet_batch(batch, params, torch.device("cpu"))
            all_history.append(x.numpy())
            all_true.append(y.numpy())
            node_pred = node(x).numpy(); all_node.append(node_pred)
            flat_x, _ = _flatten_for_dgm(x, y)
            pred, _ = joint(flat_x, edge_index, None)
            all_joint.append(pred.reshape(x.shape[0], 50, 12, 2).permute(0, 2, 1, 3).numpy())
    true = np.concatenate(all_true); joint_pred = np.concatenate(all_joint)
    node_pred = np.concatenate(all_node); history = np.concatenate(all_history)
    mean, std = test_loader.dataset.mean, test_loader.dataset.std
    true_raw, joint_raw, node_raw = true * std + mean, joint_pred * std + mean, node_pred * std + mean
    history_raw = history * std + mean
    joint_err = ((joint_pred - true) ** 2).mean(axis=(0, 1, 3))
    node_err = ((node_pred - true) ** 2).mean(axis=(0, 1, 3))
    improvement = node_err - joint_err
    data = np.load(DATA_DIR / "data.npz"); edges = {(min(u,v),max(u,v)) for u,v in data["edge_index"].T if u!=v}
    graph = nx.Graph(); graph.add_nodes_from(range(50)); graph.add_edges_from(edges)
    central = max(nx.closeness_centrality(graph), key=nx.closeness_centrality(graph).get)
    peripheral = min(graph.degree, key=lambda item: (item[1], -np.linalg.norm(data["node_xy"][item[0]]-.5)))[0]
    best = int(np.argmax(improvement)); reduced = int(np.argmin(np.abs(improvement)))
    nodes = list(dict.fromkeys([central, peripheral, best, reduced]))
    while len(nodes) < 4:
        nodes.append(next(i for i in np.argsort(-improvement) if i not in nodes))
    roles = ["central", "peripheral", "largest graph gain", "similar performance"]
    sample = 0; feature = 1
    rows=[]; apply_style(); fig,axes=plt.subplots(2,2,figsize=(7.0,4.4),sharex=True)
    for ax,node_idx,role in zip(axes.flat,nodes,roles):
        hist = history_raw[sample,:,node_idx,feature]; gt=true_raw[sample,:,node_idx,feature]
        jp=joint_raw[sample,:,node_idx,feature]; npred=node_raw[sample,:,node_idx,feature]
        xh=np.arange(-len(hist)+1,1); xf=np.arange(1,len(gt)+1)
        ax.plot(xh,hist,color="#666",label="Observed history"); ax.plot(xf,gt,color="#111",label="Ground truth")
        ax.plot(xf,npred,color=COLORS["candidate"],ls="--",label="Node-only")
        ax.plot(xf,jp,color=COLORS["learned"],label="Graph model")
        ax.axvline(.5,color="#999",lw=.8); ax.set_ylabel(f"Node {node_idx} demand\n({role})")
        rows.extend({"node":node_idx,"role":role,"relative_time":int(t),"series":"observed","value":float(v)} for t,v in zip(xh,hist))
        for name,values in [("ground_truth",gt),("node_only",npred),("graph_model",jp)]:
            rows.extend({"node":node_idx,"role":role,"relative_time":int(t),"series":name,"value":float(v)} for t,v in zip(xf,values))
    for ax in axes[-1]: ax.set_xlabel("Relative time [h]")
    handles,labels=axes[0,0].get_legend_handles_labels(); fig.legend(handles,labels,loc="upper center",ncol=4,frameon=False)
    fig.subplots_adjust(top=.86); save_figure(fig,ROOT/"figures","signal_forecast_examples"); plt.close(fig)
    pd.DataFrame(rows).to_csv(ROOT/"figures/data/signal_forecast_examples.csv",index=False)
    (ROOT/"figures/data/signal_node_selection.json").write_text(json.dumps({"nodes":nodes,"roles":roles,
        "node_mse":node_err[nodes].tolist(),"graph_mse":joint_err[nodes].tolist()},indent=2),encoding="utf-8")
    print(f"[OK] nodes={nodes} checkpoint={ckpt}")


if __name__ == "__main__":
    main()
