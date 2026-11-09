"""Reduced 3 graph-realization x 3 training-seed validation with resume."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results_section_v"
FINAL = PROJECT / "results_section_v_final"
ROOT = FINAL / "multiple_graph_realizations"
DATA_ROOT = ROOT / "datasets"
RUN_ROOT = ROOT / "runs"
sys.path.insert(0, str(PROJECT / "scripts/validation"))
sys.path.insert(0, str(PROJECT))

import run_campaign_v2 as campaign  # noqa: E402
from run_simple_topology_baselines import (  # noqa: E402
    distance_matrix, edge_set, mst_edges, rank_similarity_from_distance,
    topk_edges, topology_metrics, train_one,
)
from src.config import Parameters  # noqa: E402
from src.dataset.dataset_epanet import EpanetDataModule  # noqa: E402

GRAPH_SIGNALS = {101: 1101, 211: 1211, 307: 1307}
TRAIN_SEEDS = [42, 43, 44]
METHODS_FIXED = ["euclidean_mst", "geometric_knn3_mutual", "geometric_knn3_union",
                 "proposed_learned_graph_frozen"]


def dataset_path(graph_seed):
    signal_seed = GRAPH_SIGNALS[graph_seed]
    if graph_seed == 211:
        return RESULTS / "campaign_v2/datasets/homogeneous_relational_g211_s1211"
    return DATA_ROOT / f"homogeneous_relational_g{graph_seed}_s{signal_seed}"


def graph_stats(graph_seed, signal_seed, data):
    archive = np.load(data / "data.npz")
    edges = edge_set(archive["edge_index"])
    graph = nx.Graph(); graph.add_nodes_from(range(50)); graph.add_edges_from(edges)
    degree = np.array([d for _,d in graph.degree()])
    communities = list(nx.community.greedy_modularity_communities(graph))
    return {"graph_seed": graph_seed, "signal_seed": signal_seed, "dataset_id": data.name,
            "nodes": 50, "edges": len(edges), "density": nx.density(graph),
            "mean_degree": degree.mean(), "min_degree": degree.min(), "max_degree": degree.max(),
            "components": nx.number_connected_components(graph), "bridge_count": len(list(nx.bridges(graph))),
            "clustering": nx.average_clustering(graph), "average_shortest_path": nx.average_shortest_path_length(graph),
            "largest_greedy_community_fraction": max(map(len,communities))/50,
            "community_labels_imposed": False, "signals_finite": bool(all(np.isfinite(archive[k]).all() for k in ["pressure","demand"]))}


def run_joint(graph_seed, signal_seed, training_seed, data):
    if graph_seed == 211:
        out = RESULTS / "campaign_v2/relational/main_experiment" / (
            f"main_experiment_g211_s1211_t{training_seed}_l6_h12_a0p80_kd2_kc3_kq2_e300")
    else:
        out = RUN_ROOT / f"g{graph_seed}_s{signal_seed}" / f"joint_t{training_seed}_e300"
        if not (out/"forecast_metrics.json").exists():
            out.mkdir(parents=True, exist_ok=True)
            cmd = campaign.joint_command(out, out.name, 6, 12, .8, 2, 3, training_seed, 300)
            cmd[cmd.index("--data_dir")+1] = str(data)
            with (out/"training.log").open("w",encoding="utf-8") as log:
                subprocess.run(cmd,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
    forecast=json.loads((out/"forecast_metrics.json").read_text(encoding="utf-8"))
    test=forecast["test"]
    learned=edge_set(np.load(out/"graph_comparison_softprob/learned_edge_index_undirected.npy"))
    reference=edge_set(np.load(data/"data.npz")["edge_index"])
    topo=topology_metrics(reference,learned)
    prediction_path = (FINAL/"error_analysis/joint_retrained"/
        f"joint_g211_s1211_t{training_seed}_l6_h12_e300/test_predictions.npz") if graph_seed==211 else out/"test_predictions.npz"
    predictions=np.load(prediction_path)
    error=predictions["y_pred"].astype(float)-predictions["y_true"].astype(float)
    mse=float(np.mean(error*error)); mae=float(np.mean(np.abs(error)))
    return {"method":"proposed_joint","graph_seed":graph_seed,"signal_seed":signal_seed,
            "training_seed":training_seed,"mse":mse,"mae":mae,"rmse":np.sqrt(mse),
            **topo,"output_dir":str(out.relative_to(PROJECT))}, learned


def run_node(graph_seed, signal_seed, training_seed, data):
    if graph_seed==211:
        out=RESULTS/"campaign_v2/relational/node_only"/f"node_g211_s1211_t{training_seed}_l6_h12_e300"
    else:
        out=RUN_ROOT/f"g{graph_seed}_s{signal_seed}"/f"node_only_t{training_seed}_e300"
        if not (out/"forecast_baseline_metrics.json").exists():
            out.mkdir(parents=True,exist_ok=True)
            cmd=[sys.executable,"scripts/validation/train_node_only_campaign.py","--data_dir",str(data),
                 "--out_dir",str(out),"--lags","6","--prediction_window","12","--epochs","300",
                 "--batch_size","32","--hidden","64","--lr","0.001","--loss_type","huber",
                 "--feature_normalization","target_std","--seed",str(training_seed)]
            with (out/"training.log").open("w",encoding="utf-8") as log:
                subprocess.run(cmd,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
    metrics=json.loads((out/"forecast_baseline_metrics.json").read_text(encoding="utf-8"))["test"]["normalized"]
    return {"method":"node_only","graph_seed":graph_seed,"signal_seed":signal_seed,"training_seed":training_seed,
            "mse":metrics["mse"],"mae":metrics["mae"],"rmse":metrics["rmse"],"precision":0.0,
            "recall":0.0,"f1":0.0,"jaccard":0.0,"tp":0,"fp":0,"fn":79,"edge_count":0,
            "components":50,"output_dir":str(out.relative_to(PROJECT))}


def fixed_context(data):
    class Cli: batch_size=32; verbose=False
    params=Parameters(Cli()); params.dataset_name="synthetic_water"
    params.data_npz=str(data/"data.npz"); params.meta_json=str(data/"meta.json")
    params.num_nodes=50; params.lags=6; params.prediction_window=12
    params.input_features=2; params.output_features=2
    params.task_loss_type="huber"; params.task_huber_delta=1.0
    params.task_feature_normalization="target_std"; params.task_feature_combine="weighted_mean"
    params.task_target="all"; params.pressure_weight=1.0; params.demand_weight=1.0
    dm=EpanetDataModule(params)
    return params,(dm.train_dataloader(),dm.val_dataloader(),dm.test_dataloader())


def run_fixed(graph_seed,signal_seed,training_seed,data,learned):
    archive=np.load(data/"data.npz"); xy=archive["node_xy"].astype(float)
    geo=rank_similarity_from_distance(distance_matrix(xy)); reference=edge_set(archive["edge_index"])
    variants={"euclidean_mst":mst_edges(xy),"geometric_knn3_mutual":topk_edges(geo,3,"mutual"),
              "geometric_knn3_union":topk_edges(geo,3,"union"),"proposed_learned_graph_frozen":learned}
    if graph_seed==211:
        source=pd.read_csv(RESULTS/"campaign_v2/relational/simple_topological_baselines/fixed_graph_forecasting_runs.csv")
    else:
        params,loaders=fixed_context(data)
    rows=[]
    for method,edges in variants.items():
        if graph_seed==211:
            source_method="proposed_final_decoded_graph" if method=="proposed_learned_graph_frozen" else method
            item=source[(source.method==source_method)&(source.training_seed==training_seed)].iloc[0]
            metric={"test_mse":item.test_mse,"test_mae":item.test_mae,"test_rmse":item.test_rmse}
            out=RESULTS/"campaign_v2/relational/simple_topological_baselines/runs"/f"{source_method}_seed{training_seed}_l6_h12_e300"
        else:
            out=RUN_ROOT/f"g{graph_seed}_s{signal_seed}"/f"{method}_t{training_seed}_e300"
            metric=train_one(method,edges,training_seed,params,loaders,300,out)
        topo=topology_metrics(reference,edges)
        rows.append({"method":method,"graph_seed":graph_seed,"signal_seed":signal_seed,
                     "training_seed":training_seed,"mse":metric["test_mse"],"mae":metric["test_mae"],
                     "rmse":metric["test_rmse"],**topo,"output_dir":str(out.relative_to(PROJECT))})
    return rows


def aggregate(rows):
    frame=pd.DataFrame(rows)
    node=frame[frame.method=="node_only"][["graph_seed","training_seed","mse"]].rename(columns={"mse":"node_mse"})
    frame=frame.merge(node,on=["graph_seed","training_seed"],how="left")
    frame["improvement_vs_node"]=(frame.node_mse-frame.mse)/frame.node_mse
    frame.to_csv(ROOT/"multiple_graph_runs.csv",index=False)
    metrics=["mse","mae","rmse","improvement_vs_node","precision","recall","f1","jaccard","edge_count","components"]
    per=frame.groupby(["graph_seed","signal_seed","method"])[metrics].agg(["mean","std"]).reset_index()
    per.columns=["_".join(c).rstrip("_") for c in per.columns]
    per.to_csv(ROOT/"per_graph_summary.csv",index=False)
    records=[]
    for method,group in frame.groupby("method"):
        for metric in metrics:
            values=group[metric].dropna(); graph_means=group.groupby("graph_seed")[metric].mean()
            within=group.groupby("graph_seed")[metric].var(ddof=1).mean()
            records.append({"method":method,"metric":metric,"mean_all_runs":values.mean(),
                            "std_all_runs":values.std(ddof=1),"total_variance":values.var(ddof=1),
                            "within_graph_variance":within,"between_graph_variance":graph_means.var(ddof=1),
                            "graph_realizations":group.graph_seed.nunique(),"training_runs":len(values)})
    pd.DataFrame(records).to_csv(ROOT/"across_graph_variance_decomposition.csv",index=False)


def main():
    ROOT.mkdir(parents=True,exist_ok=True); RUN_ROOT.mkdir(parents=True,exist_ok=True)
    stats=[]; rows=[]
    for graph_seed,signal_seed in GRAPH_SIGNALS.items():
        data=dataset_path(graph_seed); stats.append(graph_stats(graph_seed,signal_seed,data))
        for training_seed in TRAIN_SEEDS:
            joint,learned=run_joint(graph_seed,signal_seed,training_seed,data); rows.append(joint)
            rows.append(run_node(graph_seed,signal_seed,training_seed,data))
            rows.extend(run_fixed(graph_seed,signal_seed,training_seed,data,learned))
            aggregate(rows)
            print(f"[OK] graph={graph_seed} train={training_seed} rows={len(rows)}",flush=True)
    pd.DataFrame(stats).to_csv(ROOT/"graph_realization_statistics.csv",index=False)
    aggregate(rows)
    print(f"[DONE] graphs={len(stats)} method-runs={len(rows)}")


if __name__=="__main__": main()
