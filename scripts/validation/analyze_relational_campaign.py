"""Create final relational-campaign tables, figures, decoder matrix and summaries."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results_section_v"
ROOT = RESULTS_ROOT / "campaign_v2/relational"
DATA_DIR = RESULTS_ROOT / "campaign_v2/datasets/homogeneous_relational_g211_s1211"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
RAW = ROOT / "raw_metrics"
sys.path.insert(0, str(RESULTS_ROOT))

from scripts.plot_style import COLORS, apply_style, save_figure


def edges_from_array(value):
    value = np.asarray(value)
    return {(min(int(u), int(v)), max(int(u), int(v))) for u, v in value.T if int(u) != int(v)}


def load_edges(path):
    try:
        value = torch.load(path, map_location="cpu", weights_only=True).numpy()
    except TypeError:
        value = torch.load(path, map_location="cpu").numpy()
    return edges_from_array(value)


def graph_metrics(reference, predicted, n=50):
    tp, fp, fn = len(reference & predicted), len(predicted - reference), len(reference - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    graph = nx.Graph(); graph.add_nodes_from(range(n)); graph.add_edges_from(predicted)
    return {"precision": precision, "recall": recall, "f1": f1,
            "jaccard": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "edge_count": len(predicted),
            "density": nx.density(graph), "mean_degree": 2 * len(predicted) / n,
            "components": nx.number_connected_components(graph)}


def add_comparable_metrics(frame):
    out = frame.copy()
    out["forecast_mse"] = 0.5 * (out.test_pressure_mse + out.test_demand_mse)
    out["task_score_mse"] = 1.0 / (1.0 + out.forecast_mse.clip(lower=0))
    if "f1" in out:
        out["eval0p5_mse"] = 0.5 * out.task_score_mse + 0.5 * out.f1
    return out


def write_table(number, slug, frame):
    TABLES.mkdir(parents=True, exist_ok=True)
    stem = TABLES / f"table_{number}_{slug}"
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".md").write_text(frame.to_markdown(index=False) + "\n", encoding="utf-8")
    stem.with_suffix(".tex").write_text(frame.to_latex(index=False, escape=True), encoding="utf-8")


def export_data(stem, frame):
    path = FIGURES / "data"; path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path / f"{stem}.csv", index=False)


def heatmap(frame, value, stem, label, fmt=".3f"):
    pivot = frame.pivot(index="lags", columns="horizon", values=value).sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(3.45, 2.9))
    image = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, format(pivot.iloc[i, j], fmt), ha="center", va="center", fontsize=8,
                    color="white" if pivot.iloc[i, j] < np.nanmedian(pivot.values) else "black")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xlabel("Prediction horizon H")
    ax.set_ylabel("Input history L")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04); cbar.set_label(label)
    save_figure(fig, FIGURES, stem); plt.close(fig)
    export_data(stem, frame[["lags", "horizon", value]])


def row_topk_decode(score, row_k, symmetry, include_mst, xy):
    n = len(score); work = np.asarray(score, float).copy(); np.fill_diagonal(work, -np.inf)
    directed = np.zeros((n, n), bool)
    for u in range(n):
        idx = np.argpartition(work[u], -row_k)[-row_k:]
        directed[u, idx] = True
    mask = directed & directed.T if symmetry == "mutual" else directed | directed.T
    edges = {(u, v) for u in range(n) for v in range(u + 1, n) if mask[u, v]}
    if include_mst:
        g = nx.Graph(); g.add_nodes_from(range(n))
        for u in range(n):
            for v in range(u + 1, n):
                g.add_edge(u, v, weight=float(np.linalg.norm(xy[u] - xy[v])))
        edges |= {tuple(sorted(e)) for e in nx.minimum_spanning_edges(g, data=False)}
    return edges


def rank_score(score):
    n = len(score); work = score.copy(); np.fill_diagonal(work, -np.inf)
    ranks = np.zeros_like(work, float)
    for u in range(n):
        order = np.argsort(work[u])
        ranks[u, order] = np.arange(n)
    ranks /= max(n - 1, 1)
    np.fill_diagonal(ranks, 0)
    return 0.5 * (ranks + ranks.T)


def minmax(score):
    out = score.copy(); mask = ~np.eye(len(out), dtype=bool); vals = out[mask]
    out = (out - vals.min()) / max(vals.max() - vals.min(), 1e-12); np.fill_diagonal(out, 0)
    return out


def decoder_consensus(main):
    data = np.load(DATA_DIR / "data.npz"); xy = data["node_xy"]; reference = edges_from_array(data["edge_index"])
    rows = []
    for record in main.to_dict("records"):
        run = PROJECT_ROOT / record["output_dir"]
        mean = np.load(run / "dgm_prob_mean.npy"); std = np.load(run / "dgm_prob_std.npy")
        binary = np.zeros_like(mean)
        for u, v in load_edges(run / "dgm_learned_edges_consensus.pt"):
            binary[u, v] = binary[v, u] = 1
        variants = {
            "mean_probability": mean,
            "symmetrized_mean": 0.5 * (mean + mean.T),
            "mean_minus_0p5std": minmax(0.5 * (mean + mean.T) - 0.5 * 0.5 * (std + std.T)),
            "confidence_ratio": minmax(0.5 * (mean + mean.T) / (0.05 + 0.5 * (std + std.T))),
            "rank_based": rank_score(mean),
            "frequency_binary": binary,
        }
        decoders = [(2, "mutual", True), (3, "mutual", True),
                    (2, "union", False), (3, "union", False)]
        for consensus, score in variants.items():
            for row_k, symmetry, mst in decoders:
                started = time.perf_counter()
                predicted = row_topk_decode(score, row_k, symmetry, mst, xy)
                metrics = graph_metrics(reference, predicted)
                rows.append({"training_seed": record["training_seed"], "consensus": consensus,
                             "decoder": f"row_k{row_k}_{symmetry}_{'mst' if mst else 'no_mst'}",
                             "k_decoder": row_k, "symmetry": symmetry, "include_mst": mst,
                             "runtime_seconds": time.perf_counter() - started, **metrics})
    detail = pd.DataFrame(rows); detail.to_csv(RAW / "decoder_consensus_matrix.csv", index=False)
    decoder = detail.groupby("decoder", as_index=False).agg(
        precision_mean=("precision", "mean"), recall_mean=("recall", "mean"), f1_mean=("f1", "mean"),
        f1_std=("f1", "std"), jaccard_mean=("jaccard", "mean"), edge_count_mean=("edge_count", "mean"),
        density_mean=("density", "mean"), components_mean=("components", "mean"), runtime_mean=("runtime_seconds", "mean"))
    consensus = detail.groupby("consensus", as_index=False).agg(
        precision_mean=("precision", "mean"), recall_mean=("recall", "mean"), f1_mean=("f1", "mean"),
        f1_std=("f1", "std"), jaccard_mean=("jaccard", "mean"), edge_count_mean=("edge_count", "mean"),
        components_mean=("components", "mean"), runtime_mean=("runtime_seconds", "mean"))
    return detail, decoder, consensus


def draw_graph(ax, xy, edges, color, width=1.1, linestyle="-"):
    for u, v in sorted(edges):
        ax.plot(xy[[u, v], 0], xy[[u, v], 1], color=color, lw=width, ls=linestyle, alpha=.8, zorder=1)
    ax.scatter(xy[:, 0], xy[:, 1], s=17, facecolor="white", edgecolor="#222", lw=.6, zorder=2)
    ax.set_xlim(-.03, 1.03); ax.set_ylim(-.03, 1.03); ax.set_aspect("equal"); ax.grid(False)
    ax.set_xlabel("Normalized coordinate 1"); ax.set_ylabel("Normalized coordinate 2")


def graph_figures(main):
    data = np.load(DATA_DIR / "data.npz"); xy = data["node_xy"]
    reference = edges_from_array(data["edge_index"]); candidate = edges_from_array(data["initial_edge_index"])
    seed42 = main.loc[main.training_seed == 42].iloc[0]; final_dir = PROJECT_ROOT / seed42.output_dir
    final = load_edges(final_dir / "dgm_learned_edges_softprob.pt")
    for stem, edges, color in [("reference_graph", reference, COLORS["reference"]),
                                ("candidate_graph", candidate, COLORS["candidate"]),
                                ("final_inferred_graph", final, COLORS["learned"])]:
        fig, ax = plt.subplots(figsize=(3.45, 3.0)); draw_graph(ax, xy, edges, color)
        save_figure(fig, FIGURES, stem); plt.close(fig)
        export_data(stem, pd.DataFrame(sorted(edges), columns=["u", "v"]))
    fig, ax = plt.subplots(figsize=(3.45, 3.0))
    draw_graph(ax, xy, reference & final, COLORS["tp"], 1.6)
    draw_graph(ax, xy, final - reference, COLORS["fp"], 1.5)
    draw_graph(ax, xy, reference - final, COLORS["fn"], 1.2, "--")
    handles = [plt.Line2D([0], [0], color=COLORS[k], lw=2, ls="--" if k == "fn" else "-") for k in ("tp","fp","fn")]
    ax.legend(handles, ["TP", "FP", "FN"], frameon=False)
    save_figure(fig, FIGURES, "reconstruction_errors"); plt.close(fig)
    evolution = []
    dirs = []
    for epoch in (1, 25, 150):
        dirs.append((epoch, next((ROOT / "graph_evolution").glob(f"*e{epoch}"))))
    dirs.append((300, final_dir))
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0), sharex=True, sharey=True)
    for ax, (epoch, directory) in zip(axes, dirs):
        edges = load_edges(directory / "dgm_learned_edges_softprob.pt")
        draw_graph(ax, xy, edges, COLORS["learned"], .8); ax.set_xlabel(f"Epoch {epoch}"); ax.set_ylabel("")
        cmp = json.loads((directory / "graph_comparison_softprob/comparison_summary.json").read_text())
        fm = json.loads((directory / "forecast_metrics.json").read_text())
        mse = .5 * (fm["test"]["pressure_mse"] + fm["test"]["demand_mse"])
        f1 = cmp["edge_recovery_undirected"]["f1"]
        evolution.append({"epoch": epoch, "edge_count": len(edges), "f1": f1,
                          "stability": float(np.mean(np.load(directory / "dgm_prob_std.npy"))),
                          "mean_probability": float(np.mean(np.load(directory / "dgm_prob_mean.npy"))),
                          "forecast_mse": mse, "task_score": 1/(1+mse),
                          "eval0p5": .5/(1+mse)+.5*f1})
    save_figure(fig, FIGURES, "graph_evolution"); plt.close(fig)
    evo = pd.DataFrame(evolution); export_data("graph_evolution_metrics", evo)
    fig, ax = plt.subplots(figsize=(3.45, 2.8))
    for column, label, color in [("f1","Graph F1",COLORS["topology"]),("eval0p5",r"$Eval_{0.5}$",COLORS["objective"]),("stability","Mean score std",COLORS["candidate"])]:
        ax.plot(evo.epoch, evo[column], marker="o", label=label, color=color)
    ax.set_xlabel("Training epoch"); ax.set_ylabel("Diagnostic value"); ax.legend(frameon=False)
    save_figure(fig, FIGURES, "graph_evolution_metrics"); plt.close(fig)
    return evo


def main():
    apply_style(); TABLES.mkdir(parents=True, exist_ok=True); (FIGURES / "data").mkdir(parents=True, exist_ok=True)
    main_runs = add_comparable_metrics(pd.read_csv(RAW / "main_runs.csv"))
    joint = add_comparable_metrics(pd.read_csv(RAW / "lh_joint_runs.csv"))
    node = pd.read_csv(RAW / "lh_node_runs.csv")
    alpha = add_comparable_metrics(pd.read_csv(RAW / "alpha_runs.csv"))
    kvals = add_comparable_metrics(pd.read_csv(RAW / "k_dgm_runs.csv"))
    merged = joint.merge(node[["lags","horizon","training_seed","test_mse","test_mae","test_rmse"]],
                         on=["lags","horizon","training_seed"], suffixes=("_joint","_node"))
    merged["joint_gain"] = (merged.test_mse_node - merged.forecast_mse) / merged.test_mse_node
    lh = merged.groupby(["lags","horizon"], as_index=False).agg(
        node_mse_mean=("test_mse_node","mean"), node_mse_std=("test_mse_node","std"),
        graph_mse_mean=("forecast_mse","mean"), graph_mse_std=("forecast_mse","std"),
        graph_gain_mean=("joint_gain","mean"), graph_gain_std=("joint_gain","std"),
        graph_f1_mean=("f1","mean"), graph_f1_std=("f1","std"))
    alpha_agg = alpha.groupby("alpha", as_index=False).agg(
        forecast_mse_mean=("forecast_mse","mean"), forecast_mse_std=("forecast_mse","std"),
        precision_mean=("precision","mean"), recall_mean=("recall","mean"), f1_mean=("f1","mean"),
        f1_std=("f1","std"), jaccard_mean=("jaccard","mean"), edge_count_mean=("edge_count","mean"),
        mean_degree_mean=("mean_degree","mean"), eval0p5_mean=("eval0p5_mse","mean"),
        runtime_mean=("runtime_seconds","mean"))
    k_agg = kvals.groupby("k_dgm", as_index=False).agg(
        forecast_mse_mean=("forecast_mse","mean"), forecast_mse_std=("forecast_mse","std"),
        precision_mean=("precision","mean"), recall_mean=("recall","mean"), f1_mean=("f1","mean"),
        f1_std=("f1","std"), edge_count_mean=("edge_count","mean"), mean_degree_mean=("mean_degree","mean"),
        eval0p5_mean=("eval0p5_mse","mean"), runtime_mean=("runtime_seconds","mean"))
    detail, decoder, consensus = decoder_consensus(main_runs)
    evo = graph_figures(main_runs)

    data = np.load(DATA_DIR / "data.npz"); edges = edges_from_array(data["edge_index"]); graph = nx.Graph(); graph.add_nodes_from(range(50)); graph.add_edges_from(edges)
    lengths = [np.linalg.norm(data["node_xy"][u]-data["node_xy"][v]) for u,v in edges]
    table1 = pd.DataFrame([
        ("generator", "mst_homogeneous"), ("signal mode", "relational"), ("graph seed", 211),
        ("signal seed", 1211), ("nodes", 50), ("edges", len(edges)), ("density", nx.density(graph)),
        ("mean degree", 2*len(edges)/50), ("degree range", "2--4"), ("components", 1),
        ("clustering", nx.average_clustering(graph)), ("average shortest path", nx.average_shortest_path_length(graph)),
        ("bridges", len(list(nx.bridges(graph)))), ("mean edge length", np.mean(lengths)),
        ("timesteps", 720), ("split", "503/109/108"), ("demand self/neighbor", "0.15/0.65"),
        ("pressure self/neighbor state", "0.30/0.60"), ("smoothing tau", 2.2)], columns=["parameter","value"])
    table2 = pd.DataFrame([
        ("Generator","scripts/validation/generate_relational_variant.py","generate","No","graph+signals","writes GT; model never receives it"),
        ("Data module","src/dataset/dataset_epanet.py","EpanetDataModule","No","all","train-only scaler"),
        ("Node-only","scripts/validation/train_node_only_campaign.py","NodeMLP","Yes","baseline","no GT/no edges"),
        ("DGM","src/model/dgm.py","DGM_d","Yes","training","no GT"),
        ("Training/loss","src/train_dgm.py","train_one_epoch","Yes","training","forecast targets; no GT edges"),
        ("Observable Q","src/train_dgm.py","_build_pseudo_edge_target","No","training","observations only"),
        ("Decoder","scripts/experiments/run_train_synthetic.py","export_soft_probability_graph","No","post-processing","no GT"),
        ("Evaluation", "scripts/validation/analyze_relational_campaign.py","graph_metrics","No","post-hoc","uses GT")],
        columns=["module","file","class_or_function","trainable","stage","ground_truth_use"])
    table3 = pd.DataFrame([("scenario","homogeneous_relational_g211_s1211"),("L/H","6/12"),("epochs",300),("training seeds","42--46"),
                           ("optimizer","Adam, lr=1e-3"),("modules","MLP/GCN"),("alpha",.8),("lambda_Q",.5),("lambda_stability",0),
                           ("k_dgm/k_candidate/k_q/k_decoder","2/3/2/3"),("task loss","Huber, target_std, weighted mean"),
                           ("selection","observable J_obs"),("decoder","mean probability, row-k3 mutual + MST")], columns=["parameter","value"])
    table4 = main_runs[["training_seed","best_epoch","forecast_mse","test_mae","test_rmse","precision","recall","f1","jaccard","edge_count","components","eval0p5_mse","runtime_seconds"]]
    table5 = lh[["lags","horizon","node_mse_mean","node_mse_std","graph_mse_mean","graph_mse_std","graph_gain_mean","graph_gain_std"]]
    best_final = evo.rename(columns={"epoch":"checkpoint_epoch"})
    for num, slug, frame in [(1,"synthetic_graph_signal_parameters",table1),(2,"model_modules_roles",table2),(3,"main_training_configuration",table3),
                              (4,"main_multiseed_results",table4),(5,"node_only_vs_graph_forecasting",table5),(6,"lh_sensitivity",lh),
                              (7,"alpha_sensitivity",alpha_agg),(8,"k_dgm_sensitivity",k_agg),(9,"decoder_comparison",decoder),
                              (10,"consensus_comparison",consensus),(11,"checkpoint_comparison",best_final)]: write_table(num,slug,frame)

    heatmap(lh,"node_mse_mean","lh_node_mse_heatmap","Node-only MSE")
    heatmap(lh,"graph_mse_mean","lh_graph_mse_heatmap","Graph-model MSE")
    heatmap(lh,"graph_gain_mean","lh_graph_gain_heatmap","Relative graph gain", ".1%")
    heatmap(lh,"graph_f1_mean","lh_graph_f1_heatmap","Graph F1")
    for xcol, group, stem, xlabel in [("horizon","lags","lh_gain_vs_horizon","Prediction horizon H"),("lags","horizon","lh_gain_vs_lags","Input history L")]:
        fig, ax = plt.subplots(figsize=(3.45,2.8)); rows=[]
        for key, part in lh.groupby(group):
            part=part.sort_values(xcol); ax.plot(part[xcol],100*part.graph_gain_mean,marker="o",label=f"{group[0].upper()}={key}")
            rows.append(part)
        ax.axhline(0,color="#555",lw=.8); ax.set_xlabel(xlabel); ax.set_ylabel("Graph gain over node-only [%]"); ax.legend(frameon=False,ncol=2)
        save_figure(fig,FIGURES,stem); plt.close(fig); export_data(stem,pd.concat(rows))

    def error_plot(frame,x,columns,labels,stem,ylabel):
        fig,ax=plt.subplots(figsize=(3.45,2.8))
        for col,label,color in zip(columns,labels,[COLORS["forecast"],COLORS["topology"],COLORS["objective"]]):
            ax.plot(frame[x],frame[col],marker="o",label=label,color=color)
        ax.set_xlabel(x.replace("_"," ")); ax.set_ylabel(ylabel); ax.legend(frameon=False)
        save_figure(fig,FIGURES,stem); plt.close(fig); export_data(stem,frame)
    error_plot(alpha_agg,"alpha",["forecast_mse_mean"],["Forecast MSE"],"alpha_forecasting","Forecast MSE")
    error_plot(alpha_agg,"alpha",["precision_mean","recall_mean","f1_mean"],["Precision","Recall","F1"],"alpha_graph_metrics","Graph metric")
    error_plot(alpha_agg,"alpha",["eval0p5_mean"],[r"$Eval_{0.5}$"],"alpha_eval0p5",r"$Eval_{0.5}$")
    fig,ax=plt.subplots(figsize=(3.45,2.8)); sc=ax.scatter(alpha_agg.forecast_mse_mean,alpha_agg.f1_mean,c=alpha_agg.alpha,cmap="viridis",s=45)
    ax.set_xlabel("Forecast MSE"); ax.set_ylabel("Graph F1"); fig.colorbar(sc,ax=ax,label=r"$\alpha$")
    save_figure(fig,FIGURES,"alpha_tradeoff"); plt.close(fig); export_data("alpha_tradeoff",alpha_agg)
    error_plot(k_agg,"k_dgm",["precision_mean","recall_mean","f1_mean"],["Precision","Recall","F1"],"k_dgm_graph_metrics","Graph metric")
    error_plot(k_agg,"k_dgm",["forecast_mse_mean"],["Forecast MSE"],"k_dgm_forecasting","Forecast MSE")
    error_plot(k_agg,"k_dgm",["edge_count_mean"],["Inferred edges"],"k_dgm_edge_count","Edges")
    error_plot(k_agg,"k_dgm",["eval0p5_mean"],[r"$Eval_{0.5}$"],"k_dgm_eval0p5",r"$Eval_{0.5}$")

    seed_edges={int(r.training_seed):load_edges(PROJECT_ROOT/r.output_dir/"dgm_learned_edges_softprob.pt") for _,r in main_runs.iterrows()}
    stab=[]
    for a in seed_edges:
        for b in seed_edges:
            if a<b: stab.append({"seed_a":a,"seed_b":b,"edge_jaccard":len(seed_edges[a]&seed_edges[b])/len(seed_edges[a]|seed_edges[b])})
    stab=pd.DataFrame(stab); fig,ax=plt.subplots(figsize=(3.45,2.8)); ax.bar(range(len(stab)),stab.edge_jaccard,color=COLORS["learned"])
    ax.set_xticks(range(len(stab)),[f"{a}-{b}" for a,b in zip(stab.seed_a,stab.seed_b)],rotation=45); ax.set_ylabel("Edge-set Jaccard"); ax.set_xlabel("Training-seed pair")
    save_figure(fig,FIGURES,"seed_stability"); plt.close(fig); export_data("seed_stability",stab)
    final_dir=PROJECT_ROOT/main_runs.loc[main_runs.training_seed==42,"output_dir"].iloc[0]
    score=np.load(final_dir/"dgm_soft_score.npy"); ref=edges_from_array(data["edge_index"]); score_rows=[]
    for u in range(50):
        for v in range(u+1,50): score_rows.append({"score":score[u,v],"reference_edge":(u,v) in ref})
    scores=pd.DataFrame(score_rows); fig,ax=plt.subplots(figsize=(3.45,2.8))
    for flag,label,color in [(True,"Reference edges",COLORS["tp"]),(False,"Non-edges",COLORS["fp"])]: ax.hist(scores.loc[scores.reference_edge==flag,"score"],bins=25,density=True,histtype="step",lw=1.8,label=label,color=color)
    ax.set_xlabel("Relational score"); ax.set_ylabel("Density"); ax.legend(frameon=False); save_figure(fig,FIGURES,"score_distribution"); plt.close(fig); export_data("score_distribution",scores)
    hist=pd.read_csv(final_dir/"training_history.csv")
    for cols,labels,stem,ylabel in [(["train_total"],["Total optimized loss"],"loss_total","Loss"),(["train_task"],["Task loss"],"loss_task","Loss"),(["train_topology","train_w_topology"],["Topology loss","Weighted topology term"],"loss_topology","Loss")]:
        error_plot(hist,"epoch",cols,labels,stem,ylabel)
    val=hist.dropna(subset=["val_task"]); error_plot(val,"epoch",["val_task"],["Validation task loss"],"validation_forecasting","Validation loss")
    print(f"[OK] tables=11 figures={len(list((FIGURES/'png').glob('*.png')))} matrix_rows={len(detail)}")


if __name__ == "__main__":
    main()
