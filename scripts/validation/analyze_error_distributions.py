"""Element-wise error analysis for aligned main-scenario test predictions."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results_section_v"
FINAL = PROJECT / "results_section_v_final"
ROOT = FINAL / "error_analysis"
PRED = ROOT / "predictions"
JOINT = ROOT / "joint_retrained"
FIG = ROOT / "figures"
TABLE = ROOT / "tables"
sys.path.insert(0, str(RESULTS))
from scripts.plot_style import COLORS, apply_style, save_figure  # noqa: E402

METHOD_FILES = {
    "node_only": "node_only",
    "proposed_joint": None,
    "euclidean_mst_frozen": "euclidean_mst",
    "geometric_mutual_knn_frozen": "geometric_knn3_mutual",
    "geometric_union_knn_frozen": "geometric_knn3_union",
    "correlation_frozen": "correlation_abs_knn3_union",
    "hybrid_frozen": "distance_correlation_hybrid_knn3_union",
    "proposed_learned_graph_frozen": "proposed_final_decoded_graph",
    "reference_graph_frozen": "oracle_reference_graph",
    "random_graph_frozen": "random_final_decoded_edge_count_control",
}
LABEL = {
    "node_only":"Node-only", "proposed_joint":"Proposed joint", "euclidean_mst_frozen":"MST frozen",
    "geometric_mutual_knn_frozen":"Mutual 3-NN", "geometric_union_knn_frozen":"Union 3-NN",
    "correlation_frozen":"Correlation", "hybrid_frozen":"Hybrid",
    "proposed_learned_graph_frozen":"Learned graph frozen", "reference_graph_frozen":"Reference control",
    "random_graph_frozen":"Random control",
}
VARIABLES = ["pressure", "demand"]


def load(method, seed):
    base = METHOD_FILES[method]
    if method == "proposed_joint":
        path = JOINT / f"joint_g211_s1211_t{seed}_l6_h12_e300/test_predictions.npz"
    else:
        path = PRED / f"{base}_seed{seed}.npz"
    archive = np.load(path)
    return archive["y_true"].astype(np.float64), archive["y_pred"].astype(np.float64)


def distribution_stats(error):
    absolute = np.abs(error).reshape(-1)
    return {"mse": np.mean(error*error), "rmse": np.sqrt(np.mean(error*error)),
            "mae": absolute.mean(), "median_absolute_error": np.median(absolute),
            "q25_absolute_error": np.quantile(absolute,.25), "q75_absolute_error": np.quantile(absolute,.75),
            "q90_absolute_error": np.quantile(absolute,.90), "q95_absolute_error": np.quantile(absolute,.95),
            "q99_absolute_error": np.quantile(absolute,.99), "max_absolute_error": absolute.max(),
            "std_absolute_error": absolute.std(ddof=1)}


def save_table(frame, stem):
    TABLE.mkdir(parents=True,exist_ok=True)
    frame.to_csv(TABLE/f"{stem}.csv",index=False)
    (TABLE/f"{stem}.md").write_text(frame.to_markdown(index=False)+"\n",encoding="utf-8")
    (TABLE/f"{stem}.tex").write_text(frame.to_latex(index=False,float_format="%.6f"),encoding="utf-8")


def main():
    apply_style(); (FIG/"data").mkdir(parents=True,exist_ok=True)
    overall=[]; horizon=[]; variable=[]; node=[]; cdf=[]
    reference=None
    probabilities=np.linspace(0,1,501)
    pooled={method:[] for method in METHOD_FILES}
    for method in METHOD_FILES:
        for seed in [42,43,44,45,46]:
            truth,pred=load(method,seed)
            if truth.shape != (91,50,12,2): raise RuntimeError(f"unexpected shape {truth.shape}: {method}")
            if reference is None: reference=truth
            if not np.allclose(reference,truth): raise RuntimeError(f"unaligned y_true: {method}, seed={seed}")
            error=pred-truth; absolute=np.abs(error); pooled[method].append(absolute.reshape(-1))
            overall.append({"method":method,"training_seed":seed,**distribution_stats(error)})
            for step in range(12):
                horizon.append({"method":method,"training_seed":seed,"horizon_step":step+1,
                                **distribution_stats(error[:,:,step,:])})
            for feature,name in enumerate(VARIABLES):
                variable.append({"method":method,"training_seed":seed,"variable":name,
                                 **distribution_stats(error[:,:,:,feature])})
            for node_id in range(50):
                node.append({"method":method,"training_seed":seed,"node":node_id,
                             **distribution_stats(error[:,node_id,:,:])})
    overall=pd.DataFrame(overall); horizon=pd.DataFrame(horizon)
    variable=pd.DataFrame(variable); node=pd.DataFrame(node)
    for method,parts in pooled.items():
        values=np.concatenate(parts)
        quantiles=np.quantile(values,probabilities)
        cdf.extend({"method":method,"cdf_probability":p,"absolute_error":q} for p,q in zip(probabilities,quantiles))
    cdf=pd.DataFrame(cdf)
    overall.to_csv(FIG/"data/error_statistics_by_seed.csv",index=False)
    horizon.to_csv(FIG/"data/error_by_horizon_seed.csv",index=False)
    variable.to_csv(FIG/"data/error_by_variable_seed.csv",index=False)
    node.to_csv(FIG/"data/error_by_node_seed.csv",index=False)
    cdf.to_csv(FIG/"data/error_ecdf_quantiles.csv",index=False)
    summary=overall.groupby("method").agg({c:["mean","std"] for c in overall.columns[2:]}).reset_index()
    summary.columns=["_".join(c).rstrip("_") for c in summary.columns]
    save_table(summary,"table_7_error_distribution_statistics")

    focus=["node_only","proposed_joint","proposed_learned_graph_frozen","reference_graph_frozen"]
    fig,ax=plt.subplots(figsize=(4.9,3.2))
    for method in focus:
        frame=cdf[cdf.method==method]
        ax.plot(frame.absolute_error,frame.cdf_probability,label=LABEL[method])
    ax.set(xlabel="Absolute error",ylabel="Empirical CDF",xlim=(0,cdf[cdf.cdf_probability<=.995].absolute_error.max()))
    ax.legend(frameon=False); save_figure(fig,FIG,"absolute_error_ecdf"); plt.close(fig)

    fig,ax=plt.subplots(figsize=(6.6,3.2)); boxes=[]; labels=[]
    for method in METHOD_FILES:
        row=summary[summary.method==method].iloc[0]
        boxes.append({"label":LABEL[method],"med":row.median_absolute_error_mean,"q1":row.q25_absolute_error_mean,
                      "q3":row.q75_absolute_error_mean,"whislo":0,"whishi":row.q95_absolute_error_mean,"fliers":[]})
        labels.append(LABEL[method])
    ax.bxp(boxes,showfliers=False); ax.tick_params(axis="x",rotation=27); ax.set_ylabel("Absolute error")
    save_figure(fig,FIG,"absolute_error_boxplot"); plt.close(fig)

    horizon_summary=horizon.groupby(["method","horizon_step"]).mae.agg(["mean","std"]).reset_index()
    horizon_summary.to_csv(FIG/"data/error_by_horizon_summary.csv",index=False)
    fig,ax=plt.subplots(figsize=(5.4,3.3))
    for method in focus:
        frame=horizon_summary[horizon_summary.method==method]
        ax.plot(frame.horizon_step,frame["mean"],marker="o",label=LABEL[method])
        ax.fill_between(frame.horizon_step,frame["mean"]-frame["std"],frame["mean"]+frame["std"],alpha=.09)
    ax.set(xlabel="Forecast horizon step",ylabel="Mean absolute error",xticks=[1,3,6,9,12])
    ax.legend(frameon=False); save_figure(fig,FIG,"error_vs_horizon"); plt.close(fig)

    variable_summary=variable.groupby(["method","variable"])[["mse","mae"]].agg(["mean","std"]).reset_index()
    variable_summary.columns=["_".join(c).rstrip("_") for c in variable_summary.columns]
    variable_summary.to_csv(FIG/"data/error_by_variable_summary.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(7.0,2.9),sharex=True)
    for ax,var in zip(axes,VARIABLES):
        frame=variable_summary[(variable_summary.variable==var)&(variable_summary.method.isin(focus))]
        x=np.arange(len(frame)); ax.bar(x,frame.mae_mean,yerr=frame.mae_std,capsize=3,color=COLORS["forecast"])
        ax.set_xticks(x,[LABEL[m] for m in frame.method],rotation=24,ha="right"); ax.set_ylabel(f"{var.capitalize()} MAE")
    save_figure(fig,FIG,"error_by_variable"); plt.close(fig)

    node_summary=node.groupby(["method","node"]).mae.agg(["mean","std"]).reset_index()
    wide=node_summary.pivot(index="node",columns="method",values="mean")
    wide["joint_gain_vs_node"]=(wide.node_only-wide.proposed_joint)/wide.node_only
    best_joint=wide.proposed_joint.nsmallest(3).index.tolist(); worst_joint=wide.proposed_joint.nlargest(3).index.tolist()
    max_gain=wide.joint_gain_vs_node.nlargest(4).index.tolist()
    selected=list(dict.fromkeys(best_joint+worst_joint+max_gain))
    selection=pd.DataFrame({"node":selected,"selection_role":["selected" for _ in selected]})
    selection.to_csv(FIG/"data/selected_error_nodes.csv",index=False)
    plot=node_summary[(node_summary.node.isin(selected))&(node_summary.method.isin(["node_only","proposed_joint"]))]
    fig,ax=plt.subplots(figsize=(6.1,3.2)); x=np.arange(len(selected)); width=.36
    for offset,method in [(-width/2,"node_only"),(width/2,"proposed_joint")]:
        frame=plot[plot.method==method].set_index("node").loc[selected]
        ax.bar(x+offset,frame["mean"],width,yerr=frame["std"],capsize=2,label=LABEL[method])
    ax.set_xticks(x,[f"N{n}" for n in selected]); ax.set_ylabel("Node MAE"); ax.legend(frameon=False)
    save_figure(fig,FIG,"error_by_selected_node"); plt.close(fig)

    scatter=wide[["node_only","proposed_joint","joint_gain_vs_node"]].reset_index()
    scatter.to_csv(FIG/"data/node_error_scatter.csv",index=False)
    fig,ax=plt.subplots(figsize=(3.6,3.4)); ax.scatter(scatter.node_only,scatter.proposed_joint,s=24,color=COLORS["learned"])
    bound=max(scatter.node_only.max(),scatter.proposed_joint.max()); ax.plot([0,bound],[0,bound],"--",color="black",lw=.9)
    ax.set(xlabel="Node-only node MAE",ylabel="Proposed-joint node MAE",xlim=(0,bound*1.04),ylim=(0,bound*1.04))
    save_figure(fig,FIG,"node_only_vs_joint_node_error"); plt.close(fig)

    jf=horizon_summary[horizon_summary.method.isin(["proposed_joint","proposed_learned_graph_frozen"])]
    jf.to_csv(FIG/"data/joint_vs_frozen_horizon.csv",index=False)
    fig,ax=plt.subplots(figsize=(4.3,3.1))
    for method in ["proposed_joint","proposed_learned_graph_frozen"]:
        frame=jf[jf.method==method]; ax.plot(frame.horizon_step,frame["mean"],marker="o",label=LABEL[method])
    ax.set(xlabel="Forecast horizon step",ylabel="Mean absolute error"); ax.legend(frameon=False)
    save_figure(fig,FIG,"joint_vs_frozen_horizon"); plt.close(fig)

    joint=summary[summary.method=="proposed_joint"].iloc[0]; node_row=summary[summary.method=="node_only"].iloc[0]
    findings={"joint_mse_lower":bool(joint.mse_mean<node_row.mse_mean),"joint_mae_lower":bool(joint.mae_mean<node_row.mae_mean),
              "joint_median_lower":bool(joint.median_absolute_error_mean<node_row.median_absolute_error_mean),
              "joint_q95_lower":bool(joint.q95_absolute_error_mean<node_row.q95_absolute_error_mean),
              "joint_q99_lower":bool(joint.q99_absolute_error_mean<node_row.q99_absolute_error_mean),
              "nodes_joint_mae_better":int((wide.proposed_joint<wide.node_only).sum()),"nodes_total":50}
    (ROOT/"error_distribution_findings.json").write_text(__import__("json").dumps(findings,indent=2),encoding="utf-8")
    print(findings); print(f"[OK] rows overall={len(overall)} horizon={len(horizon)} variable={len(variable)} node={len(node)}")


if __name__=="__main__": main()
