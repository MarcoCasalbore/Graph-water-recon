"""Rerun only joint sweeps whose former 'MSE' was actually Huber loss."""
import argparse
from pathlib import Path
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[2]
RESULTS=PROJECT/"results_section_v"
FINAL=PROJECT/"results_section_v_final"
DATA=RESULTS/"campaign_v2/datasets/homogeneous_relational_g211_s1211"
ROOT=FINAL/"metric_corrected_runs"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_campaign_v2 as campaign  # noqa: E402


def specs():
    for lags in [6,12,24,48]:
        for horizon in [1,3,6,12]:
            for seed in [42,43,44]:
                yield "lh",lags,horizon,.8,2,3,seed
    for alpha in [0,.1,.25,.5,.7,.85,1.0]:
        for seed in [42,43,44]:
            yield "alpha",6,12,alpha,2,3,seed
    for k in [1,2,3,4,5]:
        for seed in [42,43,44]:
            yield "k_dgm",6,12,.8,k,6,seed


def summarize(out,group,lags,horizon,alpha,k,candidate,seed,runtime):
    pred=np.load(out/"test_predictions.npz")
    error=pred["y_pred"].astype(float)-pred["y_true"].astype(float)
    comparison=json.loads((out/"graph_comparison_softprob/comparison_summary.json").read_text(encoding="utf-8"))
    rec=comparison["edge_recovery_undirected"]; graph=comparison["learned_undirected"]
    forecast=json.loads((out/"forecast_metrics.json").read_text(encoding="utf-8"))
    campaign.parse_history(out/"training.log",out/"training_history.csv")
    history=pd.read_csv(out/"training_history.csv")
    selected=int(forecast["selected_epoch"])
    selected_rows=history[history.epoch==selected]
    selected_row=selected_rows.iloc[-1] if len(selected_rows) else history.iloc[-1]
    mse=float(np.mean(error*error)); mae=float(np.mean(np.abs(error)))
    row={"group":group,"lags":lags,"horizon":horizon,"alpha":alpha,"k_dgm":k,
         "k_candidate":candidate,"k_decoder":3,"k_q":2,"training_seed":seed,"epochs":120,
         "selected_epoch":selected,"mse":mse,"rmse":np.sqrt(mse),"mae":mae,
         "precision":rec["precision"],"recall":rec["recall"],"f1":rec["f1"],
         "jaccard":rec["jaccard"],"edge_count":graph["num_edges"],"mean_degree":graph["degree_mean"],
         "components":graph["num_connected_components_undirected"],"runtime_seconds":runtime,
         "total_loss":selected_row.get("train_total",np.nan),"task_loss":selected_row.get("train_task",np.nan),
         "topology_loss":selected_row.get("train_topology",np.nan),
         "task_score":1/(1+mse),"eval0p5":.5/(1+mse)+.5*rec["f1"],"output_dir":str(out.relative_to(PROJECT))}
    return row


def main():
    rows=[]; ROOT.mkdir(parents=True,exist_ok=True)
    for index,(group,lags,horizon,alpha,k,candidate,seed) in enumerate(specs(),1):
        name=f"{group}_g211_s1211_t{seed}_l{lags}_h{horizon}_a{alpha:.2f}_kd{k}_kc{candidate}_e120".replace(".","p")
        out=ROOT/group/name; out.mkdir(parents=True,exist_ok=True)
        started=time.perf_counter()
        if not (out/"test_predictions.npz").exists():
            cmd=campaign.joint_command(out,name,lags,horizon,alpha,k,candidate,seed,120)
            cmd[cmd.index("--data_dir")+1]=str(DATA)
            with (out/"training.log").open("w",encoding="utf-8") as log:
                subprocess.run(cmd,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
        runtime=time.perf_counter()-started
        rows.append(summarize(out,group,lags,horizon,alpha,k,candidate,seed,runtime))
        pd.DataFrame(rows).to_csv(ROOT/"corrected_joint_sweeps.csv",index=False)
        print(f"[OK {index:02d}/84] {name} mse={rows[-1]['mse']:.6f} mae={rows[-1]['mae']:.6f}",flush=True)
    print("[DONE] 84 corrected metric runs")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
