"""Validate non-oracle construction, matched forecasting and baseline artifacts."""
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
CAMPAIGN = PROJECT / "results_section_v/campaign_v2/relational"
ROOT = CAMPAIGN / "simple_topological_baselines"
RUNS = ROOT / "runs"
DATASET = PROJECT / "results_section_v/campaign_v2/datasets/homogeneous_relational_g211_s1211"
checks = []


def check(name, condition, detail):
    checks.append({"check": name, "status": "PASS" if condition else "FAIL", "detail": str(detail)})


def edges(path):
    frame = pd.read_csv(path)
    return {tuple(sorted((int(r.u), int(r.v)))) for r in frame.itertuples()}


def metrics(reference, predicted):
    tp, fp, fn = len(reference & predicted), len(predicted-reference), len(reference-predicted)
    p, r = tp/max(1,tp+fp), tp/max(1,tp+fn)
    return tp, fp, fn, p, r, 2*p*r/max(1e-12,p+r), tp/max(1,tp+fp+fn)


def main():
    frame = pd.read_csv(ROOT / "fixed_graph_forecasting_runs.csv")
    methods = ["euclidean_mst", "geometric_knn3_mutual", "geometric_knn3_union",
               "correlation_abs_knn3_union", "distance_correlation_hybrid_knn3_union",
               "proposed_final_decoded_graph", "random_final_decoded_edge_count_control",
               "oracle_reference_graph"]
    expected = {(m, s) for m in methods for s in [42,43,44,45,46]}
    actual = set(zip(frame.method, frame.training_seed))
    check("complete_40_run_grid", len(frame)==40 and actual==expected and (frame.status=="completed").all(),
          f"rows={len(frame)}, methods={frame.method.nunique()}, seeds={frame.training_seed.nunique()}")
    check("finite_forecast_metrics", np.isfinite(frame[["test_mse","test_mae","test_rmse"]]).all().all(),
          "MSE/MAE/RMSE finite")

    archive = np.load(DATASET / "data.npz")
    reference = {tuple(sorted((int(u),int(v)))) for u,v in archive["edge_index"].T if u!=v}
    configs_ok = topology_ok = no_self = True
    max_error = 0.0
    current_edges = {}
    for row in frame.itertuples():
        run = RUNS / f"{row.method}_seed{row.training_seed}_l6_h12_e300"
        config = json.loads((run/"config.json").read_text(encoding="utf-8"))
        predicted = edges(run/"edges.csv")
        current_edges[(row.method,row.training_seed)] = predicted
        configs_ok &= (config["graph_frozen"] is True and config["lags"]==6 and config["horizon"]==12
                       and config["epochs"]==300 and config["correlation_fit_end"]==503)
        configs_ok &= config["reference_used_for_construction"] == (row.method=="oracle_reference_graph")
        no_self &= all(u!=v for u,v in predicted)
        truth = metrics(reference,predicted)
        observed = (row.tp,row.fp,row.fn,row.precision,row.recall,row.f1,row.jaccard)
        max_error = max(max_error,max(abs(float(a)-float(b)) for a,b in zip(truth,observed)))
        topology_ok &= max_error < 1e-10
        configs_ok &= all((run/name).exists() for name in ["best_forecaster.pt","training_history.csv","metrics.json"])
    check("saved_configs_and_artifacts", configs_ok, "40 configs, edge lists, checkpoints and histories")
    check("posthoc_topology_recomputed", topology_ok, f"max_error={max_error:.3g}")
    check("no_self_loops", no_self, "all baseline edge lists")

    static = ["euclidean_mst","geometric_knn3_mutual","geometric_knn3_union",
              "correlation_abs_knn3_union","distance_correlation_hybrid_knn3_union","oracle_reference_graph"]
    deterministic = all(current_edges[(m,42)]==current_edges[(m,s)] for m in static for s in [43,44,45,46])
    check("static_rules_seed_invariant", deterministic, static)
    matched_random = all(len(current_edges[("random_final_decoded_edge_count_control",s)]) ==
                         len(current_edges[("proposed_final_decoded_graph",s)]) for s in [42,43,44,45,46])
    check("random_matches_proposed_edge_count", matched_random, "matched per training seed, not reference count")

    score_ok = True
    for seed in [42,43,44,45,46]:
        saved = np.load(ROOT/f"baseline_scores_seed{seed}.npz")
        score_ok &= int(saved["train_end"])==503 and np.isfinite(saved["correlation_abs"]).all()
    check("correlation_train_split_only", score_ok, "train_end=503 stored in every score archive")
    source = (PROJECT/"scripts/validation/run_simple_topology_baselines.py").read_text(encoding="utf-8")
    check("fixed_exact_forecast_path", all(x in source for x in ["GCNConv(32, 32)","GCNConv(32, 16)","GCNConv(16, 8)","graph_frozen"]),
          "12->32 MLP, 32->32->16->8 GCN, 8->24 head")

    seed_metrics = pd.read_csv(ROOT/"all_method_seed_metrics.csv")
    beta_cols = [c for c in seed_metrics if c.startswith("joint_beta_")]
    expected_beta = {"joint_beta_0","joint_beta_0.1","joint_beta_0.25","joint_beta_0.5",
                     "joint_beta_0.75","joint_beta_0.9","joint_beta_1"}
    formula_ok = set(beta_cols)==expected_beta
    for col in beta_cols:
        beta=float(col.split("_")[-1])
        formula_ok &= np.allclose(seed_metrics[col], beta*seed_metrics.task_score+(1-beta)*seed_metrics.f1)
    check("joint_score_beta_formula", formula_ok, sorted(beta_cols))
    node = seed_metrics[seed_metrics.method=="node_only_empty_graph"]
    check("node_only_empty_graph_convention", (node.f1==0).all() and (node.edge_count==0).all(),
          "GraphScore=0 and E=0")
    main = pd.read_csv(CAMPAIGN/"tables/table_4_main_multiseed_results.csv")
    proposed = seed_metrics[seed_metrics.method=="proposed_joint"].sort_values("training_seed")
    check("proposed_metrics_match_main", np.allclose(proposed.test_mse, main.sort_values("training_seed").forecast_mse),
          "five primary rows reused without retraining")
    check("eval0p5_exact_convention", np.allclose(proposed.joint_beta_0p5 if "joint_beta_0p5" in proposed else proposed["joint_beta_0.5"],
          .5/(1+proposed.test_mse)+.5*proposed.f1), "beta weights TaskScore; 1-beta weights F1")

    required_tables = ["table_A_topology_only_baselines","table_B_fixed_graph_forecasting",
                       "table_C_joint_task_topology"]
    tables_ok = all((ROOT/f"tables/{stem}.{ext}").stat().st_size>0 for stem in required_tables for ext in ["csv","md","tex"])
    check("required_tables_all_formats", tables_ok, "Tables A-C x CSV/Markdown/LaTeX")
    stems={p.stem for p in (ROOT/"figures/png").glob("*.png")}
    figures_ok=len(stems)==7 and all((ROOT/f"figures/{ext}/{stem}.{ext}").stat().st_size>0
                                    for stem in stems for ext in ["png","pdf","svg"])
    check("seven_figure_sets", figures_ok, sorted(stems))
    check("figure_source_data", len(list((ROOT/"figures/data").glob("*.csv")))>=5,
          f"csv={len(list((ROOT/'figures/data').glob('*.csv')))}")
    main_readme=(CAMPAIGN/"README.md").read_text(encoding="utf-8")
    baseline_readme=(ROOT/"README.md").read_text(encoding="utf-8")
    check("readme_interpretation", "## Simple Topological Baselines" in main_readme and "beta" in baseline_readme,
          "main and standalone README")
    forbidden = re.compile(r"graph F1.*(choose|select)|reference edge count.*baseline",re.I)
    check("no_documented_oracle_tuning", not forbidden.search(baseline_readme), "manual/source audit")

    report=pd.DataFrame(checks)
    report.to_csv(ROOT/"validation_checks.csv",index=False)
    (ROOT/"validation_checks.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    (ROOT/"validation_checks.md").write_text("# Simple topology baseline validation\n\n"+report.to_markdown(index=False)+
        f"\n\nOverall: **{'PASS' if (report.status=='PASS').all() else 'FAIL'}**.\n",encoding="utf-8")
    print(report.to_string(index=False))
    failed=report[report.status!="PASS"]
    if len(failed): raise SystemExit(f"{len(failed)} validation checks failed")
    print(f"[PASS] {len(report)}/{len(report)} checks")


if __name__ == "__main__": main()
