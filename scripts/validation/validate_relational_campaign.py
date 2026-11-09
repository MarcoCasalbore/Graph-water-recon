"""Validate completeness and numerical consistency of the final Section V package."""
from pathlib import Path
import json
import re

import networkx as nx
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results_section_v"
ROOT = RESULTS / "campaign_v2/relational"
RAW = ROOT / "raw_metrics"
checks = []


def check(name, passed, detail):
    checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": str(detail)})


def main():
    archive = np.load(RESULTS / "campaign_v2/datasets/homogeneous_relational_g211_s1211/data.npz")
    finite = all(np.isfinite(archive[k]).all() for k in archive.files)
    check("dataset_finite", finite, archive.files)
    pairs = {tuple(sorted((int(u), int(v)))) for u, v in archive["edge_index"].T if u != v}
    graph = nx.Graph()
    graph.add_nodes_from(range(len(archive["node_xy"])))
    graph.add_edges_from(pairs)
    degree = np.array([v for _, v in graph.degree()])
    graph_ok = (graph.number_of_nodes() == 50 and graph.number_of_edges() == 79
                and nx.is_connected(graph) and len(list(nx.bridges(graph))) == 0
                and degree.min() == 2 and degree.max() == 4)
    check("graph_invariants", graph_ok,
          f"N={len(graph)}, E={graph.number_of_edges()}, components={nx.number_connected_components(graph)}, "
          f"bridges={len(list(nx.bridges(graph)))}, degree={degree.min()}-{degree.max()}")

    expected = {"main_runs.csv": 5, "main_node_runs.csv": 5, "lh_joint_runs.csv": 48,
                "lh_node_runs.csv": 48, "alpha_runs.csv": 21, "k_dgm_runs.csv": 15}
    frames = {}
    for filename, count in expected.items():
        frame = pd.read_csv(RAW / filename)
        frames[filename] = frame
        status_ok = len(frame) == count and (frame.status == "completed").all()
        numeric = frame.select_dtypes(include=np.number)
        check(f"runs_{filename}", status_ok and np.isfinite(numeric.to_numpy()).all(),
              f"rows={len(frame)}, expected={count}")

    lh_joint, lh_node = frames["lh_joint_runs.csv"], frames["lh_node_runs.csv"]
    grid = {(l, h, s) for l in [6, 12, 24, 48] for h in [1, 3, 6, 12] for s in [42, 43, 44]}
    for name, frame in [("joint", lh_joint), ("node", lh_node)]:
        actual = set(zip(frame.lags, frame.horizon, frame.training_seed))
        check(f"lh_grid_{name}", actual == grid, f"cells={len(actual)}")

    joint = frames["main_runs.csv"].copy()
    joint["forecast_mse"] = .5 * (joint.test_pressure_mse + joint.test_demand_mse)
    node = frames["main_node_runs.csv"]
    paired = joint.merge(node, on="training_seed", suffixes=("_joint", "_node"))
    gains = (paired.test_mse_node - paired.forecast_mse) / paired.test_mse_node
    check("main_joint_wins_all_seeds", (gains > 0).all(),
          f"mean_gain={100*gains.mean():.3f}%, wins={(gains > 0).sum()}/5")
    final_main = pd.read_csv(ROOT / "tables/table_4_main_multiseed_results.csv")
    eval_expected = .5 / (1 + final_main.forecast_mse) + .5 * final_main.f1
    check("comparable_eval0p5_formula", np.allclose(eval_expected, final_main.eval0p5_mse),
          f"max_error={np.max(np.abs(eval_expected-final_main.eval0p5_mse)):.3g}")

    histories = list((ROOT / "main_experiment").glob("*/training_history.csv"))
    loss_ok, max_error = len(histories) == 5, 0.0
    required = {"train_total", "train_task", "train_topology", "train_stability",
                "train_w_task", "train_w_graph", "train_w_topology", "train_w_stability"}
    for path in histories:
        history = pd.read_csv(path)
        loss_ok &= required.issubset(history.columns) and len(history) == 300
        reconstructed = history.train_w_task + history.train_w_graph + history.train_w_topology + history.train_w_stability
        max_error = max(max_error, float(np.max(np.abs(history.train_total - reconstructed))))
    check("loss_logging_and_identity", loss_ok and max_error < 2e-4,
          f"histories={len(histories)}, max_rounding_error={max_error:.3g}")

    epoch = pd.read_csv(ROOT / "figures/data/epoch_diagnostics.csv")
    epoch_ok = (len(epoch) == 300 and epoch.epoch.tolist() == list(range(1, 301))
                and epoch.oracle_posthoc.all() and (~epoch.used_for_selection).all())
    expected_eval = .5 * epoch.diagnostic_task_score + .5 * epoch.diagnostic_graph_f1
    check("epoch_diagnostic", epoch_ok and np.allclose(expected_eval, epoch.diagnostic_eval0p5),
          f"epochs={len(epoch)}, oracle_posthoc={epoch.oracle_posthoc.all()}")

    matrix = pd.read_csv(RAW / "decoder_consensus_matrix.csv")
    check("decoder_consensus_matrix", len(matrix) == 120,
          f"rows={len(matrix)}, decoders={matrix.decoder.nunique()}, consensus={matrix.consensus.nunique()}")
    manifest = pd.read_csv(ROOT / "experiment_manifest.csv")
    check("unified_manifest", len(manifest) == 142 and (manifest.status == "completed").all(),
          f"rows={len(manifest)}")

    table_ok = all((ROOT / "tables" / f"table_{i}_{suffix}.{ext}").exists()
                   for i, suffix in [(1,"synthetic_graph_signal_parameters"),(2,"model_modules_roles"),
                   (3,"main_training_configuration"),(4,"main_multiseed_results"),
                   (5,"node_only_vs_graph_forecasting"),(6,"lh_sensitivity"),
                   (7,"alpha_sensitivity"),(8,"k_dgm_sensitivity"),
                   (9,"decoder_comparison"),(10,"consensus_comparison"),(11,"checkpoint_comparison")]
                   for ext in ["csv", "md", "tex"])
    check("tables_all_formats", table_ok, "11 tables x CSV/Markdown/LaTeX")
    stems = {p.stem for p in (ROOT / "figures/png").glob("*.png")}
    formats_ok = all((ROOT / f"figures/{ext}/{stem}.{ext}").stat().st_size > 0
                     for stem in stems for ext in ["png", "pdf", "svg"])
    check("figures_all_formats", len(stems) >= 32 and formats_ok, f"figure_sets={len(stems)}")
    check("figure_source_data", len(list((ROOT / "figures/data").glob("*"))) >= 31,
          f"source_files={len(list((ROOT / 'figures/data').glob('*')))}")

    corrupt = []
    pattern = re.compile(r"\|\s*k\s*\|\s*e\s*\|\s*y\s*\|\s*s\s*\|", re.I)
    for path in RESULTS.rglob("*.md"):
        if pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
            corrupt.append(str(path.relative_to(PROJECT)))
    check("no_characterwise_keys_headers", not corrupt, corrupt or "zero matches")
    check("documentation", (ROOT / "README.md").exists() and (ROOT / "generator_protocol.md").exists(),
          "README.md and generator_protocol.md")

    report = pd.DataFrame(checks)
    report.to_csv(ROOT / "validation_checks.csv", index=False)
    (ROOT / "validation_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    lines = ["# Relational campaign validation", "", report.to_markdown(index=False), "",
             f"Overall: **{'PASS' if (report.status == 'PASS').all() else 'FAIL'}**."]
    (ROOT / "validation_checks.md").write_text("\n".join(lines), encoding="utf-8")
    print(report.to_string(index=False))
    failures = report[report.status != "PASS"]
    if len(failures):
        raise SystemExit(f"validation failed: {len(failures)} checks")
    print(f"[PASS] {len(report)}/{len(report)} checks")


if __name__ == "__main__":
    main()
