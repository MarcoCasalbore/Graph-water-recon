"""Definitive GT-free study on four fixed SimBench-based electric grids."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import simbench as sb
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.electric.generate_simbench_electric import (  # noqa: E402
    active_topology_edges,
    add_plausible_tie_lines,
    aggregate_bus_power,
    apply_profile_row,
    build_knn_edges,
    bus_lookup,
    make_splits,
    parse_bus_xy,
    to_edge_index,
)
from scripts.experiments.run_section_vi_water_final import (  # noqa: E402
    DECODER_CONFIG,
    evaluate_neural_decoder,
    observable_score,
    undirected_edges,
)
from scripts.experiments.run_train_synthetic import (  # noqa: E402
    build_knn_candidate_mask,
    build_q_dynamic_candidate_mask,
)
from src.train_dgm import (  # noqa: E402
    _geometry_rank_affinity,
    _inverse_source_target_affinity,
    _nonmediated_geometry_affinity,
    _offdiag_minmax,
    _pressure_smooth_affinity,
)


SCENARIOS = {
    "electric_a_semiurban_radial": {
        "topology": "electric_a_semiurban_radial.npz",
        "code": "1-LV-semiurb4--0-sw",
        "tie_lines": 0,
        "tie_seed": 1101,
    },
    "electric_b_urban_radial": {
        "topology": "electric_b_urban_radial.npz",
        "code": "1-LV-urban6--0-sw",
        "tie_lines": 0,
        "tie_seed": 1102,
    },
    "electric_c_semiurban_tied": {
        "topology": "electric_c_semiurban_tied.npz",
        "code": "1-LV-semiurb4--0-sw",
        "tie_lines": 12,
        "tie_seed": 1101,
    },
    "electric_d_urban_tied": {
        "topology": "electric_d_urban_tied.npz",
        "code": "1-LV-urban6--0-sw",
        "tie_lines": 16,
        "tie_seed": 1102,
    },
}

TUNING_SEED = 7
FINAL_SEEDS = (11, 22, 33, 44, 55)
PROFILE_STARTS = {7: 0, 11: 4000, 22: 8000, 33: 12000, 44: 16000, 55: 20000}

ALPHAS = tuple(round(value / 10, 1) for value in range(1, 10))
LAMBDA_QS = (0.1, 0.25, 0.5, 1.0, 2.0)
CANDIDATE_KS = (3, 5, 8)
DGM_KS = (2, 3)
LEARNING_RATES = (5e-4, 1e-3)
COORDINATE_DIMS = (0, 4)

COMMON = {
    "timesteps": 720,
    "profile_stride": 4,
    "profile_step_seconds": 900,
    "initial_knn": 3,
    "lags": 4,
    "prediction_window": 12,
    "batch_size": 16,
    "check_every": 4,
    "ffun": "mlp",
    "gfun": "gcn",
    "prior": "observable_electric_full",
    "lambda_stability": 0.1,
    "task_loss": "huber",
    "task_huber_delta": 1.0,
    "task_feature_combine": "weighted_mean",
    "task_feature_normalization": "target_std",
    "voltage_weight": 1.0,
    "net_power_weight": 1.0,
}


def canonical(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _simulate_profiles(net, timesteps: int, stride: int, start: int, lookup: dict[int, int]):
    absolute = sb.get_absolute_values(net, True)
    max_steps = min(len(value) for value in absolute.values() if hasattr(value, "__len__") and len(value))
    profile_indices = (start + stride * np.arange(timesteps)) % max_steps
    n = len(net.bus)

    voltage = np.zeros((timesteps, n), dtype=np.float32)
    va_degree = np.zeros((timesteps, n), dtype=np.float32)
    p_load = np.zeros((timesteps, n), dtype=np.float32)
    q_load = np.zeros((timesteps, n), dtype=np.float32)
    p_sgen = np.zeros((timesteps, n), dtype=np.float32)
    q_sgen = np.zeros((timesteps, n), dtype=np.float32)
    converged = np.zeros(timesteps, dtype=bool)

    import pandapower as pp

    for t, profile_idx in enumerate(profile_indices):
        apply_profile_row(net, absolute, int(profile_idx))
        try:
            pp.runpp(net, calculate_voltage_angles=False, init="auto", numba=False)
            converged[t] = bool(net.converged)
        except Exception:
            converged[t] = False
        if converged[t]:
            voltage[t] = net.res_bus["vm_pu"].to_numpy(dtype=np.float32)
            va_degree[t] = net.res_bus["va_degree"].to_numpy(dtype=np.float32)
        elif t:
            voltage[t] = voltage[t - 1]
            va_degree[t] = va_degree[t - 1]
        else:
            voltage[t] = 1.0
        p_l, q_l, p_g, q_g = aggregate_bus_power(net, n, lookup)
        p_load[t], q_load[t], p_sgen[t], q_sgen[t] = p_l, q_l, p_g, q_g

    return {
        "voltage_pu": voltage,
        "va_degree": va_degree,
        "p_load_mw": p_load,
        "q_load_mvar": q_load,
        "p_sgen_mw": p_sgen,
        "q_sgen_mvar": q_sgen,
        "net_p_mw": p_load - p_sgen,
        "net_q_mvar": q_load - q_sgen,
        "profile_index": profile_indices.astype(np.int64),
        "powerflow_converged": converged,
    }


def prepare_dataset(
    topology_path: Path,
    out_dir: Path,
    scenario: str,
    signal_seed: int,
) -> None:
    data_path = out_dir / "data.npz"
    meta_path = out_dir / "meta.json"
    if data_path.exists() and meta_path.exists():
        return

    cfg = SCENARIOS[scenario]
    fixed = np.load(topology_path)
    xy = fixed["node_xy"].astype(np.float32)
    gt = undirected_edges(fixed["edge_index"])

    net = sb.get_simbench_net(str(cfg["code"]))
    lookup = bus_lookup(net)
    rebuilt_xy = parse_bus_xy(net)
    add_plausible_tie_lines(
        net,
        node_xy=rebuilt_xy,
        lookup=lookup,
        count=int(cfg["tie_lines"]),
        seed=int(cfg["tie_seed"]),
    )
    rebuilt = active_topology_edges(net, lookup)
    if rebuilt != gt or not np.allclose(rebuilt_xy, xy):
        raise RuntimeError(f"Fixed topology does not match regenerated SimBench network: {scenario}")

    series = _simulate_profiles(
        net,
        timesteps=int(COMMON["timesteps"]),
        stride=int(COMMON["profile_stride"]),
        start=int(PROFILE_STARTS[signal_seed]),
        lookup=lookup,
    )
    initial = build_knn_edges(xy, int(COMMON["initial_knn"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        data_path,
        pressure=series["voltage_pu"],
        demand=series["net_p_mw"],
        node_xy=xy,
        edge_index=to_edge_index(gt, bidirectional=True),
        initial_edge_index=to_edge_index(initial, bidirectional=True),
        time_s=series["profile_index"] * int(COMMON["profile_step_seconds"]),
        voltage_pu=series["voltage_pu"],
        va_degree=series["va_degree"],
        p_load_mw=series["p_load_mw"],
        q_load_mvar=series["q_load_mvar"],
        p_sgen_mw=series["p_sgen_mw"],
        q_sgen_mvar=series["q_sgen_mvar"],
        net_p_mw=series["net_p_mw"],
        net_q_mvar=series["net_q_mvar"],
        profile_index=series["profile_index"],
        powerflow_converged=series["powerflow_converged"],
    )
    meta = {
        "name": "section_vi_fixed_electric",
        "dataset_id": scenario,
        "simbench_code": cfg["code"],
        "topology_source": str(topology_path.relative_to(PROJECT_ROOT)),
        "topology_regenerated": False,
        "signal_seed": signal_seed,
        "num_nodes": int(len(xy)),
        "num_gt_undirected_edges": int(len(gt)),
        "num_initial_undirected_edges": int(len(initial)),
        "node_ids": fixed["node_names"].astype(str).tolist(),
        "splits": make_splits(int(COMMON["timesteps"])),
        "compatibility_mapping": {
            "pressure": "voltage_pu",
            "demand": "net_p_mw",
        },
        "allowed_for_training": [
            "node_xy",
            "voltage_pu / pressure",
            "net_p_mw / demand",
        ],
        "posthoc_only": [
            "edge_index",
            "line / transformer / switch connectivity",
            "line impedances and lengths",
            "number of ground-truth edges",
        ],
        "generator": {
            "source": "SimBench + pandapower",
            "tie_lines": int(cfg["tie_lines"]),
            "profile_start": int(PROFILE_STARTS[signal_seed]),
            "profile_stride": int(COMMON["profile_stride"]),
            "timesteps": int(COMMON["timesteps"]),
            "converged_steps": int(series["powerflow_converged"].sum()),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"[DATA] {scenario} seed={signal_seed}: nodes={len(xy)} edges={len(gt)} "
        f"converged={int(series['powerflow_converged'].sum())}/{COMMON['timesteps']}"
    )


def tag(value: float | int) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def run_dgm(
    data_dir: Path,
    out_dir: Path,
    scenario: str,
    seed: int,
    *,
    alpha: float,
    lambda_q: float,
    candidate_k: int,
    dgm_k: int,
    learning_rate: float,
    coordinate_dim: int,
    epochs: int,
) -> None:
    if (out_dir / "forecast_metrics.json").exists() and (out_dir / "dgm_prob_mean.npy").exists():
        return
    data = np.load(data_dir / "data.npz")
    n = int(data["node_xy"].shape[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = (
        f"{scenario}_s{seed}_a{tag(alpha)}_lq{tag(lambda_q)}_ck{candidate_k}_"
        f"k{dgm_k}_lr{tag(learning_rate)}_ce{coordinate_dim}_e{epochs}"
    )
    cmd = [
        sys.executable,
        "scripts/experiments/run_train_synthetic.py",
        "--data_dir", str(data_dir),
        "--out_dir", str(out_dir),
        "--run_name", run_name,
        "--num_nodes", str(n),
        "--lags", str(COMMON["lags"]),
        "--prediction_window", str(COMMON["prediction_window"]),
        "--max_epochs", str(epochs),
        "--check_val_every_n_epoch", str(COMMON["check_every"]),
        "--batch_size", str(COMMON["batch_size"]),
        "--lr", str(learning_rate),
        "--k", str(dgm_k),
        "--ffun", str(COMMON["ffun"]),
        "--gfun", str(COMMON["gfun"]),
        "--lambda_graph", "0.0",
        "--use_complementary_loss",
        "--alpha_forecast", str(alpha),
        "--lambda_soft_pseudo_edge", str(lambda_q),
        "--lambda_stability", str(COMMON["lambda_stability"]),
        "--pseudo_score_mode", str(COMMON["prior"]),
        "--pseudo_use_static_target",
        "--pseudo_topk", str(dgm_k),
        "--pseudo_include_mst",
        "--select_best_by", "observable_val",
        "--assign_candidate_mask",
        "--candidate_mode", str(COMMON.get("candidate_mode", "q_dynamic")),
        "--candidate_k", str(candidate_k),
        "--candidate_signal_k", str(COMMON.get("candidate_signal_k", 4)),
        "--static_top_edges", "0",
        "--static_row_k", "2",
        "--static_symmetry", "mutual",
        "--static_include_mst",
        "--ensure_connected_consensus",
        "--export_soft_graph",
        "--save_soft_score",
        "--task_loss_type", str(COMMON["task_loss"]),
        "--task_huber_delta", str(COMMON["task_huber_delta"]),
        "--task_feature_combine", str(COMMON["task_feature_combine"]),
        "--task_feature_normalization", str(COMMON["task_feature_normalization"]),
        "--pressure_weight", str(COMMON["voltage_weight"]),
        "--demand_weight", str(COMMON["net_power_weight"]),
        "--coordinate_embedding_dim", str(coordinate_dim),
        "--seed", str(seed),
    ]
    with (out_dir / "run.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode:
        tail = (out_dir / "run.log").read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()[-60:]
        raise RuntimeError(f"Run failed: {out_dir}\n" + "\n".join(tail))


def validation_parts(run_dir: Path) -> tuple[float, float, float]:
    metrics = json.loads((run_dir / "forecast_metrics.json").read_text(encoding="utf-8"))
    val = metrics.get("best_observable_validation") or {}
    return (
        float(val["mse"]),
        float(val["soft_pseudo_edge_loss"]),
        float(val["stability_loss"]),
    )


def rank01(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=np.float64)
    ranks[order] = np.arange(len(array), dtype=np.float64)
    return ranks / max(1, len(array) - 1)


def add_fixed_selection_score(rows: list[dict]) -> None:
    for scenario in SCENARIOS:
        indices = [i for i, row in enumerate(rows) if row["scenario"] == scenario]
        task_rank = rank01([float(rows[i]["val_mse"]) for i in indices])
        q_rank = rank01([float(rows[i]["val_LQ_raw"]) for i in indices])
        stability_rank = rank01([float(rows[i]["val_stability"]) for i in indices])
        for local, index in enumerate(indices):
            rows[index]["task_rank"] = float(task_rank[local])
            rows[index]["q_rank"] = float(q_rank[local])
            rows[index]["stability_rank"] = float(stability_rank[local])
            rows[index]["selection_score"] = float(
                0.475 * task_rank[local]
                + 0.475 * q_rank[local]
                + 0.05 * stability_rank[local]
            )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def electric_prior_score(data_path: Path) -> np.ndarray:
    data = np.load(data_path)
    xy = torch.tensor(data["node_xy"], dtype=torch.float32)
    voltage = torch.tensor(data["voltage_pu"], dtype=torch.float32)
    net_power = torch.tensor(data["net_p_mw"], dtype=torch.float32)
    dist = torch.cdist(xy, xy)
    geometry = _offdiag_minmax(
        _geometry_rank_affinity(dist) * _nonmediated_geometry_affinity(dist)
    )
    voltage_smooth = _pressure_smooth_affinity(voltage)
    inverse_response = _inverse_source_target_affinity(net_power, voltage, ridge=0.05)
    return _offdiag_minmax(
        geometry * voltage_smooth * inverse_response
    ).detach().cpu().numpy()


def top_edges(score: np.ndarray, budget: int) -> set[tuple[int, int]]:
    pairs = [
        (float(score[u, v]), u, v)
        for u in range(score.shape[0])
        for v in range(u + 1, score.shape[0])
    ]
    pairs.sort(reverse=True)
    return {(u, v) for _, u, v in pairs[:budget]}


def proxy_diagnostics(data_root: Path, output_root: Path) -> list[dict]:
    rows = []
    for scenario in SCENARIOS:
        data_path = data_root / scenario / f"seed_{TUNING_SEED}" / "data.npz"
        data = np.load(data_path)
        xy = data["node_xy"]
        gt = undirected_edges(data["edge_index"])
        proxy = top_edges(electric_prior_score(data_path), len(gt))
        tp = len(proxy & gt)
        proxy_f1 = 2.0 * tp / max(1, len(proxy) + len(gt))
        for candidate_k in CANDIDATE_KS:
            mask = build_q_dynamic_candidate_mask(
                str(data_path),
                str(COMMON["prior"]),
                candidate_k,
                0.05,
                1.0,
                True,
            ).numpy()
            candidates = {
                (u, v)
                for u in range(len(xy))
                for v in range(u + 1, len(xy))
                if mask[u, v]
            }
            geometry = build_knn_candidate_mask(xy, candidate_k).numpy()
            geometry_edges = {
                (u, v)
                for u in range(len(xy))
                for v in range(u + 1, len(xy))
                if geometry[u, v]
            }
            rows.append({
                "scenario": scenario,
                "gt_edges": len(gt),
                "proxy_top_budget_f1": proxy_f1,
                "candidate_k": candidate_k,
                "candidate_edges": len(candidates),
                "candidate_recall": len(candidates & gt) / len(gt),
                "geometry_knn_recall": len(geometry_edges & gt) / len(gt),
            })
    write_csv(output_root / "proxy_candidate_diagnostics.csv", rows)
    return rows


def stage1(
    data_root: Path,
    output_root: Path,
    collect_only: bool,
) -> tuple[list[dict], dict[str, dict]]:
    rows = []
    for scenario in SCENARIOS:
        for alpha in ALPHAS:
            for lambda_q in LAMBDA_QS:
                run_dir = output_root / "stage1_weights" / scenario / f"a{tag(alpha)}_lq{tag(lambda_q)}"
                if not collect_only:
                    run_dgm(
                        data_root / scenario / f"seed_{TUNING_SEED}",
                        run_dir,
                        scenario,
                        TUNING_SEED,
                        alpha=alpha,
                        lambda_q=lambda_q,
                        candidate_k=5,
                        dgm_k=2,
                        learning_rate=1e-3,
                        coordinate_dim=4,
                        epochs=20,
                    )
                val_mse, val_lq, val_stability = validation_parts(run_dir)
                rows.append({
                    "scenario": scenario,
                    "alpha": alpha,
                    "lambda_Q": lambda_q,
                    "candidate_k": 5,
                    "dgm_k": 2,
                    "learning_rate": 1e-3,
                    "coordinate_dim": 4,
                    "epochs": 20,
                    "val_mse": val_mse,
                    "val_LQ_raw": val_lq,
                    "val_stability": val_stability,
                    "checkpoint_J_obs": observable_score(run_dir),
                    "run_dir": str(run_dir),
                })
    add_fixed_selection_score(rows)
    write_csv(output_root / "stage1_weights.csv", rows)
    selected = {
        scenario: min(
            (row for row in rows if row["scenario"] == scenario),
            key=lambda row: (float(row["selection_score"]), float(row["val_stability"])),
        )
        for scenario in SCENARIOS
    }
    return rows, selected


def stage2(
    data_root: Path,
    output_root: Path,
    weights: dict[str, dict],
    collect_only: bool,
) -> tuple[list[dict], dict[str, dict]]:
    rows = []
    for scenario, weight_row in weights.items():
        alpha = float(weight_row["alpha"])
        lambda_q = float(weight_row["lambda_Q"])
        for candidate_k in CANDIDATE_KS:
            for dgm_k in DGM_KS:
                for learning_rate in LEARNING_RATES:
                    for coordinate_dim in COORDINATE_DIMS:
                        name = f"ck{candidate_k}_k{dgm_k}_lr{tag(learning_rate)}_ce{coordinate_dim}"
                        run_dir = output_root / "stage2_structure" / scenario / name
                        if not collect_only:
                            run_dgm(
                                data_root / scenario / f"seed_{TUNING_SEED}",
                                run_dir,
                                scenario,
                                TUNING_SEED,
                                alpha=alpha,
                                lambda_q=lambda_q,
                                candidate_k=candidate_k,
                                dgm_k=dgm_k,
                                learning_rate=learning_rate,
                                coordinate_dim=coordinate_dim,
                                epochs=40,
                            )
                        val_mse, val_lq, val_stability = validation_parts(run_dir)
                        rows.append({
                            "scenario": scenario,
                            "alpha": alpha,
                            "lambda_Q": lambda_q,
                            "candidate_k": candidate_k,
                            "dgm_k": dgm_k,
                            "learning_rate": learning_rate,
                            "coordinate_dim": coordinate_dim,
                            "epochs": 40,
                            "val_mse": val_mse,
                            "val_LQ_raw": val_lq,
                            "val_stability": val_stability,
                            "checkpoint_J_obs": observable_score(run_dir),
                            "run_dir": str(run_dir),
                        })
    add_fixed_selection_score(rows)
    write_csv(output_root / "stage2_structure.csv", rows)
    selected = {
        scenario: min(
            (row for row in rows if row["scenario"] == scenario),
            key=lambda row: (float(row["selection_score"]), float(row["val_stability"])),
        )
        for scenario in SCENARIOS
    }
    return rows, selected


def final_runs(
    data_root: Path,
    output_root: Path,
    selected: dict[str, dict],
    collect_only: bool,
) -> list[dict]:
    rows = []
    for scenario, config in selected.items():
        for seed in FINAL_SEEDS:
            run_dir = output_root / "final_runs" / scenario / f"seed_{seed}"
            if not collect_only:
                run_dgm(
                    data_root / scenario / f"seed_{seed}",
                    run_dir,
                    scenario,
                    seed,
                    alpha=float(config["alpha"]),
                    lambda_q=float(config["lambda_Q"]),
                    candidate_k=int(config["candidate_k"]),
                    dgm_k=int(config["dgm_k"]),
                    learning_rate=float(config["learning_rate"]),
                    coordinate_dim=int(config["coordinate_dim"]),
                    epochs=60,
                )
            rows.append(
                evaluate_neural_decoder(
                    run_dir,
                    data_root / scenario / f"seed_{seed}",
                    scenario,
                    seed,
                    float(config["alpha"]),
                    float(config["lambda_Q"]),
                )
            )
    write_csv(output_root / "results_per_seed.csv", rows)
    return rows


def aggregate(
    rows: list[dict],
    selected: dict[str, dict],
    output_root: Path,
) -> list[dict]:
    metrics = ("precision", "accuracy", "recall", "f1", "mse", "mae", "joint_eval_0p5")
    output = []
    for scenario in SCENARIOS:
        values = [row for row in rows if row["scenario"] == scenario]
        config = selected[scenario]
        record = {
            "scenario": scenario,
            "n_seeds": len(values),
            "alpha": config["alpha"],
            "lambda_Q": config["lambda_Q"],
            "candidate_k": config["candidate_k"],
            "dgm_k": config["dgm_k"],
            "learning_rate": config["learning_rate"],
            "coordinate_dim": config["coordinate_dim"],
            "epochs": 60,
        }
        for metric in metrics:
            array = np.asarray([float(row[metric]) for row in values])
            record[f"{metric}_mean"] = float(array.mean())
            record[f"{metric}_std"] = float(array.std(ddof=1))
        output.append(record)
    write_csv(output_root / "results_aggregate.csv", output)
    return output


def copy_figures(output_root: Path) -> None:
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        source = (
            output_root
            / "final_runs"
            / scenario
            / f"seed_{FINAL_SEEDS[0]}"
            / "neural_consensus"
            / "graph_comparison"
            / "graph_compare_undirected.png"
        )
        shutil.copy2(
            source,
            figure_dir / f"{scenario}_gt_vs_neural_seed_{FINAL_SEEDS[0]}.png",
        )


def write_protocol(
    topology_root: Path,
    output_root: Path,
    selected: dict[str, dict],
) -> None:
    protocol = {
        "topologies": {
            scenario: str((topology_root / cfg["topology"]).relative_to(PROJECT_ROOT))
            for scenario, cfg in SCENARIOS.items()
        },
        "topology_regenerated_during_experiments": False,
        "tuning_seed": TUNING_SEED,
        "final_seeds": list(FINAL_SEEDS),
        "profile_starts": PROFILE_STARTS,
        "stage1_grid": {
            "alpha": list(ALPHAS),
            "lambda_Q": list(LAMBDA_QS),
            "epochs": 20,
        },
        "stage2_grid": {
            "candidate_k": list(CANDIDATE_KS),
            "dgm_k": list(DGM_KS),
            "learning_rate": list(LEARNING_RATES),
            "coordinate_dim": list(COORDINATE_DIMS),
            "epochs": 40,
        },
        "final_epochs": 60,
        "selection": (
            "GT-free within-scenario percentile score: "
            "0.475*rank(val_MSE) + 0.475*rank(raw val_LQ) "
            "+ 0.05*rank(val_stability)"
        ),
        "common_architecture": COMMON,
        "selected": selected,
        "neural_decoder": DECODER_CONFIG,
        "joint_evaluation": "0.5*(1/(1+test_MSE)) + 0.5*F1; post-hoc only",
    }
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topology_root",
        type=Path,
        default=PROJECT_ROOT / "results_section_vi" / "electric_topologies",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=PROJECT_ROOT / "results_section_vi" / "electric_final_datasets",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=PROJECT_ROOT / "results_section_vi" / "electric_final_experiments",
    )
    parser.add_argument("--collect_only", action="store_true")
    args = parser.parse_args()

    for scenario, cfg in SCENARIOS.items():
        topology_path = args.topology_root / cfg["topology"]
        if not topology_path.exists():
            raise FileNotFoundError(
                f"Missing fixed topology {topology_path}. Run "
                "the maintained topology generator described in docs/PAPER_REPRODUCTION.md first."
            )
        for seed in (TUNING_SEED, *FINAL_SEEDS):
            prepare_dataset(
                topology_path,
                args.data_root / scenario / f"seed_{seed}",
                scenario,
                seed,
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    proxy_diagnostics(args.data_root, args.output_root)
    _, weights = stage1(args.data_root, args.output_root, args.collect_only)
    _, selected = stage2(
        args.data_root,
        args.output_root,
        weights,
        args.collect_only,
    )
    rows = final_runs(
        args.data_root,
        args.output_root,
        selected,
        args.collect_only,
    )
    aggregate(rows, selected, args.output_root)
    copy_figures(args.output_root)
    write_protocol(args.topology_root, args.output_root, selected)
    print(f"[OK] Results: {args.output_root}")


if __name__ == "__main__":
    main()
