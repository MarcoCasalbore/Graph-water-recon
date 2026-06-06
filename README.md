# Graph Water Recon

Graph topology reconstruction from node coordinates and node-level time series.
The motivating application is water distribution networks: each node has
pressure and demand signals, node coordinates are known, and pipe connections
are hidden. The goal is to recover a static topology as close as possible to the
ground-truth network.

The current method starts from a Dynamic Graph Module (DGM). Forecasting remains
the neural task, but the learned dynamic graph is distilled into a static graph
using observable geometry/signal proxies. Ground-truth edges are never used
inside the training loss.

![Best simple benchmark reconstruction](docs/figures/simple_best_graph_compare.png)

## Current Status

The clean experimental pipeline is now centered on controlled synthetic
water-like benchmarks:

- `simple_water_n50`
- `water_branch_n50_base`
- `water_branch_n50_bridges`
- `water_branch_n50_loopy`

The latest final suite reaches topology F1 in the `0.79-0.93` range on these
benchmarks. See [docs/FINAL_RESULTS.md](docs/FINAL_RESULTS.md).

Important caveat: current DGM runs are optimized strongly toward topology, not
pure forecasting. Deep non-graph baselines forecast better, which is useful
evidence for the next research step rather than a contradiction.

## Main Results

| Benchmark | Best run | Topology F1 | Components |
| --- | --- | ---: | ---: |
| simple water-like graph | `simple_a010_q2_mst_e80` | 0.9344 | 1 |
| water-branch base | `water_base_a002_q2_mst_e80` | 0.8254 | 1 |
| water-branch loopy | `water_loopy_a010_q2_mst_e80` | 0.8082 | 1 |
| water-branch bridges | `water_bridges_a005_q3_mst_e80` | 0.8065 | 1 |

Example water-like reconstructions:

![Water base reconstruction](docs/figures/water_base_best_graph_compare.png)

![Water loopy reconstruction](docs/figures/water_loopy_best_graph_compare.png)

![Water bridge-heavy reconstruction](docs/figures/water_bridges_best_graph_compare.png)

## Repository Layout

```text
.
├── configs/                  # Configuration notes and future config files
├── data/
│   └── synthetic/            # Small controlled synthetic benchmarks
├── docs/                     # Project structure, refactoring notes, results
├── legacy/                   # Historical material kept out of the main path
├── notebooks/                # Exploratory notebooks
├── scripts/
│   ├── diagnostics/          # Graph diagnostics and postprocess sweeps
│   ├── experiments/          # Training, final suite, summaries, baselines
│   ├── synthetic/            # Synthetic benchmark generators
│   ├── visualization/        # Plot helpers
│   └── water/                # EPANET/WNTR-specific utilities
└── src/
    ├── dataset/              # Dataset loaders
    ├── model/                # DGM and neural model definitions
    ├── train.py              # Baseline training utilities
    ├── train_dgm.py          # DGM loss/training/evaluation logic
    └── utils/                # Factories and helper utilities
```

Generated outputs and checkpoints are ignored by Git. The main generated result
directory is `outputs/final_suite`.

## Setup

Create a Python environment and install dependencies:

```bash
python -m venv .codex_venv
source .codex_venv/bin/activate
pip install -r requirements.txt
```

For PyTorch Geometric, installation can depend on your local CPU/GPU/CUDA
version. If `pip install -r requirements.txt` does not resolve the correct PyG
wheels, follow the official PyG install selector and then rerun the command.

If you are working on the water-network/EPANET side, inspect
`requirements-water-gpu.txt`; it contains the fuller GPU environment snapshot
used during these experiments.

## Standard Pipeline

The usual workflow is:

```bash
# 1. Generate or refresh synthetic water-like benchmarks.
python scripts/synthetic/generate_synthetic_water.py \
  --out_dir data/synthetic/water_branch_n50_base \
  --graph_type water_branch \
  --num_nodes 50 \
  --timesteps 720 \
  --water_districts 5 \
  --water_loops_per_district 2 \
  --water_bridge_edges 4 \
  --initial_knn 3 \
  --seed 31

# 2. Train one DGM run.
python scripts/experiments/run_train_synthetic.py \
  --data_dir data/synthetic/water_branch_n50_base \
  --out_dir outputs/example_run \
  --run_name example_run \
  --max_epochs 80 \
  --check_val_every_n_epoch 5 \
  --batch_size 32 \
  --k 2 \
  --static_top_edges 63 \
  --use_complementary_loss \
  --alpha_forecast 0.05 \
  --lambda_graph 0.0 \
  --lambda_soft_geo 0.0 \
  --lambda_soft_pseudo_edge 1.0 \
  --pseudo_score_mode geo_global_dynamic_proxy \
  --pseudo_use_static_target \
  --pseudo_topk 2 \
  --pseudo_include_mst \
  --assign_candidate_mask \
  --candidate_mode q_dynamic \
  --candidate_k 2 \
  --ensure_connected_consensus \
  --export_soft_graph \
  --save_soft_score

# 3. Compare learned and ground-truth graphs.
python scripts/water/compare_graphs_epanet.py \
  --learned outputs/example_run/dgm_learned_edges_softprob.pt \
  --npz data/synthetic/water_branch_n50_base/data.npz \
  --meta data/synthetic/water_branch_n50_base/meta.json \
  --outdir outputs/example_run/graph_comparison_softprob
```

## Generate Synthetic Benchmarks

The main generator is:

```bash
python scripts/synthetic/generate_synthetic_water.py \
  --out_dir data/synthetic/water_branch_n50_base \
  --graph_type water_branch \
  --num_nodes 50 \
  --timesteps 720 \
  --water_districts 5 \
  --water_loops_per_district 2 \
  --water_bridge_edges 4 \
  --initial_knn 3 \
  --seed 31
```

`water_branch` creates compact service districts, local tree-like branches,
short loops, and a small number of longer bridge edges.

## Run One DGM Experiment

```bash
python scripts/experiments/run_train_synthetic.py \
  --data_dir data/synthetic/water_branch_n50_base \
  --out_dir outputs/example_run \
  --run_name example_run \
  --max_epochs 80 \
  --check_val_every_n_epoch 5 \
  --batch_size 32 \
  --k 2 \
  --static_top_edges 63 \
  --use_complementary_loss \
  --alpha_forecast 0.05 \
  --lambda_graph 0.0 \
  --lambda_soft_geo 0.0 \
  --lambda_soft_pseudo_edge 1.0 \
  --pseudo_score_mode geo_global_dynamic_proxy \
  --pseudo_use_static_target \
  --pseudo_topk 2 \
  --pseudo_include_mst \
  --assign_candidate_mask \
  --candidate_mode q_dynamic \
  --candidate_k 2 \
  --ensure_connected_consensus \
  --export_soft_graph \
  --save_soft_score
```

The effective loss is:

```text
L = alpha * L_forecast + (1 - alpha) * L_topology
L_topology = L_soft_pseudo_edge
```

where the pseudo-edge target `Q` is built from node coordinates and time series,
not from ground-truth edges.

## Run the Final Suite

```bash
python scripts/experiments/run_water_benchmark_suite.py --epochs 80
python scripts/experiments/collect_final_results.py --out_root outputs/final_suite
```

This runs:

- DGM grid over `alpha`, `Q top-k`, and candidate support;
- diagnostic soft graph postprocess sweeps;
- non-graph deep forecasting baselines (`global_gru`, `node_mlp`);
- result collection into CSV/Markdown tables.

Main generated files:

```text
outputs/final_suite/topology_results.csv
outputs/final_suite/forecast_results.csv
outputs/final_suite/RESULTS.md
```

To run only the forecasting baselines and reuse already trained DGM runs:

```bash
python scripts/experiments/run_water_benchmark_suite.py --epochs 80 --skip_dgm
python scripts/experiments/collect_final_results.py --out_root outputs/final_suite
```

## Compare a Learned Graph

```bash
python scripts/water/compare_graphs_epanet.py \
  --learned outputs/example_run/dgm_learned_edges_softprob.pt \
  --npz data/synthetic/water_branch_n50_base/data.npz \
  --meta data/synthetic/water_branch_n50_base/meta.json \
  --outdir outputs/example_run/graph_comparison_softprob
```

This produces numerical metrics plus visual comparisons:

- `comparison_summary.json`
- `graph_compare_undirected.png`
- `graph_overlay_errors_undirected.png`
- true/false positive/negative edge CSVs.

## Water-Network Application

EPANET/WNTR utilities are kept under `scripts/water/`.

Typical entry points:

```bash
python scripts/water/epanet_data_ts_creator.py
python scripts/experiments/run_train.py
python scripts/water/compare_graphs_epanet.py
```

The synthetic benchmark is currently the controlled research environment. The
water-network application should be treated as the next transfer/evaluation
stage.

## Research Notes

- Ground-truth edges must not be used inside training losses or pseudo-targets.
- Ground truth is allowed for evaluation, diagnostic sweeps, and final metrics.
- Dynamic DGM graphs are aggregated into static learned topologies using
  consensus or mean soft probabilities.
- The best current topology results come from soft probabilities plus a
  length-aware static decoder.
- The next important step is replacing the diagnostic GT-selected decoder sweep
  with a validation rule that does not inspect ground-truth edges.
