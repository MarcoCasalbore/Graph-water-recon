# Running and reproducing the project

Run from the repository root, with the chosen virtual environment activated. The README smoke path needs only `requirements-core.txt`; generation/export needs `requirements-benchmarks.txt`; synthetic Section V training needs `requirements.txt`.

## Main benchmark campaign

After [generating the datasets](DATASETS.md), run the saved final settings:

```bash
python scripts/benchmarks/run_paper_campaign.py --observation-dir data/benchmarks/water/net3/observations --evaluation-dir data/benchmarks/water/net3/evaluation_only --output-dir outputs/net3_final --spec configs/paper_campaign/water_net3_final.json --seeds 100:120 --device auto
python scripts/benchmarks/run_paper_campaign.py --observation-dir data/benchmarks/electric/iowa240_feeder_b/observations --evaluation-dir data/benchmarks/electric/iowa240_feeder_b/evaluation_only --output-dir outputs/iowa_final --spec configs/paper_campaign/electric_iowa_final.json --seeds 100:120 --device auto
```

This runs 20 seeds per dataset at 60 epochs, not a quick test. A one-epoch wiring check uses `--spec configs/smoke.json --seeds 0 --device cpu` and a new output directory. Re-running final settings reproduces the selected configuration, not the historical coarse-to-fine search. To reproduce a search stage, use its grid JSON in `configs/paper_campaign/` and document the tuning seed set and predecessor stage.

`J = alpha * (forecast_MSE / initial_forecast_scale) + (1-alpha) * (structural_CE / initial_structural_scale)`.

Both scales are fixed from the initial training loss. Validation selects checkpoints and the mean objective across requested seeds selects configurations. `alpha` near one means **forecast-dominant**, not structure-dominant. Consensus is fitted on training-window probabilities and decoded using a fixed mutual row-k rule with a coordinate MST. The post-hoc score `0.5/(1+MSE) + 0.5*F1` is descriptive and does not select the model. Main benchmark training never receives the evaluation directory.

## Outputs and resume

```text
outputs/<campaign>/
  gt_free_trials.csv          model-selection inputs
  selection_gt_free.json      selected configuration and validation rule
  selected_seed_results.csv   selected seeds, optionally with post-hoc metrics
  selected_aggregate.json     mean/std/count
  trials/<config-hash>/seed_<n>/
    run_identity.json         observation content hash, full config and seed
    environment.json          Python/platform/Torch/backend versions
    config.json, history.json
    best.pt, last.pt           trusted local checkpoints
    prior_q_cache.pt           deterministic observable prior cache
    metrics_gt_free.json      validation/runtime/forecast metrics
    probability_mean.npy, probability_std.npy, omega.npy, learned_edges.csv
    metrics_posthoc.json      only with reference evaluation
```

Identical completed runs are reused; interrupted runs restore optimizer and RNG states from `last.pt`. Changing the input content, manifest, configuration or seed requires a new output directory. Earlier checkpoints without an identity record are refused rather than assumed compatible. Loading checkpoint files uses PyTorch pickle support: only resume checkpoints you trust. Keep software versions fixed during a resume. Source changes are not automatically hashed; use a new output directory after changing training code.

For controlled comparisons use the same split, seed set, history/horizon, preprocessing and explicitly stated loss. GPU kernels and changed software versions need not produce bitwise-identical results. Record `python -m pip freeze` and the Git commit alongside a scientific run. The historical reference environment was Linux/WSL, Python 3.11.14, Torch 2.8.0+cu129 and NumPy 2.3.3; the old NVIDIA freeze is retained separately.

## Baselines and local exports

```bash
python scripts/benchmarks/run_fixed_graph_baselines.py --self_check
python scripts/benchmarks/run_fixed_graph_baselines.py --config configs/paper_campaign/fixed_graph_baselines.json --output_dir outputs/fixed_graph_baselines
python scripts/benchmarks/assemble_final_artifacts.py --observation-dir data/benchmarks/water/net3/observations --evaluation-dir data/benchmarks/water/net3/evaluation_only --campaign-dir outputs/net3_final
python scripts/benchmarks/assemble_final_artifacts.py --observation-dir data/benchmarks/electric/iowa240_feeder_b/observations --evaluation-dir data/benchmarks/electric/iowa240_feeder_b/evaluation_only --campaign-dir outputs/iowa_final
python scripts/benchmarks/build_final_table.py --entry Water data/benchmarks/water/net3/observations outputs/net3_final --entry Electric data/benchmarks/electric/iowa240_feeder_b/observations outputs/iowa_final --output outputs/final_table.csv
```

The fixed-graph script is an older companion runner; its checkpoint/resume checks differ from the main campaign. Use a fresh output directory after changing its inputs or code. Rendering requires an evaluated campaign and `requirements-benchmarks.txt`. On headless Linux, set `MPLBACKEND=Agg` if needed.

For smoke exports, substitute the README observation/evaluation/campaign paths in `assemble_final_artifacts.py`. All generated results stay local and ignored.

## Earlier synthetic validation

Generate the relational fixture as described in [datasets](DATASETS.md), install the full requirements, then:

```bash
python scripts/validation/run_relational_campaign.py --group main --epochs_main 300
python scripts/validation/run_relational_campaign.py --group lh --epochs_sweep 120
python scripts/validation/run_corrected_metric_sweeps.py --help
```

The main campaign fixes graph/signal seeds 211/1211 and training seeds 42–46; sweeps use 42–44. Output remains in `results_section_v/` for compatibility. Supplementary analyzers expect those artifacts and should not be run on an empty clone before generating the corresponding campaigns. Some historical summaries called a scaled/Huber objective MSE; use aligned saved predictions for actual MSE/MAE comparisons. Do not combine these numbers with the main protocol's normalized MSE.

## Paper-specific sources

See [PAPER_REPRODUCTION.md](PAPER_REPRODUCTION.md) for alpha, candidate-size/k, temporal and supplementary-network producers. Table exports require `tabulate`, declared in `requirements-benchmarks.txt`. Drawing notebooks are saved without execution outputs and write into isolated per-notebook output directories.
