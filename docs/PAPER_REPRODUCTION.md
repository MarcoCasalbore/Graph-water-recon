# Paper scope and reproduction map

The maintained project covers the controlled synthetic study, Net3, Iowa feeder B and the eight supplementary water/electric layouts. Source pruning follows this scope. The broad historical C-Town/proxy/early grid-search entry points have been retired; shared modules imported by maintained experiments remain intact.

## Experiments and producers

| Paper material | Producer / entry points | Local artifact inputs |
| --- | --- | --- |
| Net3 and Iowa data | `scripts/benchmarks/generate_water_benchmark.py`, `generate_iowa240_feeder.py` | Upstream INP, full OpenDSS model and QSTS A/B/C; see DATASETS.md |
| Main real-derived training and evaluation | `scripts/benchmarks/run_paper_campaign.py` | `configs/paper_campaign/*final.json`, separate observation/evaluation directories |
| Four real-derived baselines | `scripts/benchmarks/run_fixed_graph_baselines.py` | `configs/paper_campaign/fixed_graph_baselines.json` |
| Controlled synthetic dataset | `scripts/validation/generate_relational_variant.py` | Graph/signal seeds and underlying homogeneous generator |
| Synthetic joint and node-only experiments | `scripts/validation/run_relational_campaign.py` | `results_section_v/campaign_v2/datasets/homogeneous_relational_g211_s1211` |
| Fixed topology baselines and matched forecasts | `scripts/validation/run_simple_topology_baselines.py`, `analyze_simple_topology_baselines.py` | Synthetic main runs, saved graphs and per-seed data |
| Alpha ablation | `scripts/validation_supplement/alpha_tradeoff_ablation/run_alpha_tradeoff_ablation.py` | Dense alpha baseline plus masked/unmasked variants; preserve the declared primary variant/decoder |
| Candidate size 3 versus 6 and sampled k | `scripts/validation_supplement/k_dgm_sensitivity_nominal_kc3/`, `k_dgm_sensitivity_candidate6_extended/` | Synthetic data and corresponding corrected per-seed sweeps |
| History/horizon and NodeMLP | `scripts/validation/run_lf_extreme_extension.py`, `search_lf_forecast_config.py`, `run_corrected_metric_sweeps.py` | L/F grid, boundary extensions, final selected configuration and node-only runs |
| Intermediate training graphs | `scripts/validation/run_graph_evolution.py`, `run_water_base_graph_evolution_fontcheck.py` | Training prefixes, saved graph samples, source epoch/display metadata |
| Synthetic tables and figures | `scripts/validation/assemble_section_v_final.py`, `finalize_posthoc_and_figures.py`, `create_signal_forecast_figure.py` | Corrected metrics, aligned predictions, baseline CSVs and figure input data |
| Training loss figures | `scripts/validation_supplement/figures/loss_evolution/` | Per-epoch histories from the retained main seeds |
| Four supplementary water layouts | `scripts/visualization/generate_water_base_candidates_section_vi.py`, `scripts/experiments/run_water_base_candidates_extended.py` | `results_section_vi/water_topologies/base_candidates`, datasets/final runs under `water_base_candidate_experiments` |
| Four supplementary electric layouts | `scripts/visualization/generate_diverse_electric_topology_candidates.py`, `scripts/experiments/run_electric_diverse_four.py` | Diverse topology candidates, `electric_diverse_datasets` and corresponding final runs |
| Edited publication layouts | `notebooks/paper_figures/` | Embedded drawing snapshots; new-run graphs use the benchmark renderers |

The names Section V/VI in directories come from an earlier manuscript structure. They are stable storage paths, not current paper section numbers. The real-derived benchmark learner and earlier synthetic/supplementary learner use distinct settings and must not be silently interchanged.

## Running the workflow

Start with the README for installation and self-contained software checks. Follow [DATASETS.md](DATASETS.md) to acquire sources and generate observations; then [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the main campaigns and exports.

For the synthetic study, generate the relational data first. The following commands launch full experiments, not help or smoke checks:

```bash
python scripts/validation/run_relational_campaign.py --group main --epochs_main 300
python scripts/validation/run_relational_campaign.py --group lh --epochs_sweep 120
python scripts/validation/run_lf_extreme_extension.py
```

Additional alpha/k runners similarly require their predecessor data/runs and can launch long sweeps. Keep each variant's config, decoder, per-seed metrics, predictions and graph artifacts; use the saved generation/training commands and seed records. The generic table assemblers also consume small earlier support tables, retained locally to avoid breaking their existing consumers.

## What Git contains

Code, configuration, tests, dataset provenance manifests, drawing-source notebooks and instructions are versioned. Downloaded inputs, generated observations, fitted models, metric tables, predictions, exported figures and the publication PDF/PPTX remain local and ignored. For a new scientific run, acquire/generate data and train. To recover the preserved paper inputs/results without access to the source PC, download the [complete workspace Release](https://github.com/MarcoCasalbore/Graph-water-recon/releases/tag/paper-workspace-2026-11-09) and follow [MIGRATION.md](MIGRATION.md). A clone alone does not include those historical artifacts.

Some hand-edited layouts embed graph snapshots. The source needed to compute and export the learned graphs is maintained separately. The cleanup preserves these snapshots and their inputs; it does not certify an exact numerical match to every manuscript cell or the agreement of every implementation setting with the prose.

## Local storage after cleanup

- `data/benchmarks/`: selected benchmark sources and observations.
- `final_water/`, `final_electric/`, `baseline_joint/`: final application campaigns and controls.
- `results_section_v/`: selected synthetic data/campaigns and a small set of support inputs.
- `results_section_v_final/`: corrected synthetic analyses, ablations, tables and curated figures.
- `results_section_vi/`: supplementary topologies, datasets, selected final runs and figures.
- `paper_artifacts/publication/`: editable deck, exported deck and reference manuscript; not versioned.
- `outputs/`: new runs and the small retained support inputs used by generic exporters.

The containing local workspace also has `other_research/` for original Rieti/CSA data and unrelated INP models. Those originals are preserved outside the paper area. Their Windows download metadata can be discarded without discarding the measurements.

No training or experimental campaigns were repeated during cleanup. Structural checks, source parsing, notebook AST comparison and hash verification protect the retained material. Native GPU/platform guarantees remain those stated in PLATFORMS.md.
