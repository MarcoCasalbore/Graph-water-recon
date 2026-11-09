# Code map

| Area | Entry point / owner | Purpose |
| --- | --- | --- |
| Main pipeline | `src/paper_protocol.py` | Load observations; fit train-only statistics; learn dynamic graph and forecast; consolidate and decode |
| Campaign | `scripts/benchmarks/run_paper_campaign.py` | Expand grids, resume trials, select by mean validation joint objective, optionally evaluate reference topology |
| Main data adapters | `scripts/benchmarks/generate_*benchmark.py`, `generate_iowa240_feeder.py` | Separate observations and evaluation-only reference edges |
| Baselines | `scripts/benchmarks/run_fixed_graph_baselines.py` | GT-free fixed graphs and forecasting controls |
| Artifact exports | `scripts/benchmarks/assemble_final_artifacts.py`, `build_final_table.py` | Local summaries, figures and tables after evaluation |
| Consensus implementation | `scripts/diagnostics/train_consensus_learner.py` | Shared NumPy/PyTorch learner and decoder imported by the main pipeline |
| Earlier synthetic training | `src/train_dgm.py`, `src/model/dgm.py`, `src/dataset/dataset_epanet.py` | PyG research implementation; separate scientific settings |
| Synthetic validation | `scripts/validation/` | Section V data generation, controlled campaigns, baselines and corrected metrics |
| Supplementary sweeps | `scripts/validation_supplement/` | Alpha and neighbourhood-size analyses using local campaign artifacts |
| Tests | `tests/test_paper_protocol.py` | Temporary deterministic fixtures; no external dataset dependency |

Run scripts from the repository root. Keep source under `src/` or `scripts/`, configuration under `configs/`, generated files under `outputs/` or the historical ignored experiment directories. The synthetic validation sources were extracted from those output directories; their output paths remain stable to preserve existing experiments.

Do not route the main benchmark loader through the earlier NPZ loader: the old artifact combines reference edges and observations. The new contract physically separates them. Earlier PV, solar, wind and traffic code remains experimental; its optional TSL/Lightning dependencies and external datasets are not part of the validated entry point.
