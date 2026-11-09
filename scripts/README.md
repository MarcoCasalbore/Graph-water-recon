# Entry points

Run from the repository root using the active environment's `python`.

- `benchmarks/`: supported starting point; independent data adapters, GT-free campaign, fixed-graph baselines and local exports.
- `validation/`: synthetic Section V generators and controlled campaigns extracted from result directories.
- `validation_supplement/`: sensitivity analyses, generally requiring previously generated local runs.
- `experiments/`: earlier PyG training, grids, inference and summaries.
- `synthetic/`, `electric/`: earlier-format data generators.
- `water/`: EPANET/WNTR utilities.
- `diagnostics/`: exploratory tools; some intentionally inspect GT or rank post-hoc results and must not be used for model selection.
- `visualization/`: plotting and topology exploration.

Start with the commands in [the README](../README.md) and [reproduction guide](../docs/REPRODUCIBILITY.md). Compilation is checked across the tree; full historical sweeps and every diagnostic are not guaranteed by the main smoke test.
