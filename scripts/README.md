# Scripts

These scripts are entry points around the reusable code in `src/`.

## Folders

- `experiments/`: training, grid search, inference, testing, and ranking.
- `water/`: EPANET/WNTR-specific dataset generation and graph comparison.
- `diagnostics/`: debugging and inspection utilities.
- `visualization/`: standalone plotting helpers.

Scripts may still contain hard-coded research settings from previous
experiments. Prefer moving stable parameters into `configs/` as the next cleanup
step.

For graph-reconstruction experiments, start from the topology-sensitive benchmark
protocol in docs/TOPOLOGY_SENSITIVE_BENCHMARK_PROTOCOL.md. Keep ground-truth
edges out of training, proxy construction, candidate masks, checkpoint selection,
decoder selection, and post-processing; use them only for post-hoc metrics and
figures.
