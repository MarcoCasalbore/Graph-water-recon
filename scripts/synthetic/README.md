# Synthetic Benchmarks

Synthetic water-like datasets use the same file interface as the EPANET
datasets:

- `pressure`: `[T, N]`
- `demand`: `[T, N]`
- `node_xy`: `[N, 2]`
- `edge_index`: hidden ground-truth topology, used only for evaluation
- `initial_edge_index`: geometry-only proxy graph used by the model input

Start with:

```bash
python scripts/synthetic/generate_synthetic_water.py
python scripts/experiments/run_train_synthetic.py
```
