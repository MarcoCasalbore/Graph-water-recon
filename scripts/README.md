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
