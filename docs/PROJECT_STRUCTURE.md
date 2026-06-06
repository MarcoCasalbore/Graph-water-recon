# Project Structure

The project is being moved from a water-network-specific prototype toward a
generic graph reconstruction framework with water networks as a concrete
application.

## Conceptual Boundaries

### Generic Core

Code in the generic core should not depend on EPANET, WNTR, pressure, demand,
or hydraulic assumptions.

Expected responsibilities:

- graph/topology learning modules;
- temporal forecasting backbones;
- synthetic graph and time-series benchmarks;
- topology recovery metrics;
- training loops and reusable losses.

### Water Application

Water-specific code is kept under `scripts/water` and `src/dataset` for now.
It may depend on EPANET/WNTR concepts such as `.inp` files, pressure, demand,
and hydraulic network ground truth.

Expected responsibilities:

- generate time series from EPANET/WNTR;
- load pressure/demand datasets;
- compare learned graphs with EPANET ground truth;
- run water-specific experiments.

## Current Layout

```text
scripts/water/
  epanet_data_ts_creator.py
  compare_graphs_epanet.py
  compare_consensus_graphs_epanet_configurable.py
  node_pressure_test.py

scripts/experiments/
  run_train.py
  run_grid_search.py
  run_grid_search_water_fixed.py
  rank_grid_results_topology.py
  run_inference.py
  run_test.py

scripts/diagnostics/
  inspect_batch_edge_stability.py

scripts/visualization/
  print_graph.py

src/dataset/
  dataset.py
  dataset_Solar.py
  dataset_epanet.py

src/model/
  dgm.py
  gcn1d.py
  gcn1d_new.py
  gconvrnn.py
  dcrnn.py
  export_graph_stats.py
```

## Next Refactoring Pass

The high-value cleanup target is to split `src/train_dgm.py` into:

- `src/training/loops.py`
- `src/training/losses.py`
- `src/evaluation/graph_metrics.py`
- `src/evaluation/export_graph.py`

Then split `src/model/dgm.py` into:

- `src/model/dgm.py` for the actual DGM module;
- `src/model/graph_sampling.py` for edge sampling / KNN / candidate masks;
- `src/model/backbones.py` for temporal and graph backbones.

This should be done only after a small smoke test is available, so that the
cleanup remains behavior-preserving.
