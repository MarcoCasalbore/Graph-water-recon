# SimBench Electric Benchmark Protocol

This benchmark mirrors the water setting with an electric distribution grid.

## Variable Mapping

The current training pipeline expects the names `pressure` and `demand`.
For compatibility, the electric dataset stores:

```text
pressure = voltage_pu
demand   = net_p_mw = p_load_mw - p_sgen_mw
```

The original electric variables are also saved in `data.npz`.

## Allowed During Training

The model, topology prior, and consensus step may use:

- bus coordinates `node_xy`;
- voltage time series `voltage_pu`;
- load/generation time series:
  - `p_load_mw`;
  - `q_load_mvar`;
  - `p_sgen_mw`;
  - `q_sgen_mvar`;
  - `net_p_mw`;
  - `net_q_mvar`;
- derived observable scores such as voltage smoothness, net-load coupling,
  correlations, lags, and conditional dependencies.

## Post-Hoc Only

The following information is ground truth and must not be used in training,
losses, candidate masks, decoder choice, checkpoint selection, or hyperparameter
choice:

- `edge_index`;
- SimBench line, transformer, and switch connectivity;
- line impedances and line lengths;
- number of ground-truth edges.

These are used only after training for F1, precision, recall, TP/FP/FN, AP, and
figures.

## Initial Datasets

The first three datasets are intentionally not too large:

| dataset | SimBench code | role |
|---|---|---|
| `electric_simple` | `1-LV-semiurb4--0-sw` | small LV grid |
| `electric_intermediate` | `1-LV-urban6--0-sw` | medium-small urban LV grid |
| `electric_medium` | `1-LV-rural2--0-sw` | larger LV grid |
| `electric_loopy` | `1-MV-rural--0-sw` | small MV grid with extra connectivity |
| `electric_mesh_simple` | `1-LV-semiurb4--0-sw` + 12 plausible tie-lines | controlled mesh-like LV grid |
| `electric_mesh_medium` | `1-LV-urban6--0-sw` + 16 plausible tie-lines | controlled mesh-like LV grid |

## Benchmark Intention

The target signal is voltage. The benchmark should not be solvable only from the
local history of the target node. As in the water case, topology-sensitive tests
should use longer forecast horizons, limited local lags, and diagnostics without
self-loops.

The electric expert prior should stay simple:

```text
T_obs = Combine(geometry, expert)
```

where `expert` can include voltage smoothness and load/generation-voltage
coupling.



The electric_mesh_* datasets are controlled SimBench-based variants: a small number of same-voltage, geographically plausible tie-lines are added before running pandapower. These tie-lines are part of the hidden ground-truth topology and remain post-hoc only, exactly like original SimBench lines.


## New Synthetic Electric Benchmarks

These datasets are not modified SimBench grids. They are generated from scratch with water-like spatial districts, local links, bridges, and loops, then simulated with pandapower power flow:

| dataset | nodes | role |
|---|---:|---|
| electric_synth_branch | 52 | controlled district/bridge electric grid |
| electric_synth_mesh | 52 | more meshed controlled electric grid |

They use the same allowed/post-hoc split: coordinates and nodal time series are observable; edge_index and line connectivity are post-hoc only.
