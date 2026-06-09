# Minimal Observable Loss Experiments

These experiments test a deliberately small loss for synthetic water-like graph
reconstruction. Ground-truth edges are not used for training, checkpoint
selection, masks, or hyperparameter selection. They are used only after a run is
fixed, to report reconstruction metrics.

## Loss

The active training loss is:

```text
L = alpha * L_forecast
    + (1 - alpha) * (lambda_Q * L_Q + lambda_stability * L_stability)
```

where:

```text
L_Q = mean_i - sum_j Q_ij log(P_ij + eps)
```

`P_ij` is the DGM row probability of selecting node `j` from node `i`.
`Q_ij` is an observable pseudo-target built from coordinates and node time
series only.

The best current pseudo-target is `geo_global_dynamic_proxy`:

```text
B_demand: D(t+1) ~= D(t) B
B_cross:  P(t+1) ~= [P(t), D(t)] B
S_dynamic = minmax(pressure_smooth * max(|B_demand|, |B_cross_demand_to_pressure|))
Q_score = geo_affinity * S_dynamic
Q = row top-k normalization of Q_score, with optional coordinate MST support
```

This keeps the graph prior interpretable: nearby nodes are preferred, but
longer edges can survive when the observable signal dynamics justify them.

## Non-Oracle Selection

Runs are ranked by the observable validation objective:

```text
J_obs = alpha * MSE_val + (1 - alpha) * L_topology_val
```

The training code can now also save and restore the best checkpoint by
`J_obs` via:

```text
--select_best_by observable_val
```

This matters because choosing by `val_mse` alone can favor forecasting while
ignoring the topology objective.

## Grid V2: Proxy/Alpha Search

Output:

```text
outputs/minimal_loss_grid_v2_observable/
```

Setup:

```text
datasets: simple, water_base, water_bridges, water_loopy
proxy modes: pressure_corr, pressure_smooth, geo_global_dynamic_proxy
alpha: 0.05, 0.20, 0.40
k: 2
candidate_k: 2
epochs: 25
selection: observable_val
```

Best per dataset by observable `J_obs`:

| dataset | selected run | F1 | test MSE |
| --- | --- | ---: | ---: |
| simple | `simple_geo_global_dynamic_proxy_a0p40_q2_ck2_e25` | 0.7869 | 0.4178 |
| water_base | `water_base_geo_global_dynamic_proxy_a0p40_q2_ck2_e25` | 0.7619 | 0.4684 |
| water_bridges | `water_bridges_geo_global_dynamic_proxy_a0p40_q2_ck2_e25` | 0.7419 | 0.4954 |
| water_loopy | `water_loopy_geo_global_dynamic_proxy_a0p40_q2_ck2_e25` | 0.7123 | 0.4443 |

Conclusion: `geo_global_dynamic_proxy` dominates the simpler pressure-only
proxies. The observable objective is reasonably aligned with final topology F1,
without using ground truth for selection.

## Stability Ablation

Output:

```text
outputs/minimal_loss_grid_stability_ablation/
```

Setup:

```text
proxy mode: geo_global_dynamic_proxy only
alpha: 0.05, 0.20, 0.40
lambda_stability: 0.0, 1.0, 5.0
k: 2
candidate_k: 2
epochs: 25
selection: observable_val
```

Best per dataset by observable `J_obs`:

| dataset | selected run | F1 | test MSE |
| --- | --- | ---: | ---: |
| simple | `simple_geo_global_dynamic_proxy_a0p40_q2_ck2_stab1p000_e25` | 0.8033 | 0.4322 |
| water_base | `water_base_geo_global_dynamic_proxy_a0p40_q2_ck2_stab5p000_e25` | 0.7937 | 0.4742 |
| water_bridges | `water_bridges_geo_global_dynamic_proxy_a0p40_q2_ck2_stab1p000_e25` | 0.7419 | 0.5074 |
| water_loopy | `water_loopy_geo_global_dynamic_proxy_a0p40_q2_ck2_stab5p000_e25` | 0.7534 | 0.4254 |

Best F1 values observed after evaluation only:

| dataset | best evaluated run | F1 |
| --- | --- | ---: |
| simple | `simple_geo_global_dynamic_proxy_a0p40_q2_ck2_stab1p000_e25` | 0.8033 |
| water_base | `water_base_geo_global_dynamic_proxy_a0p05_q2_ck2_stab5p000_e25` | 0.8095 |
| water_bridges | `water_bridges_geo_global_dynamic_proxy_a0p05_q2_ck2_stab1p000_e25` | 0.7419 |
| water_loopy | `water_loopy_geo_global_dynamic_proxy_a0p05_q2_ck2_stab5p000_e25` | 0.7808 |

Conclusion: a small stability term is useful, especially for `simple`,
`water_base`, and `water_loopy`. `water_bridges` remains the hardest case and
does not improve beyond 0.7419 F1 in this ablation.

## Current Takeaways

- The clean loss should stay small: `forecast + Q`, with optional stability.
- The strongest current topology driver is not raw pressure correlation, but
  geometry times global observable dynamics.
- `alpha=0.40` is the best non-oracle choice in the current grids because it
  tends to minimize `J_obs` and gives good F1 after evaluation.
- `lambda_stability` in `{1.0, 5.0}` improves some benchmarks, but it should
  remain an ablated optional term, not another pile of priors.
- The next hard problem is `water_bridges`: it likely needs a better observable
  signal for sparse bridge edges, not more unrelated loss terms.
