# Minimal Observable Loss Experiments

These experiments test a deliberately small loss for synthetic water-like graph
reconstruction. Ground-truth edges are not used for training, checkpoint
selection, masks, or hyperparameter selection. They are used only after a run is
fixed, to report reconstruction metrics.

The main scientific question is not benchmark-specific:

```text
Can we build an observable, ground-truth-free objective whose low-loss
configurations also correspond to good static topology reconstructions?
```

Every ablation below should be read through that lens. Dataset-specific failures
are useful only insofar as they reveal where the observable objective is not yet
aligned with final topological quality.

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

## Long-Edge Q Ablation

Output:

```text
outputs/minimal_loss_grid_bridge_proxy/
```

Setup:

```text
dataset: water_bridges
proxy mode: geo_global_dynamic_bridge_proxy
alpha: 0.05, 0.20, 0.40
lambda_stability: 0.0, 1.0, 5.0
k: 2
candidate_k: 2
epochs: 25
selection: observable_val
```

This variant keeps the same minimal loss, but changes the observable static
target `Q`. It starts from `geo_global_dynamic_proxy` and splits each row target
into:

```text
Q = (1 - r_bridge) Q_local + r_bridge Q_bridge
```

where `Q_local` is the usual geometric-global-dynamic support and `Q_bridge`
uses the same observable global dynamic score, gated by relative geometric
length. No ground-truth edges are used in this proxy, candidate mask, checkpoint
selection, or hyperparameter selection.

Best by observable `J_obs`:

| dataset | selected run | F1 | test MSE |
| --- | --- | ---: | ---: |
| water_bridges | `water_bridges_geo_global_dynamic_bridge_proxy_a0p40_q2_ck2_stab1p000_e25` | 0.7419 | 0.4949 |

Best F1 values observed after evaluation only:

| dataset | best evaluated run | F1 |
| --- | --- | ---: |
| water_bridges | `water_bridges_geo_global_dynamic_bridge_proxy_a0p05_q2_ck2_stab5p000_e25` and `water_bridges_geo_global_dynamic_bridge_proxy_a0p40_q2_ck2_stab1p000_e25` | 0.7419 |

Conclusion: the long-edge channel substantially lowers the observable topology
loss scale (`L_Q` around 1.96 instead of 2.31-2.36), but does not improve final
F1 beyond the existing 0.7419 plateau. This is a useful negative result: a
simple long-edge mass in `Q` is not enough by itself. The next attempt should
target the export/selection bottleneck or make bridge evidence sharper, rather
than adding more loss terms.

## Candidate-Mask Ablation

Output:

```text
outputs/minimal_loss_grid_candidate_k_ablation/
```

Setup:

```text
dataset: water_bridges
proxy mode: geo_global_dynamic_proxy
alpha: 0.40
lambda_stability: 0.0, 1.0, 5.0
k: 2
candidate_k: 3, 4
epochs: 25
selection: observable_val
```

A post-hoc diagnostic first checked the observable `q_dynamic` candidate mask
capacity. Ground-truth edges were used only after the masks were fixed, to
measure recall:

| candidate_k | candidate edges | GT recall in mask |
| ---: | ---: | ---: |
| 2 | 79 | 52/62 = 0.8387 |
| 3 | 104 | 55/62 = 0.8871 |
| 4 | 130 | 56/62 = 0.9032 |
| 5 | 160 | 57/62 = 0.9194 |
| 8 | 250 | 59/62 = 0.9516 |

This showed that `candidate_k=2` caps the recoverable edge set, but increasing
the mask is not sufficient by itself.

Best by observable `J_obs`:

| dataset | selected run | F1 | test MSE |
| --- | --- | ---: | ---: |
| water_bridges | `water_bridges_geo_global_dynamic_proxy_a0p40_q2_ck3_stab1p000_e25` | 0.6935 | 0.4801 |

Best F1 values observed after evaluation only:

| dataset | best evaluated run | F1 |
| --- | --- | ---: |
| water_bridges | `water_bridges_geo_global_dynamic_proxy_a0p40_q2_ck3_stab0p000_e25` | 0.7419 |

Conclusion: a larger observable candidate mask improves forecast and lowers
`J_obs`, but does not improve topology under non-oracle selection. In fact,
`J_obs` prefers a run with lower F1. The bottleneck is now more specific:
additional candidate capacity must be paired with a sharper observable topology
signal or a better GT-free export rule.

## Observable Objective Audit

Output:

```text
outputs/observable_objective_audit/
outputs/observable_objective_audit_by_dataset/
```

This audit is post-hoc and diagnostic. It reads completed grid results and asks
which observable validation criterion would have selected a run close to the
best final F1. Ground-truth F1 is used only after runs are complete, to measure
alignment; it is not used to select any reported run.

Aggregated by dataset, the current audit shows:

| dataset | best audited observable selector | selected F1 | best evaluated F1 | regret |
| --- | --- | ---: | ---: | ---: |
| simple | `j_val_current` | 0.8033 | 0.8033 | 0.0000 |
| water_base | `val_lq` / `val_topology` | 0.8095 | 0.8095 | 0.0000 |
| water_bridges | `balanced_norm_mse_lq_alpha` | 0.7419 | 0.7419 | 0.0000 |
| water_loopy | `val_lq` / `val_topology` | 0.7671 | 0.7808 | 0.0137 |

Key diagnostic result:

- `val_mse` is often poorly aligned with topology and can be anti-aligned.
- `L_Q` / `val_topology` is more aligned on several water-like settings.
- The current `J_obs = alpha*MSE_val + (1-alpha)*L_topology_val` is useful, but
  can still overweight forecast when the candidate/search space changes.
- The next objective should keep a small form, but should treat forecasting as
  an observability guard rather than letting low MSE dominate topology.

## Final Hyperparameter Grid

Output:

```text
outputs/final_observable_hyperparam_grid/
```

Setup:

```text
datasets: simple, water_base, water_bridges, water_loopy
proxy mode: geo_global_dynamic_proxy
alpha: 0.05, 0.20, 0.40, 0.70
k / pseudo_topk: 1, 2, 3
candidate mask: q_dynamic, candidate_k = max(k, 2)
lambda_stability: 0.0, 1.0
epochs: 25
selection: observable_val
```

Every run used the same minimal observable loss:

```text
L = alpha * L_forecast
    + (1 - alpha) * (lambda_Q * L_Q + lambda_stability * L_stability)

L_Q = mean_i -sum_j Q_ij log(P_ij + eps)
lambda_Q = 1.0
Q = geo_global_dynamic_proxy from coordinates + observed node time series
J_obs = alpha*MSE_val + (1-alpha)*L_topology_val
```

Best per dataset by observable `J_obs`:

| dataset | selected run | F1 | J_obs | test MSE |
| --- | --- | ---: | ---: | ---: |
| simple | `simple_geo_global_dynamic_proxy_a0p70_q1_ck2_stab0p000_e25` | 0.8197 | 0.5091 | 0.3282 |
| water_base | `water_base_geo_global_dynamic_proxy_a0p70_q1_ck2_stab1p000_e25` | 0.7937 | 0.5313 | 0.3657 |
| water_bridges | `water_bridges_geo_global_dynamic_proxy_a0p70_q1_ck2_stab0p000_e25` | 0.7258 | 0.5147 | 0.3521 |
| water_loopy | `water_loopy_geo_global_dynamic_proxy_a0p70_q1_ck2_stab0p000_e25` | 0.7808 | 0.4273 | 0.3098 |

Best F1 values observed after evaluation only:

| dataset | best evaluated run | F1 | J_obs |
| --- | --- | ---: | ---: |
| simple | `simple_geo_global_dynamic_proxy_a0p20_q1_ck2_stab0p000_e25` / `simple_geo_global_dynamic_proxy_a0p40_q1_ck2_stab0p000_e25` | 0.8361 | 0.9123 / 0.7728 |
| water_base | `water_base_geo_global_dynamic_proxy_a0p20_q1_ck2_stab0p000_e25` / `water_base_geo_global_dynamic_proxy_a0p20_q1_ck2_stab1p000_e25` | 0.8095 | 0.7691 / 0.7412 |
| water_bridges | `water_bridges_geo_global_dynamic_proxy_a0p40_q2_ck2_stab0p000_e25` | 0.7742 | 1.6130 |
| water_loopy | several `k=1` runs | 0.7808 | 0.4273-0.6290 |

Hyperparameter influence, averaged over the 96 final runs:

| factor | setting | mean F1 | best F1 | mean J_obs |
| --- | --- | ---: | ---: | ---: |
| alpha | 0.05 | 0.7410 | 0.8033 | 1.9400 |
| alpha | 0.20 | 0.7447 | 0.8361 | 1.7143 |
| alpha | 0.40 | 0.7434 | 0.8361 | 1.3979 |
| alpha | 0.70 | 0.7357 | 0.8197 | 0.9056 |
| k | 1 | 0.7754 | 0.8361 | 0.7122 |
| k | 2 | 0.7511 | 0.7869 | 1.5444 |
| k | 3 | 0.6970 | 0.7705 | 2.2117 |
| lambda_stability | 0.0 | 0.7375 | 0.8361 | 1.4903 |
| lambda_stability | 1.0 | 0.7448 | 0.8197 | 1.4886 |

Conclusions from this grid:

- `k` is the strongest tested lever. `k=1` has the best mean F1 and best
  observable objective; `k=3` consistently hurts.
- `alpha=0.20` and `alpha=0.40` give the best evaluated F1. `alpha=0.70`
  minimizes `J_obs` because forecast MSE improves, but it is not the best
  topology setting on average.
- `lambda_stability=1.0` slightly improves mean F1, but its effect is much
  smaller than `k`.
- The current `J_obs` is useful but not fully aligned: it tends to select
  `alpha=0.70, k=1`, while the best evaluated topology often prefers
  `alpha=0.20/0.40`, with `k=1` generally and `k=2` on `water_bridges`.
- The next thing to push is not a larger loss. It is a better GT-free selection
  rule: topology-first, with forecast as a guardrail.

## Current Takeaways

- The clean loss should stay small: `forecast + Q`, with optional stability.
- The strongest current topology driver is not raw pressure correlation, but
  geometry times global observable dynamics.
- `alpha=0.40` is the best non-oracle choice in the current grids because it
  tends to minimize `J_obs` and gives good F1 after evaluation.
- `lambda_stability` in `{1.0, 5.0}` improves some benchmarks, but it should
  remain an ablated optional term, not another pile of priors.
- The next hard problem is still `water_bridges`: a small long-edge channel in
  `Q` is GT-free and interpretable, but it does not break the 0.7419 F1 plateau.
- Increasing candidate-mask capacity alone also does not break the plateau; it
  can even make the observable validation objective less aligned with final F1.
- The research target is now objective alignment: design a GT-free validation
  and training objective whose minima are consistently good topologies, rather
  than optimizing forecast quality and hoping topology follows.
- The final grid suggests using `k=1` as the default first choice, keeping
  `k=2` as a secondary candidate, and avoiding `k=3` for now.
- `alpha=0.20/0.40` should be the main topology-oriented range; `alpha=0.70`
  is useful diagnostically but too forecast-heavy for topology claims.
