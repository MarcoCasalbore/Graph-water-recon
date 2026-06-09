# Experiment Log

Short running log of topology-reconstruction experiments. Ground-truth edges
are used only for evaluation/diagnostics, never as training loss inputs unless
explicitly marked as an oracle diagnostic.

| Date | Experiment | Dataset | Result | Keep? | Note |
| --- | --- | --- | --- | --- | --- |
| 2026-06-05 | Edge score search v4 global VAR | simple_water_n50 | oracle score F1 0.9508 | yes | Best observed GT-free property so far. |
| 2026-06-05 | Edge score search v4 global VAR | er_n50_p004 | oracle score F1 0.7500 | yes | Dynamic VAR helps sparse ER. |
| 2026-06-05 | Edge score search v4 global VAR | geo_er_n50_p004 | oracle score F1 0.7414 | yes | Dynamic+geometry useful offline. |
| 2026-06-05 | DGM global VAR alpha=0.25 | simple_water_n50 | F1 0.6230 | yes | DGM improves but does not match Q oracle. |
| 2026-06-05 | DGM global VAR alpha=0.05 geo=0.1 candidate_k=4 | simple_water_n50 | F1 0.6557 | yes | Best DGM run so far in this batch. |
| 2026-06-05 | DGM global VAR alpha=0.05 | er_n50_p004 | F1 0.4643 | no | Much worse than direct Q oracle. |
| 2026-06-05 | DGM global VAR alpha=0.05 geo=0.1 candidate_k=8 | geo_er_n50_p004 | F1 0.4138 | no | Much worse than direct Q oracle. |
| 2026-06-05 | DGM distillation with Q candidate mask k=2 | simple_water_n50 | F1 0.7213; learned-vs-Q 0.7213 | yes | Best DGM run so far; candidate restriction helps. |
| 2026-06-05 | DGM distillation with Q candidate mask k=3 | simple_water_n50 | F1 0.6885; learned-vs-Q 0.7049 | no | Worse than Q-k=2. |
| 2026-06-05 | DGM distillation with Q candidate mask k=2 | er_n50_p004 | F1 0.5893; learned-vs-Q 0.6964 | yes | Big improvement over unrestricted DGM. |
| 2026-06-05 | DGM distillation with Q candidate mask k=2 | geo_er_n50_p004 | F1 0.5345; learned-vs-Q 0.6034 | yes | Big improvement over unrestricted DGM. |
| 2026-06-05 | Local bridge pseudo-target | geo_er_n50_p010 | F1 0.3413-0.3810 | no | Increased edge length but added many false positives. |
| 2026-06-05 | geo_partial_smooth static target k3/q3 | geo_er_n50_p010 | F1 0.4127 | yes | Best old geo_er_p010 DGM baseline. |
| 2026-06-05 | DGM k=3/top3 with Q candidate mask | simple_water_n50 | F1 0.6066 | no | More local choices made the hard graph drift; deleted output. |
| 2026-06-05 | Q candidate mask plus geometry MST support | simple_water_n50 | hard F1 0.7541 | superseded | Coherent target/support improved over 0.7213; deleted non-soft duplicate. |
| 2026-06-05 | Q candidate mask plus geometry MST support | er_n50_p004 | F1 0.4821 | no | MST bias is harmful on ER; deleted output. |
| 2026-06-05 | Q candidate mask plus geometry MST support | geo_er_n50_p004 | F1 0.5172 | no | Slightly worse than no-MST Q mask; deleted output. |
| 2026-06-05 | Alpha ablation alpha=0.01 with MST support | simple_water_n50 | F1 0.7377 | no | Too little forecast did not improve topology; deleted output. |
| 2026-06-05 | Soft-probability graph export from mean P_ij | simple_water_n50 | soft F1 0.7869; learned-vs-Q 0.8033 | yes | Best DGM result so far; DGM soft probabilities are better than hard consensus. |
| 2026-06-05 | Soft export without connected postprocess | simple_water_n50 | soft F1 0.7705; 4 components | no | Connectivity postprocess helps here; deleted output. |
| 2026-06-05 | Q target top3, DGM k=2 | simple_water_n50 | soft F1 0.7541 | no | Softer Q target diluted useful edge signal; deleted output. |
| 2026-06-05 | Soft score postprocess sweep, GT-selected diagnostic | simple_water_n50 | best F1 0.9180 | yes | Strong evidence that DGM soft score contains the right topology signal on water-like geometry; best selector used GT only diagnostically. |
| 2026-06-05 | Soft score postprocess sweep, GT-selected diagnostic | er_n50_p004 | best F1 0.5536 | yes | Weak on ER; best rule uses no geometry/MST penalty, suggesting ER is not aligned with water-like assumptions. |
| 2026-06-05 | Soft score postprocess sweep, GT-selected diagnostic | geo_er_n50_p004 | best F1 0.5517 | yes | Weak on geo-ER despite geometric generation; signal/topology relationship still under-identified. |
| 2026-06-05 | DGM distillation Q-dynamic k=2 | sbm_n50_b4_pin018_pout025 | saved soft score for postprocess | yes | Same clean loss as current best: pseudo-edge distillation only, no GT edges in training. |
| 2026-06-05 | Soft score postprocess sweep, GT-selected diagnostic | sbm_n50_b4_pin018_pout025 | best F1 0.5224 | yes | SBM communities alone do not provide enough observable edge evidence for this proxy. |
| 2026-06-05 | DGM distillation Q-dynamic k=2 | geo_sbm_n50_b4_pin018_pout025 | saved soft score for postprocess | yes | Same clean loss as current best: pseudo-edge distillation only, no GT edges in training. |
| 2026-06-05 | Soft score postprocess sweep, GT-selected diagnostic | geo_sbm_n50_b4_pin018_pout025 | best F1 0.5135 | yes | Even geometric SBM remains around chance-plus; benchmark needs stronger water-like signal/topology coupling. |
| 2026-06-05 | Added water_branch benchmark generator | synthetic generator | created base/bridges/loopy families | yes | District-based positions, local MST branches, sparse loops, and a small bridge budget. |
| 2026-06-05 | DGM distillation Q-dynamic k=2 + MST support | water_branch_n50_base | best diagnostic F1 0.8254 | yes | Clean water-like benchmark; reaches target range with soft postprocess. |
| 2026-06-05 | DGM distillation Q-dynamic k=2 + MST support | water_branch_n50_bridges | best diagnostic F1 0.7903 | yes | Harder bridge-heavy case; just below 0.80 but still much better than ER/SBM. |
| 2026-06-05 | DGM distillation Q-dynamic k=2 + MST support | water_branch_n50_loopy | best diagnostic F1 0.8082 | yes | Loopy case reaches target range; same F1 is also available with a connected postprocess setting. |
| 2026-06-05 | Final water benchmark suite, 4 DGM variants per dataset | simple + 3 water_branch datasets | best F1 0.9344; water-like F1 0.7903-0.8254 | yes | Final numerical/visual suite saved in outputs/final_suite; docs/FINAL_RESULTS.md summarizes key results. |
| 2026-06-09 | Minimal observable-loss grid | simple + 3 water_branch datasets | non-oracle selected F1: simple 0.7869, base 0.7619, bridges 0.7419, loopy 0.7123 | superseded | Selection uses `J_obs = alpha*MSE_val + (1-alpha)*L_topology_val`; GT only for final metrics. |
| 2026-06-09 | Minimal observable-loss grid with stability ablation | simple + 3 water_branch datasets | non-oracle selected F1: simple 0.8033, base 0.7937, bridges 0.7419, loopy 0.7534 | yes | Current clean result. Loss is `forecast + Q` with optional stability; best proxy is `geo_global_dynamic_proxy`. |
| 2026-06-09 | C-Town full-network first run | C-Town 396 nodes | clean fixed soft graph F1 0.5653; diagnostic sweep F1 0.5811 | yes | Full-network run uses pressure-correlation Q and geometric fallback input graph; diagnostic sweep is not a valid selector. |

## Cleanup

| Date | Action | Result | Note |
| --- | --- | --- | --- |
| 2026-06-05 | Removed obsolete output runs | outputs reduced from about 220M to 17M | Kept only current best, key baselines, ER/geoER references, and Q score search diagnostics. |
| 2026-06-05 | Kept only readable current benchmark outputs | outputs about 40M | Added SBM/geo-SBM current runs and postprocess sweeps; no obsolete failed directories left. |
| 2026-06-05 | Removed outdated ER/SBM output directories | outputs about 32M | Kept datasets and log entries, but removed low-value run artifacts after switching to water-like benchmarks. |
| 2026-06-05 | Final cleanup for GitHub preparation | outputs about 49M, only final_suite kept | Removed old run folders/checkpoints and ER/SBM generated datasets; generated outputs remain gitignored. |
| 2026-06-09 | Removed superseded minimal grid outputs | kept `minimal_loss_grid_v2_observable`, `minimal_loss_grid_stability_ablation`, C-Town, and historical `final_suite` | Old `minimal_loss_grid` and `minimal_loss_grid_deep` were replaced by observable-checkpoint runs. |
