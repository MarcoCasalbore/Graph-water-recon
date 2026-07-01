# Topology-Sensitive Benchmark Protocol

This protocol exists to test graph reconstruction without leaking ground-truth edges into model development.

## Scientific Boundary

Ground-truth edges are never allowed in training losses, proxy construction, candidate masks, checkpoint selection, hyperparameter choice, post-processing, decoder choice, or edge-budget selection. They are used only post-hoc for final metrics and figures: F1, precision, recall, TP/FP/FN, AP, and graph visualizations.

Observable inputs allowed before evaluation include node coordinates, measured time series, train-split statistics, and domain-expert priors that do not inspect the target edge set.

## Benchmark Goal

A valid benchmark must make topology useful. The old one-step forecasting setup with long local history can be solved by node-local models, and in that regime a no-edge graph may match or beat the true graph. Such a setup is not informative for topology learning.

Use topology-sensitive settings before tuning DGM losses:

- longer forecast horizon;
- fewer local lags;
- optional removal or masking of target-node history in future diagnostics;
- fixed-graph diagnostics with GCN self-loops disabled;
- comparisons against no-edge, node-only, persistence, geometry, observable-proxy, random, and ground-truth oracle graphs.

The suite profile topology_sensitive currently uses lags=6, prediction_window=12, and disables self-loops in oracle_graph_loss_test.py. These values are defaults, not scientific constants; override them explicitly when running ablations.

## Observable Topology Prior

The prior should be formalized as a modular observable topology target:

    T_obs = Combine(
        F_geometry,
        F_observed_dynamics,
        F_domain_expert,
    )

F_geometry may include coordinates, distances, geometric ranks, geometric MST edges, and candidate plausibility.

F_observed_dynamics may include correlations, lagged dependence, VAR-like scores, conditional dependence, and other generic signals measured from observed series.

F_domain_expert is domain-specific and optional. For water it may include pressure smoothness, demand-pressure coupling, and hydraulic plausibility. For another domain it must be replaced by that domain's expert prior, not hard-coded into the DGM architecture.

## Normalization Requirement

Every component mixed into a training or prior score must have an explicit normalization story. This includes task loss, topology-prior loss, stability loss, geometry scores, dynamics scores, and expert-informed scores. Preferred options are rank or percentile normalization, robust min-max scaling, row normalization, and baseline-normalized losses.

Weights such as alpha and lambda should express scientific tradeoffs, not compensate for accidental numeric scale.

## Proxy Landscape Test

Before training DGM against a new prior, test whether the prior itself contains topology information:

1. score the true graph with the observable proxy;
2. replace true edges with non-edges and verify the proxy worsens;
3. construct the graph that directly maximizes the proxy under a GT-free decoder;
4. evaluate F1/AP only after construction.

This separates two questions: whether the proxy contains topology, and whether DGM can optimize it. If the analytic proxy maximum only reaches moderate F1/AP, improve the prior before expecting the learned model to exceed it by a large margin.

## Recommended Entry Points

Dry-run the topology-sensitive suite:

    python scripts/experiments/run_water_benchmark_suite.py --benchmark_profile topology_sensitive --datasets simple --variants a005_q2_mst --dry_run

Run the fixed-graph oracle diagnostic directly:

    python scripts/diagnostics/oracle_graph_loss_test.py --data_dir data/synthetic/water_branch_n50_base --lags 6 --prediction_window 12 --disable_self_loops

Train DGM on the same horizon/lag setup only after oracle diagnostics show that graphs matter.
