# Proxy Landscape Diagnostic

## Purpose

This is an explicitly oracle diagnostic. Ground-truth edges are used to build
controlled graph perturbations and to measure F1, never to define an observable
score or select a final method.

The question is deliberately elementary:

```text
If G_true is changed by one or more wrong edge substitutions,
does the observable proxy assign the changed graph a worse value?
```

No DGM is trained. Every graph has the same number of edges as the true graph,
which is allowed here only because the experiment studies the proxy landscape.

## Test

For each of eight development and held-out synthetic datasets, the diagnostic:

1. evaluates the true graph;
2. evaluates 200 random graphs at 1, 2, 4, 8, 16 and 32 edge substitutions;
3. enumerates every possible one-edge substitution around the true graph;
4. constructs the fixed-budget graph that directly maximizes each edge score.

There are approximately 71,000 to 84,000 one-edge substitutions per dataset.

For a graph `G` and observable edge score `S`, the tested value is:

```text
V(G; S) = mean_(i,j in G) log(S_ij + eps)
```

Higher is better, or equivalently `-V` is a proxy loss to minimize.

## Main Results

`geometry x pressure_smooth` behaves well but is not exact:

- degradation is monotonic at all six perturbation levels on all eight datasets;
- only about 2.8% of one-edge substitutions score better than the true graph;
- its direct fixed-budget maximizer has mean F1 approximately 0.757.

The proposed pairwise conditional pressure-forecast gain fails decisively:

- roughly 40-54% of wrong one-edge substitutions can beat the true graph;
- the direct maximizer has near-zero F1 on several datasets;
- adding it to geometry and smoothness usually damages the proxy.

This candidate should not be integrated into the DGM.

## Physics-Motivated Terms

Two additional properties were tested after inspecting the synthetic dynamics:

```text
demand(t) depends directly on normalized adjacency * demand(t-1)
pressure(t) depends on an inverse Laplacian smoothing operator
```

The inverse demand-to-pressure response was unstable and performed poorly. The
dense hydraulic response cannot be reliably inverted from these finite noisy
series using this simple estimator.

Demand VAR is more useful when geometrically gated. The combination

```text
geometry x pressure_smooth x demand_VAR
```

improves the direct-maximizer mean F1 from about 0.757 to 0.776 and reduces the
mean fraction of preferred wrong one-edge substitutions from about 2.8% to
2.2%. It improves `simple`, held-out base, held-out bridges and the
autoregressive held-out dataset, but does not consistently improve loopy
networks.

Representative direct-maximizer F1 values:

| dataset | geo x smooth | geo x smooth x demand VAR |
| --- | ---: | ---: |
| simple | 0.7705 | 0.8197 |
| water base | 0.7778 | 0.7778 |
| water bridges | 0.7581 | 0.7742 |
| water loopy | 0.7671 | 0.7260 |
| held-out base | 0.7619 | 0.8095 |
| held-out bridges | 0.7742 | 0.7903 |
| held-out loopy | 0.7260 | 0.6986 |
| held-out autoregressive | 0.7188 | 0.8125 |

## Interpretation

The current proxy has the correct broad landscape: progressively corrupted
graphs receive progressively worse values. Its remaining error is concentrated
in a small set of locally plausible edge substitutions.

This also shows why an F1 near one should not yet be expected from the DGM. Even
the graph that directly maximizes the analytical score is usually between F1
0.70 and 0.82. Training can only approximate the information present in `Q`; it
cannot recover distinctions that `Q` itself does not encode.

Demand VAR is worth retaining as a controlled candidate, but the next missing
property is likely graph-level rather than another pairwise similarity. In
particular, the remaining errors should be separated into:

- local edge versus a short indirect path;
- necessary tree/backbone edge versus redundant loop edge;
- intra-district edge versus inter-district connection.

The next analytical test should score the *marginal graph effect* of adding or
removing an edge, rather than assigning every edge an independent pairwise
score.

## Reproduction

```bash
python scripts/diagnostics/proxy_landscape_test.py \
  --repeats 200 \
  --out_dir outputs/proxy_landscape_physics_test
```

Detailed results are written to `landscape_summary.csv` and
`perturbation_curves.csv` in the output directory.
