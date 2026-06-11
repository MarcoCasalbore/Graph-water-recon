# Observable Proxy Methodology Study

## Scientific Question

Can an edge proxy built only from coordinates and observed node time series be
selected by observable validation quantities, while its selected graphs also
recover the hidden static topology?

The study keeps training and selection free of ground-truth edges. Ground truth
is loaded only after all observable rankings and cross-dataset choices are
frozen.

## Proxy Organization

Every candidate score has two interpretable evidence groups:

```text
S_dynamic(i,j) = evidence from pressure/demand time series
S_geometry(i,j) = evidence from node coordinates
Q_score(i,j) = fusion(S_dynamic(i,j), S_geometry(i,j))
```

Dynamic candidates include pressure smoothness and correlation, differenced
correlation, partial correlation, lag correlation, ridge-VAR coefficients,
demand-to-pressure cross-VAR coefficients, and innovation dependence.

Geometry candidates include exponential distance kernels at four scales and a
scale-free geometric neighbor-rank score. Fusion is deliberately simple:
dynamic only, product, square-root product, or a soft geometric gate.

The static graph is decoded without knowing its true edge count:

```text
directed candidates = row-top-k(Q_score)
undirected graph = union or mutual directed candidates
optional support = coordinate-only geometric MST
```

The full factorial contains 17,136 configurations per dataset:

```text
3 ridge values
x 17 dynamic scores
x 21 valid geometry/fusion choices
x 4 row-k values
x 2 symmetry rules
x 2 MST choices
```

## Observable Selection

For each candidate, the training interval is divided into two temporal halves.
The following quantities use no hidden edges:

```text
R_score = rank consistency of Q_score across the two halves
R_edge = Jaccard consistency of decoded edges across the two halves
R_forecast = validation BIC of next-pressure prediction

J_proxy = mean(percentile(R_score), percentile(R_edge), percentile(-BIC))
```

The pressure validator predicts each node's next pressure from its own and its
proposed neighbors' pressure and demand. BIC penalizes the additional regression
parameters. Predicting demand as an output was rejected because demand is
primarily an input and rewarded non-topological global correlations.

## Development Results

Each row reports the graph selected by observable criteria. F1 is computed only
after selection.

| dataset | local observable F1 | common observable F1 | post-hoc best F1 |
| --- | ---: | ---: | ---: |
| simple | 0.8909 | 0.8254 | 0.9244 |
| water base | 0.7939 | 0.7907 | 0.8455 |
| water bridges | 0.7812 | 0.7812 | 0.8348 |
| water loopy | 0.6707 | 0.7591 | 0.8451 |

The common development configuration is:

```text
dynamic = pressure_smooth
geometry = geometric row rank
fusion = product
row_k = 3
symmetry = mutual
include geometric MST = true
```

`product` and `sqrt_product` induce the same edge ordering and therefore the
same decoded graph.

## Held-Out Validation

The common configuration above was frozen before four new synthetic datasets
were generated with unseen random seeds. The first three reproduce base,
bridge-heavy, and loopy families; the fourth changes the pressure simulator to
autoregressive-neighbor dynamics.

| held-out dataset | frozen common proxy F1 | post-hoc best F1 |
| --- | ---: | ---: |
| base, seed 101 | 0.8031 | 0.8403 |
| bridges, seed 103 | 0.8000 | 0.8596 |
| loopy, seed 107 | 0.7143 | 0.8000 |
| autoregressive, seed 109 | 0.7717 | 0.8333 |

The method generalizes around F1 0.77 on average and reaches 0.80 on the two
tree/branch-oriented held-out networks. Loopy topology remains the main limit:
mutual row selection plus MST is naturally conservative about redundant cycle
edges.

## Literature Positioning

The methodology combines established ideas without claiming a new causal
identification theorem:

- Neural Relational Inference learns latent interactions from observed dynamics:
  https://arxiv.org/abs/1802.04687
- PCMCI motivates lag-aware conditional dependence for time-series structure:
  https://arxiv.org/abs/1702.07007
- Graphical-lasso network inference motivates sparse conditional-dependence
  scores and temporal stability: https://arxiv.org/abs/1703.01958
- Data-driven WDN system identification shows why input-output validation is
  useful when network parameters are unavailable: https://arxiv.org/abs/2207.05983
- Physics-informed topological signal processing for WDNs supports treating
  pressure/flow observations as topology-dependent graph signals:
  https://arxiv.org/abs/2505.07560

Our narrower contribution is an auditable construction-and-selection protocol
for a static pseudo-target `Q`, not a supervised edge estimator.

## Reproduction

```bash
python scripts/experiments/run_proxy_methodology_study.py \
  --out_dir outputs/proxy_methodology_pressure_forecast

python scripts/experiments/run_proxy_methodology_study.py \
  --datasets heldout_base heldout_bridges heldout_loopy heldout_ar \
  --dataset_dir heldout_base=data/synthetic/heldout_water_base_s101 \
  --dataset_dir heldout_bridges=data/synthetic/heldout_water_bridges_s103 \
  --dataset_dir heldout_loopy=data/synthetic/heldout_water_loopy_s107 \
  --dataset_dir heldout_ar=data/synthetic/heldout_water_ar_s109 \
  --out_dir outputs/proxy_methodology_heldout
```

## Consequences For DGM Training

The DGM loss remains minimal:

```text
L = alpha * L_forecast
    + (1-alpha) * (lambda_Q * L_Q + lambda_stability * L_stability)
```

The immediate change is in `Q` and in static decoding, not in adding loss
terms. Future admissible DGM runs must use `static_top_edges=0`; a fixed edge
budget copied from metadata is GT leakage. The observable decoder now uses
row-top-k, mutuality, and optional coordinate MST instead.

## Limits And Next Test

This analytical study evaluates the proxy directly. It does not yet prove that
a trained DGM will preserve the proxy ranking in its learned probabilities.
The next controlled experiment is therefore a small DGM ablation comparing the
old `geo_global_dynamic_proxy` with the frozen pressure-smooth/geometric-rank
target, while keeping the loss formula and observable decoder unchanged.

The subsequent controlled edge-substitution analysis is documented in
[PROXY_LANDSCAPE_DIAGNOSTIC.md](PROXY_LANDSCAPE_DIAGNOSTIC.md). It rejects a
pairwise conditional-gain term, finds a modest benefit from geometrically gated
demand VAR, and shows that the next missing evidence is likely graph-level.
