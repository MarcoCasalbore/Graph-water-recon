# Professor Strategy Experiments

## Scope

These experiments test three suggestions independently:

1. inject node coordinates into the DGM representation through a lookup table;
2. train topology first and gradually increase the forecasting weight;
3. separate topology construction from forecasting and compare fixed graphs.

All DGM topology results use the GT-free static decoder:

```text
row-top-3(mean P), mutual symmetry, coordinate MST, no fixed edge budget
```

Ground truth is used only for final F1 or for the explicitly oracle fixed-graph
forecast diagnostic.

## Coordinate Lookup

The optional coordinate representation is an `nn.Embedding` lookup indexed by
node identity. Its initial table is deterministically built from normalized
coordinates:

```text
x, y, x^2, y^2, xy, sin(pi*x), sin(pi*y), 1
```

The eight-dimensional table is trainable by default and is concatenated with
the temporal node input before `pre_fc`. With dimension zero, the previous model
is unchanged.

## Curriculum

The epoch-dependent loss is:

```text
L(e) = alpha(e) * L_forecast
     + (1-alpha(e)) * (L_Q + lambda_stability * L_stability)
```

Two curricula were tested:

- linear: `alpha=0.05 -> 0.40` across 12 epochs;
- phased: hold `alpha=0.01` for four epochs, ramp to `0.40` over four epochs,
  then keep `0.40` until epoch 20.

Checkpoint selection always uses the final objective with `alpha=0.40`, so
validation scores from different epochs remain comparable.

## Controlled DGM Results

The main fair comparison uses 20 epochs, seed 42, `k=2`, `pseudo_topk=2`,
`candidate_k=2`, `lambda_stability=1`, and the same observable proxy and
decoder for every run.

| dataset | strategy | F1 | test MSE | J_obs |
| --- | --- | ---: | ---: | ---: |
| water base | baseline | 0.7692 | 0.5109 | 1.6060 |
| water base | coordinates | **0.7813** | 0.5017 | **1.5037** |
| water base | phased curriculum | 0.7597 | 0.5030 | 1.5986 |
| water base | coordinates + phased curriculum | **0.7874** | **0.4705** | 1.5610 |
| water bridges | baseline | 0.7368 | 0.5280 | 1.6199 |
| water bridges | coordinates | **0.7481** | 0.4991 | **1.5755** |
| water bridges | phased curriculum | 0.7328 | 0.5143 | 1.6104 |
| water bridges | coordinates + phased curriculum | 0.7424 | **0.4945** | 1.5821 |

Interpretation:

- coordinate lookup gives a small, consistent topology improvement and lowers
  the observable validation objective on both datasets;
- the tested curriculum does not consistently improve topology;
- curriculum often helps forecasting, but this does not imply better topology;
- coordinates plus curriculum reached F1 0.8125 in the shorter 12-epoch base
  run, but that gain did not persist as a stable cross-dataset result.

Coordinate embedding should be retained behind a flag and tested across seeds.
The current curriculum should not become the default.

## Fixed-Graph Forecast Experiment

An explicitly oracle diagnostic trained the same two-layer GCN forecaster for
60 epochs with different fixed graphs. `no_edges` means no cross-node edges;
PyG's GCN self-loops remain, so this is a shared node-wise temporal forecaster.

### Water Base

| graph | topology F1 | test MSE |
| --- | ---: | ---: |
| no cross-node edges | 0.0000 | **0.0045** |
| ground truth | 1.0000 | 0.1898 |
| observable proxy | 0.7907 | 0.2395 |
| random, same edge count | 0.0317 | 0.2794 |
| initial geometric proxy | 0.6585 | 0.3658 |

### Water Bridges

| graph | topology F1 | test MSE |
| --- | ---: | ---: |
| no cross-node edges | 0.0000 | **0.0049** |
| ground truth | 1.0000 | 0.1771 |
| random, same edge count | 0.0323 | 0.2245 |
| observable proxy | 0.7812 | 0.2638 |
| initial geometric proxy | 0.6211 | 0.3994 |

The true graph is better than the observable and geometric graphs on both
datasets, showing that physical topology can help when message passing is
forced. However, the no-edge model is dramatically better than every graph.

The current one-step task is therefore dominated by each node's own temporal
history. Message passing oversmooths useful local information, and forecast
MSE is not a reliable topology selector. The fact that a random graph beats the
observable proxy on bridge forecasting further confirms that forecast ranking
and topology ranking are not equivalent.

## Conclusions

1. **Keep coordinate embedding.** It is simple, observable and consistently
   positive, although the gain is modest and still needs multi-seed validation.
2. **Do not promote the current curriculum.** A topology-first phase changes
   optimization but does not reliably improve the final graph.
3. **Separate topology and forecasting conceptually.** The fixed-graph test
   supports learning/evaluating topology independently from the easy one-step
   forecast objective.
4. **Make forecasting topology-sensitive before using it as evidence.** Future
   tests should use longer horizons, masked-node prediction, perturbations or
   reduced local history. Otherwise a node-wise model solves the task without
   needing any graph.

## Reproduction

```bash
python scripts/experiments/run_professor_strategy_grid.py \
  --datasets water_base water_bridges \
  --strategies baseline coordinates \
  --epochs 20 \
  --out_root outputs/professor_strategy_constant_e20

python scripts/experiments/run_professor_strategy_grid.py \
  --datasets water_base water_bridges \
  --strategies curriculum coordinates_curriculum \
  --epochs 20 --alpha_start 0.01 --alpha_end 0.40 \
  --alpha_hold_epochs 4 --alpha_ramp_epochs 4 \
  --out_root outputs/professor_strategy_phased_e20

python scripts/diagnostics/oracle_graph_loss_test.py \
  --data_dir data/synthetic/water_branch_n50_base \
  --epochs 60 --check_every 5 --random_repeats 1 \
  --out_dir outputs/fixed_graph_forecast_base_e60
```
