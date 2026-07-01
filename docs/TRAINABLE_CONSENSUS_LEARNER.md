# Trainable Consensus Learner

This module turns the stochastic DGM outputs into one final graph without using
ground-truth edges during training.

## Inputs

The learner only sees quantities produced after DGM inference:

- mean edge probability across validation batches;
- standard deviation of the same probabilities;
- confidence and rank features derived from those probabilities.

It does not use ground-truth edges, coordinates, `T_obs`, pressure, demand, or
domain features as inputs or targets.

## Training Objective

The learner is a small MLP trained with a self-supervised objective:

```text
L_consensus =
    BCE(S, mean(P))
  + lambda_stability * mean(S * std(P))
  + lambda_degree * degree_penalty(S)
  + lambda_entropy * entropy(S)
```

where:

- `P` are the DGM probabilities;
- `S` is the learned consensus score;
- `mean(P)` keeps the learned score close to the DGM evidence;
- `std(P)` penalizes edges that change too much across batches;
- `degree_penalty` avoids extremely sparse or dense graphs;
- `entropy` encourages sharper final scores.

The learned score is then decoded with simple GT-free rules such as row-top-k,
mutual edges, and optionally a geometry MST only to guarantee connectivity.

## Evaluation

Ground truth is loaded only after training to compute post-hoc metrics:

- F1, precision, recall, TP, FP, FN;
- task error;
- compact evaluation score:

```text
Eval_gamma = gamma * TaskScore + (1 - gamma) * F1
TaskScore  = 1 / (1 + task_error)
```

For the current experiments, `gamma = 0.5` means equal weight to forecast quality
and topology reconstruction.

## Reference Framing

This block is best described as a trainable consensus layer with stability
regularization. It is conceptually close to stability selection and consensus
graph estimation: repeated stochastic graph samples are summarized into a final
stable graph without supervised edge labels.

