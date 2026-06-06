# Refactoring Plan

## Pass 1: Project Boundaries

Status: in progress.

- Keep DGM and training behavior unchanged.
- Move runnable scripts into thematic folders.
- Ignore local environments, checkpoints, trials, and generated datasets.
- Document the new generic-first research direction.

## Pass 2: Training Cleanup

Split `src/train_dgm.py` into smaller behavior-preserving modules:

- forecasting and graph losses;
- CTOWN/EPANET batch reshaping helpers;
- train/evaluate loops;
- graph export utilities.

## Pass 3: Model Cleanup

Split `src/model/dgm.py` into:

- DGM module;
- graph sampling utilities;
- distance functions;
- temporal/graph backbones.

## Pass 4: Generic Benchmarks

Add synthetic graph reconstruction experiments before returning to water
networks:

- random geometric graphs;
- sparse planar or near-planar graphs;
- grid graphs;
- Barabasi-Albert graphs;
- diffusion/autoregressive node dynamics.
