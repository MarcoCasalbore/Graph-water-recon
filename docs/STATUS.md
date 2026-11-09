# Project status and audit

Audit date: 2026-11-09. Stage: **research implementation with locally completed experiments; reproducibility packaging and verification**. This is not a production deployment or evidence that every research hypothesis holds.

## Implemented stages

| Stage | State | Evidence / remaining work |
| --- | --- | --- |
| Controlled synthetic generation | Implemented | Water/electric examples plus extracted homogeneous/relational generators |
| Real-derived dataset adapters | Implemented | Net3/Fossolo/Pescara, Iowa feeder B and SimBench; manifests now catalogued |
| Reference-free main training | Implemented | Observation-only loader; train-fitted priors/statistics; split-local windows |
| Validation and checkpoint selection | Implemented | Normalized joint validation objective and per-seed checkpoints |
| Consensus / static topology | Implemented | Training-window probability statistics, consensus MLP and fixed decoder |
| Post-hoc evaluation | Implemented | Separate reference loader and all-pairs topology metrics |
| Main experimental campaigns | Existing local artifacts | Net3 and Iowa result tables each contain 20 seeds; full campaigns not rerun by this cleanup |
| Portable smoke and tests | Implemented | Self-contained fixture; CPU/GPU selection; no simulator/data download needed |
| Native Windows CPU validation | Passed in CI | Clean install, ten tests and full smoke campaign |
| Windows GPU and AMD validation | Outstanding | No corresponding GPU hardware verification |
| Scientific evidence consolidation | Outstanding | Historical summaries disagree on forecasting benefits and must be reconciled from aligned predictions |

## Audit findings and changes

- The previous README described older synthetic results while the main benchmark learner, configurations and generators were untracked. The entry point and protocol distinction are now explicit.
- Tests used an untracked local Pescara dataset. They now create temporary deterministic observations and cover boundaries, train-only statistics, priors, evaluation and a full CPU training/finalization cycle.
- A changed input could reuse an incompatible prior cache/checkpoint. The main runner now checks observation content, configuration and seed before reuse and skips completed finalization.
- Sampling now reapplies the candidate mask before top-k/Gumbel selection, preventing out-of-support edges when probabilities saturate.
- Observation shape, finite data, node IDs and split bounds were not validated. Invalid inputs now fail before training; NPZ observation files with edge arrays are rejected.
- Explicit CPU runs on GPU hosts used GPU accounting, and explicit unavailable GPU requests fell back silently. Device choice and accounting are now consistent and recorded.
- A synthetic entry point forced the Hopper-specific `TORCH_CUDA_ARCH_LIST=9.0+PTX`. It no longer overwrites the user's build environment.
- Reproduction sources were buried under result directories. They were extracted into `scripts/validation/` and `scripts/validation_supplement/` with portable project paths; historical output paths remain stable.
- Published result reports/figures are removed from the current Git tree without deleting local files. Generated results, downloads, environments and assistant/editor materials are ignored.

## Scientific cautions

The main code weights forecasting by `alpha` and structure by `1-alpha`. A local historical report called values near one a structural limit; that interpretation is reversed. The saved final settings are retained for reproduction, not retuned during repository cleanup.

Section V local summaries disagree on whether joint training improves true forecasting MSE. One corrected package attributes the earlier advantage to comparing Huber/scaled loss with element-wise MSE; another reports an advantage. This audit does not choose a favourable report or publish those numbers. Resolve the exact run set, prediction arrays, normalization and metric definition before making a paper claim.

The earlier synthetic observable-prior helpers can use full-series statistics in some modes, and some diagnostics rank choices using reference edges. They remain research/diagnostic code and do not inherit the main pipeline's leakage guarantees. Use the physically separated benchmark contract for new main experiments.

Candidate sparsity does not make the implementation fully sparse: dense pair matrices, electric pseudoinverses and stored per-window probabilities limit scalability. Existing tests and small networks do not validate large-network memory/runtime claims. The fixed-graph companion and historical scripts also retain their older resume semantics.

## Verification boundary

Local checks passed: ten self-contained tests; full fixture campaign with artifact/table exports on CPU; electric fixture on CUDA; one-epoch Net3 CPU and Iowa CUDA campaigns; fixed-graph self-check; homogeneous relational generation; source compilation and dependency consistency. These are software checks, not repeated final scientific campaigns. The [clean-install Linux and Windows CPU CI](https://github.com/MarcoCasalbore/Graph-water-recon/actions/runs/42620227193) also passed on the cleanup code commit. The local reference environment is Linux/WSL with NVIDIA and CPU execution. The full historical sweeps, native Windows simulators/GPU, AMD/ROCm and all optional PV/solar/traffic modes are not certified by passing the smoke tests. Direct dependency pins are not a full lockfile; exact historical results also need the original source data and compatible simulator/backend versions.
