# Graph Water Recon / W-DGM

Infer a hidden network topology from node coordinates and two observed time series while forecasting future observations. Water networks use pressure/demand; electric networks use voltage/active power. This is research software with two distinct experimental protocols.

**Start here:** the benchmark pipeline in `src/paper_protocol.py` and `scripts/benchmarks/`. It uses chronological 75/10/15 splits, split-local windows (history 6, horizon 12), train-fitted normalization and priors, GT-free model selection, training-window consensus and a fixed decoder. Reference edges are read only in the optional post-hoc evaluation phase.

The earlier synthetic Section V experiments use `src/train_dgm.py`, `scripts/experiments/` and `scripts/validation/`. They have different losses, splits and decoders. Their metrics are not interchangeable with the benchmark protocol. See [project status](docs/STATUS.md).

## Quick start without external datasets or a GPU

Use Python 3.11 (the reference environment). Python 3.12 is used by some newer platform-specific PyTorch packages; it has not been validated here.

```bash
git clone https://github.com/MarcoCasalbore/Graph-water-recon.git
cd Graph-water-recon
python -m venv .venv
```

Activate on Linux/WSL:

```bash
source .venv/bin/activate
```

Activate in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install CPU PyTorch, then the small core environment:

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-core.txt
python -m unittest discover -s tests -v
python scripts/benchmarks/generate_smoke_benchmark.py --output-dir outputs/smoke/data
python scripts/benchmarks/run_paper_campaign.py --observation-dir outputs/smoke/data/observations --evaluation-dir outputs/smoke/data/evaluation_only --output-dir outputs/smoke/run --spec configs/smoke.json --seeds 0 --device cpu
```

The fixture is a software check, not a physical water/electric benchmark. The last command trains, selects a checkpoint, constructs consensus, decodes a graph and evaluates it. Read `outputs/smoke/run/selected_aggregate.json` for the result. Repeating the same command reuses completed runs. Omit `--evaluation-dir` for a completely reference-free run.

For GPUs, install the appropriate PyTorch build **before** the requirements file and pass `--device cuda` (also the PyTorch API name for AMD ROCm). `--device auto` chooses a GPU when available; an explicit unavailable GPU fails with an explanation. See the [platform matrix](docs/PLATFORMS.md).

## Installation choices

| File | Purpose |
| --- | --- |
| `requirements-core.txt` | Main learner, tests and smoke fixture: NumPy and PyTorch |
| `requirements-benchmarks.txt` | Adds EPANET/WNTR, OpenDSS, SimBench and plotting/data tools |
| `requirements.txt` | Adds the earlier synthetic/PyG research path |
| `requirements-water-gpu.txt` | Historical NVIDIA/Linux environment snapshot; not the portable installer |

Non-Torch direct dependencies are pinned to versions present in the checked reference environment. The Torch range allows the hardware-specific build; the checked reference is `2.8.0+cu129`. These are direct dependency constraints, not a full cross-platform lock. See [reproducibility and commands](docs/REPRODUCIBILITY.md).

## What is in the repository

```text
configs/
  datasets/               provenance, dimensions and hashes of recent benchmarks
  paper_campaign/         validation grids and final Net3/Iowa configurations
  smoke.json              one-epoch CPU configuration
src/
  paper_protocol.py       main observation-to-graph pipeline
  model/, dataset/, ...   earlier PyG models and research utilities
scripts/
  benchmarks/             main generators, campaign, baselines and exports
  validation/             synthetic Section V generators and campaigns
  validation_supplement/  alpha/k sensitivity and supplementary analyses
  experiments/            earlier synthetic/PyG entry points
  synthetic/, electric/   earlier generators
  water/, diagnostics/, visualization/
tests/                    self-contained protocol and training checks
data/synthetic/           small, previously tracked synthetic examples
docs/                     setup, dataset contracts, status and historical methods
legacy/, notebooks/       historical placeholders
```

New downloaded datasets, simulation outputs, checkpoints, reports and figures are excluded from Git. The recent benchmarks are regenerated from their upstream sources, not silently bundled. [Dataset acquisition and formats](docs/DATASETS.md) covers Net3, Fossolo, Pescara, Iowa feeder B, SimBench and the newer homogeneous synthetic variants.

## Before running a scientific campaign

Read the [status and known limits](docs/STATUS.md), then [generate the datasets](docs/DATASETS.md) and follow the [campaign commands](docs/REPRODUCIBILITY.md). Budget for the declared seeds and epochs; the smoke test does not reproduce a scientific experiment. [Contributing](CONTRIBUTING.md) describes checks and what belongs in Git.

No new distribution licence is assigned by this cleanup. Check upstream code and dataset terms before redistribution.
