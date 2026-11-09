# Platforms and accelerators

Checked on 2026-11-09. The main benchmark path uses dense PyTorch operations (`bmm`, `cdist`, `pinv`, top-k); it does not require KNN-CUDA, KeOps, PyG or a custom CUDA extension. The earlier synthetic path additionally uses PyTorch Geometric.

| Platform | Main pipeline | Evidence / condition |
| --- | --- | --- |
| Linux / WSL2, CPU | Supported reference path | Local protocol tests and full smoke campaign |
| Linux / WSL2, NVIDIA | Supported reference path | Local CUDA smoke; historical benchmark runs exist |
| Windows native, CPU | Verified in GitHub CI | Clean Python 3.11 / Torch 2.8.0 CPU install, ten tests and full smoke campaign |
| Windows native, NVIDIA | Expected to work | Install the matching official Torch wheel; not hardware-tested here |
| Linux, AMD ROCm | Adaptable without an algorithm rewrite | Install a supported ROCm Torch build; no AMD hardware available for verification |
| Windows / WSL2, AMD | Hardware/build dependent, unverified | Check AMD's current GPU/OS/Python matrix; do not reuse the historical NVIDIA snapshot |

PyTorch ROCm intentionally exposes the GPU through `torch.cuda`; `--device cuda` is therefore correct on AMD too. `torch.version.hip` identifies a ROCm build. See [PyTorch HIP semantics](https://docs.pytorch.org/docs/stable/notes/hip.html).

The [Linux and Windows CPU workflow](https://github.com/MarcoCasalbore/Graph-water-recon/actions/runs/42620227193) passed on the cleanup code commit. Native Windows simulator adapters and GPU execution were not part of that job.

## Install and verify

Choose a wheel from the official [PyTorch selector](https://pytorch.org/get-started/locally/) for your OS, driver and accelerator. For AMD, also check the [ROCm compatibility documentation](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html). AMD publishes a separate [Windows PyTorch installation guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html); its Python/build requirements can differ from this project's reference environment.

Install Torch first, then `python -m pip install -r requirements-core.txt`. Check:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.version.hip, torch.cuda.is_available())"
python -m unittest discover -s tests -v
```

Run the README campaign with `--device cuda`. Test both water and electric fixtures (`generate_smoke_benchmark.py --domain electric`), since the electric prior uses matrix pseudoinverses. CPU fallback is explicit with `--device cpu`; `--device auto` may silently choose CPU by design, while an unavailable explicit GPU is an error.

The benchmark runner records Python, platform, Torch, NumPy, CUDA/HIP and selected device in each run's `environment.json`. CPU execution on a machine with a GPU does not record GPU memory as CPU memory.

## Data generation and earlier experiments

EPANET/WNTR and OpenDSS are CPU simulators and do not require an NVIDIA GPU. WNTR documents Windows and Linux installation in its [official guide](https://usepa.github.io/WNTR/installation.html). Simulator binary availability must still be checked for the actual OS/architecture. SimBench additionally needs pandapower and its solver stack.

`requirements-water-gpu.txt` is an old full Linux/CUDA freeze with NVIDIA libraries and compiled PyG wheels. It is retained as an environment record, not installed by any default command. Earlier TSL/Lightning forecasting modes have additional optional dependencies and are outside the validated workflow. Newer GPU builds and different devices can change numerical results; fixed seeds do not guarantee bitwise equality across platforms.
