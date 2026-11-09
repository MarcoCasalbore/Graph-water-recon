# Dataset acquisition and contracts

Small pre-existing examples under `data/synthetic/` remain versioned. Recent real-derived benchmarks and upstream downloads under `data/benchmarks/` are local and ignored. Their full acquisition manifests, source revisions, simulator settings, source hashes and observation archive hashes are recorded in `configs/datasets/`. The registry preserves provenance without redistributing third-party downloads. No measured result tables or predictions are included.

## Recent benchmark inventory

| Dataset | Nodes | Samples | Interval | Origin |
| --- | ---: | ---: | --- | --- |
| Net3 | 97 | 2880 | 15 minutes | EPA real-derived water example; main water campaign |
| Fossolo | 37 | 2880 | 15 minutes | Reduced Bologna network; screening |
| Pescara | 71 | 2880 | 15 minutes | Reduced Pescara network; screening |
| Iowa240 feeder B | 60 | 8760 | 1 hour | Public anonymized utility feeder; main electric campaign |
| SimBench LV semiurb4 | 44 | 2880 | 15 minutes | Representative synthetic electric network |
| SimBench LV urban6 | 59 | 2880 | 15 minutes | Representative synthetic electric network |

Counts are the actual observation nodes, including the boundary nodes retained by each adapter. Water pressures/demands are simulated, not field measurements. Iowa loads are source time series; voltages are simulated with the complete three-feeder OpenDSS model. Electric NPZ keys `pressure` and `demand` are compatibility names for voltage in per unit and active load in MW; `physical_variable_names` disambiguates them.

## Water: obtain the model and simulate

Install `requirements-benchmarks.txt`. Obtain the INP using the Download link on [Net3](https://waterfutures.github.io/WaterBenchmarkHub/benchmarks/network-Net3.html), [Fossolo](https://waterfutures.github.io/WaterBenchmarkHub/benchmarks/network-Fossolo.html) or [Pescara](https://waterfutures.github.io/WaterBenchmarkHub/benchmarks/network-Pescara.html). The reference acquisition used [WaterBenchmarkHub](https://github.com/WaterFutures/WaterBenchmarkHub) revision `2b92f79e41e37b07116df458d67c25d43f7bf9d2`; compare the downloaded bytes with `source_sha256` in the registry if exact provenance matters. Keep downloads outside version control, for example in `data/benchmarks/sources/`.

Replace `PATH/Net3.inp` with the downloaded file:

```bash
python scripts/benchmarks/generate_water_benchmark.py --dataset net3 --source-inp PATH/Net3.inp --provenance "real-derived EPA example based on North Marin Water District, California" --source-url https://www.epa.gov/water-research/epanet --source-revision "WaterBenchmarkHub 2b92f79e41e37b07116df458d67c25d43f7bf9d2" --samples 2880 --step-seconds 900 --seed 20260831 --simulator epanet
```

For Fossolo/Pescara use `--dataset fossolo`/`pescara`, the corresponding INP and source attribution from the registry. The acquired Fossolo/Pescara manifests record CC BY-NC 4.0; follow the upstream attribution and use conditions. The acquired Net3 source did not state an explicit licence in the INP.

The generator copies the source, removes parser-breaking NUL bytes/orphan coordinates where necessary, replaces demand patterns with the recorded diurnal/weekly/stochastic multipliers and preserves topology. It records repairs and hashes in the generated manifest. Pressure/demand arrays use WNTR's SI representation (pressure head in metres, demand in cubic metres per second); upstream INP flow units can differ. Do not interpret normalized metrics as physical-unit errors.

## Iowa feeder B: required files

Download **OpenDSS model** and **Quasi-static time-series analysis** for the 240-bus system from the [Iowa State source page](https://wzy.ece.iastate.edu/testsystem.html). Extract both archives. `--model-dir` must directly contain `Master.dss`, its redirected DSS files and `Buscoords.dss`. `--qsts-dir` must directly contain all six `Feeder{A,B,C}_{P,Q}.mat` matrices. The whole model is needed even when reconstructing only feeder B.

```bash
python scripts/benchmarks/generate_iowa240_feeder.py --dataset iowa240_feeder_b --feeder B --model-dir "PATH/OpenDSS Model (08.13.2020)" --qsts-dir "PATH/Quasi-Static Time-series Analysis (10.16.2019)" --samples 0
```

`--samples 0` uses the complete available year. A small simulator trial can use `--samples 240 --dataset iowa240_feeder_b_smoke`; it is not the full scientific dataset. The source page provides public access with an acknowledgement request but no standard licence string was recorded; this repository does not redistribute these downloads. Cite Bu et al., NAPS 2019, DOI `10.1109/NAPS46351.2019.8999982` when using the source system.

## SimBench

The adapter acquires the named network from the installed SimBench package:

```bash
python scripts/benchmarks/generate_simbench_benchmark.py --network simbench_lv_semiurb4 --samples 2880
python scripts/benchmarks/generate_simbench_benchmark.py --network simbench_lv_urban6 --samples 2880
```

These are representative synthetic networks. The manifest records the exact network code, profile start/stride and converged samples. Consult [SimBench](https://simbench.de/en/) for dataset attribution and terms.

## Main benchmark contract

```text
data/benchmarks/<domain>/<name>/
  observations/
    manifest.json              domain, node order, variables, windows, splits
    data.npz                   two [T,N] signals and node_xy [N,2]; no edges
  evaluation_only/
    reference_topology.npz     edge_index [2,E], node_ids; post-hoc only
    manifest.json              provenance/counts for evaluation tools
```

The observation loader requires two finite signal arrays, finite coordinates, unique node IDs, matching manifest dimensions, positive history/horizon and contiguous chronological train/val/test ranges with at least one complete window each. Masks are not used for imputation: complete the data before loading. Array columns must follow the manifest's node order. Normalization fits only the training slice; windows never cross a split. For final benchmarks use 75/10/15, history 6 and horizon 12.

## Newer synthetic Section V datasets

These are **a separate data contract and protocol**. Their generators were previously mixed into local result folders and now live under `scripts/validation/`:

```bash
python scripts/validation/generate_homogeneous_synthetic.py --help
python scripts/validation/generate_relational_variant.py --out_dir results_section_v/campaign_v2/datasets/homogeneous_relational_g211_s1211 --graph_seed 211 --signal_seed 1211
```

The relational variant has 50 nodes, 720 hourly samples, an MST backbone plus 30 added edges, and independent graph/signal RNGs. Other graph/signal pairs used locally were 101/1101 and 307/1307. The non-relational homogeneous variant and earlier district-based water generators are not equivalent. Its NPZ contains both reference `edge_index` and geometry-only `initial_edge_index`, with splits in `meta.json`; use the earlier synthetic runner, never the main observation loader. Details of the older NPZ format are in [synthetic scripts](../scripts/synthetic/README.md).

No source dataset is needed for the deterministic README smoke fixture. It checks software wiring only and must not be reported as physical-network evidence.
