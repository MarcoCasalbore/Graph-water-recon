# Configurations

`smoke.json` is the tiny CPU integration check. `paper_campaign/` contains the recent benchmark grids, final Net3/Iowa settings, fixed-graph baselines and scaling setup. `datasets/` records the acquired benchmark manifests and observation archive hashes; these are provenance records, not input datasets.

The campaign merges its `DEFAULTS` with `base`, then expands the Cartesian product of `grid`. Trial IDs hash the complete merged configuration. Seed ranges are half-open: `100:120` means seeds 100 through 119. `--device` overrides the configuration and therefore changes trial IDs.

In the main protocol `alpha` multiplies the forecasting loss; `1-alpha` multiplies the structural loss. The earlier synthetic options such as `lambda_soft_pseudo_edge` are a separate API and are not silently translated.
