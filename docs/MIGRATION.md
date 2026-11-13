# Move the paper workspace to another machine

There are two separate transfers: Git contains code/configuration/drawing sources; the migration archive published in the repository's [paper workspace Release](https://github.com/MarcoCasalbore/Graph-water-recon/releases/tag/paper-workspace-2026-11-09) contains the selected data, trained results and publication materials that Git intentionally excludes. The source WDGM environment stays on the original machine. Recreate the environment at the destination instead of copying its executables/shebangs.

## Choose the target

The closest reference is Linux or WSL2, Python 3.11, NVIDIA with Torch 2.8.0+cu129. CPU is a portable starting point; the main CPU workflow passed on Linux and Windows. AMD/ROCm and native Windows GPU require target validation before scientific training. Native Windows also needs long-path support for some historical run names; WSL2 avoids that restriction. See [PLATFORMS.md](PLATFORMS.md).

## Restore a complete prepared transfer

The locally prepared package has `Graph-water-recon/` as its archive prefix. Its manifest records the exact Git commit and SHA-256 of each payload file. Use a new empty destination; do not merge it into unrelated or modified experiments.

1. Download the archive and its `.sha256` sidecar from the [paper workspace Release](https://github.com/MarcoCasalbore/Graph-water-recon/releases/tag/paper-workspace-2026-11-09), following [the prepared handoff](../transfer_handoff/README.md). No source-PC access is needed. You can also copy an already verified local archive. Validate the archive hash before extraction (`sha256sum -c FILE.sha256` on Linux; `Get-FileHash FILE -Algorithm SHA256` in PowerShell).
2. Clone the repository, check out the exact commit in the handoff and overlay the archive into that clone:

```bash
git clone https://github.com/MarcoCasalbore/Graph-water-recon.git Graph-water-recon
cd Graph-water-recon
git checkout COMMIT_FROM_HANDOFF
tar -xzf PATH_TO_ARCHIVE --strip-components=1 -C .
python scripts/maintenance/migrate_workspace.py verify
```

If GitHub is unavailable, the archive also includes the pinned tracked source snapshot; extract it into an empty folder and use that source. It will not have `.git` history. Python is needed for checksum verification but scientific libraries are not.

3. Preview and apply metadata path relocation:

```bash
python scripts/maintenance/migrate_workspace.py rebase
python scripts/maintenance/migrate_workspace.py rebase --apply
python scripts/maintenance/migrate_workspace.py verify --manifest migration/manifest-relocated.json
```

The first command writes nothing. The second changes only JSON/CSV string values beginning with the recorded old project root. It does not edit arrays, weights, predictions, metric values or code; CSV numeric spellings are retained. `migration/manifest.json` remains the original transfer record; `manifest-relocated.json` records the resulting file hashes. The original archive remains the unmodified source snapshot. A failed write rolls back the metadata files already changed during that invocation.

Verify the original manifest before modifying any transfer files. After relocation use the relocated manifest. Subsequent training/analysis can legitimately change files, so a frozen transfer manifest is no longer an expected checksum of the evolving workspace.

Logs, human-readable notes and serialized checkpoint internals are not rewritten. Historical command traces may mention the original interpreter path; use the active destination Python and current runner commands rather than replaying old shell commands verbatim.

4. Once both transfer checks have passed, return a cloned checkout to the maintained source before installing dependencies and running tests:

```bash
git switch main
```

This retains the ignored transferred datasets/results and restores the current setup fixes and Windows-compatible tests. The manifest describes the historical source snapshot; after switching source, do not expect its code hashes to match current main. Finish the original/relocated manifest checks **before** this switch. The archive remains a verified immutable backup. When using the offline source snapshot without Git, install the explicit `six` dependency below; its historical Windows migration tests compare short and resolved path spellings and may fail for an equivalent 8.3 path. Use current main for the supported cross-platform tests.

## Recreate the environment

Use Python 3.11. On Linux/WSL:

```bash
python3.11 -m venv WDGM
source WDGM/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell:

```powershell
py -3.11 -m venv WDGM
.\WDGM\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install one suitable Torch build, then the full paper requirements. The explicit `six` install below is required for the frozen archive snapshot: WNTR imports it without declaring the dependency. Current main also lists it in `requirements-benchmarks.txt`; the archived historical requirements remain unchanged for checksum verification:

```bash
# Portable CPU starting point
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install six==1.17.0
python -m pip install -r requirements.txt
python -m pip check
```

For a compatible NVIDIA driver/GPU, the reference build is:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
python -m pip install six==1.17.0
python -m pip install -r requirements.txt
```

Choose other builds using the official [PyTorch version instructions](https://pytorch.org/get-started/previous-versions/). For AMD use a compatible ROCm build and hardware/OS combination, following the official compatibility matrix. PyTorch ROCm also uses `--device cuda`. Do not install NVIDIA-specific freeze files on AMD/CPU/Windows as a generic setup procedure.

`migration/source_environment.json` and `source_requirements.txt` are provenance snapshots of the original machine. The freeze contains hardware-specific and optional historical packages; it is not a portable installer. Direct requirements are pinned where appropriate, but transitive packages/backend/hardware changes can affect numerical results.

## Destination checks, before expensive runs

These commands are for the destination operator; preparing the transfer does not run them automatically:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.version.hip, torch.cuda.is_available())"
python -m unittest discover -s tests -v
python scripts/benchmarks/generate_smoke_benchmark.py --output-dir outputs/migration_check/water_data
python scripts/benchmarks/run_paper_campaign.py --observation-dir outputs/migration_check/water_data/observations --evaluation-dir outputs/migration_check/water_data/evaluation_only --output-dir outputs/migration_check/water_run --spec configs/smoke.json --seeds 0 --device cpu
python scripts/benchmarks/generate_smoke_benchmark.py --output-dir outputs/migration_check/electric_data --domain electric
python scripts/benchmarks/run_paper_campaign.py --observation-dir outputs/migration_check/electric_data/observations --evaluation-dir outputs/migration_check/electric_data/evaluation_only --output-dir outputs/migration_check/electric_run --spec configs/smoke.json --seeds 0 --device cpu
```

On GPU repeat the checks in new output directories with `--device cuda`; this must succeed explicitly, not merely fall back to CPU. In particular the electric prior needs backend support for matrix pseudoinverses. Check simulator imports too:

```bash
python -c "import wntr, opendssdirect, pandapower, simbench; print('Simulator imports OK')"
```

Then follow [PAPER_REPRODUCTION.md](PAPER_REPRODUCTION.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Keep migrated final results unchanged; put new runs under new `outputs/` names. Older final checkpoints without a run-identity record remain historical artifacts and cannot automatically resume in the current main runner. Never invent an identity record to bypass that check.

## Archive contents and exclusions

Included: tracked code/config/tests/notebooks/docs; selected data and upstream model sources; Net3/Iowa finals and baselines; synthetic ablations and aligned predictions; eight supplementary scenarios; retained support artifacts; editable deck/reference manuscript; bibliography; per-file manifest and source environment provenance.

Excluded: WDGM/other environments, Git object history, credentials/editor/assistant state, caches and temporary audits. The separate Rieti/CSA field measurements and unrelated INP models under the outer `other_research/` directory are not paper dependencies and are not included. Transfer those separately if the destination will also work on that research.

The prepared archive and checksum are GitHub Release assets of this repository. They are downloaded separately from a normal Git clone; GitHub's source ZIP alone does not include them. The reference manuscript and upstream data retain their original rights/terms. Successful transfer verification proves file integrity, not identical scientific results or target-GPU compatibility.
