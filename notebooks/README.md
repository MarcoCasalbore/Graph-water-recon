# Paper drawing sources

`paper_figures/` contains seven cleaned layout notebooks recovered from the local publication work:

| Notebook | Drawing source |
| --- | --- |
| `net3.ipynb` | Real-derived water network layout |
| `iowa_b.ipynb` | Iowa feeder B layout |
| `synthetic.ipynb` | Controlled synthetic layout |
| `scenario_a.ipynb` through `scenario_d.ipynb` | Additional scenario layouts/styles |

Open with a Jupyter-compatible editor and a kernel using the project environment, with its working directory inside the repository. Each notebook locates the repository from the working directory and saves to its own `outputs/paper_figures/<name>/` folder. Saved execution outputs are removed from Git; embedded coordinates and edge lists are retained as drawing-source snapshots. No training or model selection takes place in these notebooks.

These sources reproduce the saved drawing/layout. To draw a graph from a newly trained run, use `scripts/benchmarks/render_campaign_artifacts.py` or `assemble_final_artifacts.py`; do not treat embedded notebook edges as new model outputs. The original executable cell syntax was checked to remain identical except for output locations; comment-only cells were omitted.

Notebook PDF/PNG exports remain in ignored `outputs/paper_figures/<name>/`. Keep the editable publication deck and reference manuscript in ignored `paper_artifacts/publication/`, not in Git. See [paper reproduction](../docs/PAPER_REPRODUCTION.md).
