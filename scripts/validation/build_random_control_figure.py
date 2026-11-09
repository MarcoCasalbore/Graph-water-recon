"""Create the supplementary random-control comparison from corrected metrics."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from assemble_section_v_final import corrected_methods  # noqa: E402

P = HERE.parents[1]
R = P / "results_section_v_final"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 10, "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False})

d = corrected_methods()
keep = ["node_only_empty_graph", "proposed_joint", "proposed_learned_graph_frozen", "random_learned_edge_count_control"]
g = d[d.method.isin(keep)].groupby("method").agg(mse_mean=("test_mse", "mean"), mse_std=("test_mse", "std"),
                                                       mae_mean=("test_mae", "mean"), mae_std=("test_mae", "std")).reset_index()
g.to_csv(R / "figures/supplementary/data/random_control_forecasting.csv", index=False)
x = np.arange(len(g)); width = .36
fig, ax = plt.subplots(figsize=(7.0, 3.6))
ax.bar(x-width/2, g.mse_mean, width, yerr=g.mse_std, label="MSE", capsize=2)
ax.bar(x+width/2, g.mae_mean, width, yerr=g.mae_std, label="MAE", capsize=2)
ax.set(xticks=x, xticklabels=g.method.str.replace("_", " "), ylabel="Test error")
ax.tick_params(axis="x", rotation=18); ax.legend()
for ext in ["pdf", "svg", "png"]:
    out = R / "figures/supplementary" / ext; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"random_control_forecasting.{ext}", bbox_inches="tight", dpi=300)
plt.close(fig)
print("[DONE] supplementary random-control figure")
