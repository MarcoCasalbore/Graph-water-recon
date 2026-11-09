"""Parse and plot the explicitly post-hoc per-epoch eval0.5 diagnostic."""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results_section_v"
ROOT = RESULTS_ROOT / "campaign_v2/relational"
sys.path.insert(0, str(RESULTS_ROOT))

from scripts.plot_style import COLORS, apply_style, save_figure

EPOCH = re.compile(r"Epoch (\d+)/")
DIAG = re.compile(r"DiagnosticGraph - F1: ([0-9.eE+-]+) \| TaskScore: ([0-9.eE+-]+) \| Eval0.5: ([0-9.eE+-]+)")

log = ROOT / "epoch_diagnostics/seed42_e300_v2/training.log"
rows = []; epoch = None
for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
    match = EPOCH.search(line)
    if match: epoch = int(match.group(1))
    match = DIAG.search(line)
    if match and epoch is not None:
        rows.append({"epoch": epoch, "diagnostic_graph_f1": float(match.group(1)),
                     "diagnostic_task_score": float(match.group(2)), "diagnostic_eval0p5": float(match.group(3)),
                     "oracle_posthoc": True, "used_for_selection": False})
frame = pd.DataFrame(rows)
if len(frame) != 300:
    raise RuntimeError(f"Expected 300 diagnostic epochs, found {len(frame)}")
out = ROOT / "figures/data/epoch_diagnostics.csv"; out.parent.mkdir(parents=True, exist_ok=True); frame.to_csv(out, index=False)
apply_style(); fig, ax = plt.subplots(figsize=(7.0, 2.8))
ax.plot(frame.epoch, frame.diagnostic_task_score, color=COLORS["forecast"], label="Train TaskScore")
ax.plot(frame.epoch, frame.diagnostic_graph_f1, color=COLORS["topology"], label="Current-batch graph F1")
ax.plot(frame.epoch, frame.diagnostic_eval0p5, color=COLORS["objective"], label=r"Diagnostic $Eval_{0.5}$")
ax.set_xlabel("Training epoch"); ax.set_ylabel("Post-hoc diagnostic value"); ax.legend(frameon=False, ncol=3)
save_figure(fig, ROOT / "figures", "eval0p5_per_epoch"); plt.close(fig)
print(f"[OK] diagnostic epochs={len(frame)}")
