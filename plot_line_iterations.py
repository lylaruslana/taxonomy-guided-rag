"""
plot_line_iterations.py — IC3INA 2026
Line plot 3 metrik RAGAS per iterasi (mean ± 0.5std across 71 pertanyaan).
3 panel horizontal: Faithfulness | Answer Relevancy | Context Precision.

Tambahkan skenario V4 ke SCENARIOS saat RAGAS selesai.
Jalankan: python plot_line_iterations.py
Output  : figures/line_iterations.png (+ .pdf)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

ROOT      = Path(__file__).parent
RAGAS_DIR = ROOT / "ragas_results"
OUT_DIR   = ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Konfigurasi skenario utama.
# Catatan: hasil v3 tidak ditampilkan karena berasal dari konfigurasi taxonomy
# yang salah; figur utama hanya memakai taxonomy yang sudah diperbaiki.
# ---------------------------------------------------------------------------

SCENARIOS = [
    # (label legenda, config name, warna, linestyle)
    # CE branch: solid lines, warna kontras satu sama lain
    ("BASE",        "D003D-bgem3-sahabatai-semantic-vsi",                   "#555555", "-"),   # abu netral
    ("CE",          "D003D-CE-NOQEXP-bgem3-sahabatai-semantic",             "#1f77b4", "-"),   # biru
    ("QEXP-CE",     "D003D-QEXP-CE-CLEAN-V4-bgem3-sahabatai-semantic",      "#9467bd", "-"),   # ungu
    ("QRW-CE",      "D003D-QREWRITE-CE-CLEAN-V4-bgem3-sahabatai-semantic",  "#d62728", "-"),   # merah (terbaik)
    ("QEXP-CE-T-O", "D003D-TAX-CE-CLEAN-V4-bgem3-sahabatai-semantic",       "#e377c2", "-"),   # pink
    # BMT branch: dashed lines, warna kontras dengan CE branch
    ("QEXP-BMT",    "D003D-TAX-QEXP-CLEAN-V4-bgem3-sahabatai-semantic",     "#2ca02c", "--"),  # hijau
    ("QEXP-BMT-O",  "D003D-TAX-QEXP-OPT-CLEAN-bgem3-sahabatai-semantic",    "#8c564b", "--"),  # coklat
    ("QRW-BMT",     "D003D-QREWRITE-TAX-CLEAN-V4-bgem3-sahabatai-semantic", "#ff7f0e", "--"),  # oranye
]

METRICS = [
    ("faithfulness",      "Faithfulness"),
    ("answer_relevancy",  "Answer Relevancy"),
    ("context_precision", "Context Precision"),
]

# ---------------------------------------------------------------------------
# Load per-iterasi stats untuk satu metrik
# ---------------------------------------------------------------------------

def load_iter_stats(config: str, metric: str):
    f = RAGAS_DIR / f"{config}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    if "repeat_no" not in df.columns or metric not in df.columns:
        return None
    stats = (
        df.groupby("repeat_no")[metric]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("repeat_no")
    )
    stats["std"] = stats["std"].fillna(0)
    return stats

# ---------------------------------------------------------------------------
# Plot: 1 baris × 3 panel (satu per metrik)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=False)
fig.suptitle(
    "RAGAS Metrics Across Iterations — Corrected Taxonomy",
    fontsize=13, fontweight="bold", y=1.01,
)

for ax, (metric_col, metric_label) in zip(axes, METRICS):
    for label, cfg, color, ls in SCENARIOS:
        stats = load_iter_stats(cfg, metric_col)
        if stats is None:
            continue
        iters = stats["repeat_no"].values
        means = stats["mean"].values
        stds  = stats["std"].values
        final = means.mean()

        ax.plot(iters, means, color=color, linestyle=ls, linewidth=1.8,
                label=f"{label} ({final:.4f})", marker="o", markersize=3)
        lo = np.clip(means - 0.5 * stds, 0.0, 1.0)
        hi = np.clip(means + 0.5 * stds, 0.0, 1.0)
        ax.fill_between(iters, lo, hi, color=color, alpha=0.045)

    ax.set_title(metric_label, fontsize=11, fontweight="bold")
    ax.set_xlabel("Iteration", fontsize=10)
    ax.set_ylabel(f"{metric_label} (mean ± 0.5 std)", fontsize=9)
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0.3, 1.05)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.8, loc="lower right", framealpha=0.9, ncol=1)

plt.tight_layout()
plt.savefig(OUT_DIR / "line_iterations.png", dpi=300, bbox_inches="tight")
plt.savefig(OUT_DIR / "line_iterations.pdf", bbox_inches="tight")
print(f"Saved → {OUT_DIR}/line_iterations.png")
