"""
plot_architecture.py — IC3INA 2026
Gambar arsitektur pipeline RAG dengan taxonomy-based reranking.
Output: figures/architecture.png (+ .pdf)

Jalankan: python plot_architecture.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

# ── Warna per komponen ──────────────────────────────────────────────────────
C_INPUT    = "#d6eaf8"   # biru muda — input/output
C_TAX      = "#fdebd0"   # oranye muda — taxonomy
C_QSTRAT   = "#d5f5e3"   # hijau muda — query strategy
C_RETRIEV  = "#e8daef"   # ungu muda — retrieval
C_RERANK   = "#fadbd8"   # merah muda — reranking
C_LLM      = "#fef9e7"   # kuning muda — LLM
C_EDGE     = "#2c3e50"   # border gelap
C_ARROW    = "#5d6d7e"   # warna panah
C_SIDE     = "#eaeded"   # kotak samping (data)

FONT = "DejaVu Sans"

fig, ax = plt.subplots(figsize=(12, 14))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis("off")

# ── Helper fungsi ───────────────────────────────────────────────────────────
def box(ax, x, y, w, h, color, text, fontsize=10, bold=False,
        subtext=None, subfontsize=8.5, radius=0.25):
    patch = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0.05,rounding_size={radius}",
        facecolor=color, edgecolor=C_EDGE, linewidth=1.2,
        zorder=3,
    )
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    if subtext:
        ax.text(x, y + h*0.12, text, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, fontfamily=FONT, zorder=4)
        ax.text(x, y - h*0.22, subtext, ha="center", va="center",
                fontsize=subfontsize, color="#555555", fontfamily=FONT,
                style="italic", zorder=4)
    else:
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, fontfamily=FONT, zorder=4)

def arrow(ax, x1, y1, x2, y2, label=None, color=C_ARROW):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.5, mutation_scale=14),
                zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.1, my, label, fontsize=8, color=color,
                va="center", fontfamily=FONT, zorder=5)

def dashed_arrow(ax, x1, y1, x2, y2, color="#aab7b8"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.2, linestyle="dashed", mutation_scale=12),
                zorder=2)

# ═══════════════════════════════════════════════════════════════════════════
# LAYOUT  (x=5 = tengah, y dari atas ke bawah)
# ═══════════════════════════════════════════════════════════════════════════

# 1. User Query
box(ax, 5, 13.2, 4.5, 0.75, C_INPUT,
    "User Query (Bahasa Indonesia)", fontsize=11, bold=True)

# 2. Taxonomy Detection
box(ax, 5, 11.9, 5.2, 0.9, C_TAX,
    "Taxonomy Detection",
    subtext="detect_query_concepts()  →  max 4 konsep terdeteksi",
    fontsize=10.5, bold=True)

# File taxonomy (kanan)
box(ax, 8.6, 11.9, 1.9, 0.75, C_SIDE,
    "synonym_index_v4.json\ntaxonomy_concepts_v4.json",
    fontsize=7.5)
dashed_arrow(ax, 7.65, 11.9, 8.6 - 0.95, 11.9)

# 3. Dua cabang query strategy
# Garis bercabang dari Taxonomy Detection
ax.plot([5, 5], [11.45, 11.1], color=C_ARROW, lw=1.5, zorder=2)
ax.plot([2.8, 7.2], [11.1, 11.1], color=C_ARROW, lw=1.5, zorder=2)
arrow(ax, 2.8, 11.1, 2.8, 10.55)
arrow(ax, 7.2, 11.1, 7.2, 10.55)

# QEXP
box(ax, 2.8, 10.2, 3.6, 0.65, C_QSTRAT,
    "QEXP — Query Expansion",
    subtext="append sinonim → 1 query string",
    fontsize=9.5, bold=True)

# QREWRITE
box(ax, 7.2, 10.2, 3.6, 0.65, C_QSTRAT,
    "QREWRITE — Query Rewriting",
    subtext="substitusi sinonim → max 4 variasi",
    fontsize=9.5, bold=True)

# Label cabang
ax.text(2.8, 10.75, "QEXP", ha="center", fontsize=8,
        color="#1a6620", fontfamily=FONT)
ax.text(7.2, 10.75, "QREWRITE", ha="center", fontsize=8,
        color="#1a6620", fontfamily=FONT)

# 4. Retrieval — gabung kembali ke tengah
arrow(ax, 2.8, 9.875, 4.0, 9.35)
arrow(ax, 7.2, 9.875, 6.0, 9.35)

box(ax, 5, 9.0, 5.4, 0.65, C_RETRIEV,
    "BGE-M3 Embedding  +  ChromaDB Retrieval",
    subtext="top-20 kandidat chunk per variasi query",
    fontsize=10, bold=True)

# 5. Merge (khusus QREWRITE)
box(ax, 7.8, 8.1, 2.2, 0.55, C_SIDE,
    "Merge + Freq Boost\n(khusus QREWRITE)",
    fontsize=7.5)
dashed_arrow(ax, 6.7, 9.0, 7.8 - 1.1, 8.37)

# 6. Reranking
arrow(ax, 5, 8.675, 5, 8.15)

box(ax, 5, 7.75, 5.8, 0.7, C_RERANK,
    "Reranking",
    subtext="final = w·vec·score + w·bm25·score + w·tax·score + w·CE·score",
    fontsize=10.5, bold=True)

# Panel bobot di sebelah kiri
box(ax, 1.2, 7.1, 2.0, 1.5, C_SIDE,
    "Bobot per Skenario\n\n"
    "BM25+TAX: (0.45, 0.25, 0.30, 0)\n"
    "CE:       (0.80, 0,    0,    0.20)\n"
    "TAX-CE:   (0.70, 0,    0.10, 0.20)",
    fontsize=7)
dashed_arrow(ax, 2.2, 7.75, 1.2 + 1.0, 7.75)

# Cross-Encoder (kanan)
box(ax, 8.7, 7.75, 2.0, 0.55, C_SIDE,
    "Cross-Encoder\nBAAI/bge-reranker-v2-m3",
    fontsize=7.5)
dashed_arrow(ax, 7.9, 7.75, 8.7 - 1.0, 7.75)

# 7. Top-5 chunks
arrow(ax, 5, 7.4, 5, 6.9)
box(ax, 5, 6.65, 3.2, 0.45, C_RETRIEV,
    "Top-5 Chunks  →  Konteks LLM",
    fontsize=9.5)

# 8. LLM
arrow(ax, 5, 6.425, 5, 5.95)
box(ax, 5, 5.65, 4.6, 0.55, C_LLM,
    "LLM — Sahabat AI (Ollama)",
    subtext="csalab/sahabatai1:latest  |  port 5000",
    fontsize=10, bold=True)

# 9. Output
arrow(ax, 5, 5.375, 5, 4.9)
box(ax, 5, 4.65, 4.5, 0.5, C_INPUT,
    "Jawaban (Bahasa Indonesia)",
    fontsize=10.5, bold=True)

# ═══════════════════════════════════════════════════════════════════════════
# LEGEND skenario ablasi (bawah)
# ═══════════════════════════════════════════════════════════════════════════
ax.axhline(4.1, xmin=0.03, xmax=0.97, color="#bdc3c7", lw=0.8, linestyle="--")

ax.text(5, 3.75, "Skenario Ablasi (v4)", ha="center", fontsize=10,
        fontweight="bold", fontfamily=FONT, color=C_EDGE)

ablasi = [
    ("TAX-QEXP-CLEAN",    "QEXP",     "BM25+TAX (0.45/0.25/0.30)"),
    ("QRWT-TAX-CLEAN",    "QREWRITE", "BM25+TAX (0.45/0.25/0.30)"),
    ("CE-CLEAN",          "QEXP",     "CE       (0.80/0/0.20)"),
    ("TAX-CE-CLEAN",      "QEXP",     "CE+TAX   (0.70/0.10/0.20)"),
    ("QRWT-CE-CLEAN ★",   "QREWRITE", "CE       (0.80/0/0.20)  ← BEST"),
]

col_colors = [C_QSTRAT, C_RERANK, C_INPUT]
headers = ["Skenario", "Query Strategy", "Reranking (w_vec/w_tax/w_CE)"]
col_x = [1.5, 4.2, 7.3]
row_start = 3.35

for ci, (hdr, cx) in enumerate(zip(headers, col_x)):
    ax.text(cx, row_start, hdr, ha="left", fontsize=8,
            fontweight="bold", fontfamily=FONT, color=C_EDGE)

for ri, (nama, qstrat, rerank) in enumerate(ablasi):
    y = row_start - 0.42 * (ri + 1)
    is_best = "★" in nama
    weight = "bold" if is_best else "normal"
    color  = "#1a5276" if is_best else "#2c3e50"
    ax.text(col_x[0], y, nama,    ha="left", fontsize=8, fontfamily=FONT,
            fontweight=weight, color=color)
    ax.text(col_x[1], y, qstrat,  ha="left", fontsize=8, fontfamily=FONT,
            color="#1a6620")
    ax.text(col_x[2], y, rerank,  ha="left", fontsize=8, fontfamily=FONT,
            color=color)

# ═══════════════════════════════════════════════════════════════════════════
# Judul
# ═══════════════════════════════════════════════════════════════════════════
ax.text(5, 13.75, "Arsitektur Pipeline RAG dengan Taxonomy-Based Reranking",
        ha="center", va="center", fontsize=13, fontweight="bold",
        fontfamily=FONT, color=C_EDGE)
ax.text(5, 13.52, "Domain: Hemodialisis (Bahasa Indonesia)",
        ha="center", va="center", fontsize=9, color="#7f8c8d", fontfamily=FONT)

plt.tight_layout(pad=0.3)
plt.savefig(OUT_DIR / "architecture.png", dpi=300, bbox_inches="tight",
            facecolor="white")
plt.savefig(OUT_DIR / "architecture.pdf", bbox_inches="tight",
            facecolor="white")
print(f"Saved → {OUT_DIR}/architecture.png")
