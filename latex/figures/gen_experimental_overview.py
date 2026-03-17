#!/usr/bin/env python3
"""Generate experimental overview diagram for Chapter 8.

Shows the two-phase experimental pipeline:
  Phase 1: Stage 1 (7 DQN) → Stage 2 (4×6 grid) → Stage 3 (S32/S33) → 5 agents
  + Variance estimation (escape velocity)
  Phase 2: Train-only grid (6 agents) → Mixed-op stress test (Baseline+Baseline)

Pure matplotlib implementation matching the system_architecture.py style.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────

FIG_W, FIG_H = 14, 7.5
DPI = 200
OUT = Path(__file__).parent / "experimental_overview.png"

# Colour palette
C_PHASE1  = {"bg": "#D6E4F0", "border": "#4A7FB5", "box": "#4A7FB5",  "text": "#FFFFFF"}
C_PHASE2  = {"bg": "#D9EDDB", "border": "#5AA861", "box": "#5AA861",  "text": "#FFFFFF"}
C_EXTRA   = {"bg": "#FDEBD0", "border": "#D68A2E", "box": "#D68A2E",  "text": "#FFFFFF"}
C_RESULT  = {"bg": "#E8DAEF", "border": "#7D3C98", "box": "#7D3C98",  "text": "#FFFFFF"}
C_GREY    = {"bg": "#F0F0F0", "border": "#999999", "box": "#999999",  "text": "#FFFFFF"}

ARROW_KW = dict(
    arrowstyle="-|>",
    connectionstyle="arc3,rad=0.0",
    linewidth=1.8,
    mutation_scale=16,
    color="#333333",
    shrinkA=4,
    shrinkB=4,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def draw_band(ax, y_bot, height, label, c, x_left=0.4, x_width=13.2):
    """Draw a coloured horizontal band."""
    rect = FancyBboxPatch(
        (x_left, y_bot), x_width, height,
        boxstyle="round,pad=0.12",
        facecolor=c["bg"], edgecolor=c["border"],
        linewidth=1.6, alpha=0.45, zorder=0,
    )
    ax.add_patch(rect)
    ax.text(
        x_left + 0.28, y_bot + height / 2, label,
        fontsize=9, fontweight="bold", color=c["border"],
        rotation=90, ha="center", va="center", zorder=1,
    )


def box(ax, cx, cy, w, h, line1, c, fontsize=8.5, line2=None, line3=None):
    """Draw a rounded box with up to 3 text lines."""
    b = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.08",
        facecolor=c["box"], edgecolor="white",
        linewidth=1.0, zorder=3,
    )
    ax.add_patch(b)
    if line3:
        ax.text(cx, cy + 0.22, line1, fontsize=fontsize, fontweight="bold",
                color=c["text"], ha="center", va="center", zorder=4)
        ax.text(cx, cy, line2, fontsize=fontsize - 1.5,
                color=c["text"], ha="center", va="center", zorder=4, alpha=0.9)
        ax.text(cx, cy - 0.20, line3, fontsize=fontsize - 1.5,
                color=c["text"], ha="center", va="center", zorder=4, alpha=0.9)
    elif line2:
        ax.text(cx, cy + 0.12, line1, fontsize=fontsize, fontweight="bold",
                color=c["text"], ha="center", va="center", zorder=4)
        ax.text(cx, cy - 0.12, line2, fontsize=fontsize - 1.5,
                color=c["text"], ha="center", va="center", zorder=4, alpha=0.9)
    else:
        ax.text(cx, cy, line1, fontsize=fontsize, fontweight="bold",
                color=c["text"], ha="center", va="center", zorder=4)
    return (cx, cy)


def arr(ax, start, end, **kw):
    """Straight arrow."""
    props = {**ARROW_KW, **kw}
    a = FancyArrowPatch(start, end, **props, zorder=2)
    ax.add_patch(a)


def arr_curved(ax, start, end, rad=0.15, **kw):
    """Curved arrow."""
    props = {**ARROW_KW, **kw}
    props["connectionstyle"] = f"arc3,rad={rad}"
    a = FancyArrowPatch(start, end, **props, zorder=2)
    ax.add_patch(a)


def label(ax, x, y, text, fontsize=7.5, color="#555555"):
    """Small italic annotation."""
    ax.text(x, y, text, fontsize=fontsize, fontstyle="italic",
            color=color, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85),
            zorder=5)


# ── Main figure ────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.set_aspect("equal")
ax.axis("off")

ax.text(FIG_W / 2, FIG_H - 0.25, "Experimental Pipeline Overview",
        fontsize=14, fontweight="bold", ha="center", va="top", color="#222222")

# ── Phase bands ───────────────────────────────────────────────────────────

draw_band(ax, 3.9, 3.0, "Phase 1", C_PHASE1)
draw_band(ax, 0.4, 3.2, "Phase 2", C_PHASE2)

# ── PHASE 1 BOXES (y ~ 4.2 - 6.6) ───────────────────────────────────────

bw, bh = 2.4, 0.80

# Stage 1
s1 = box(ax, 2.8, 6.2, bw, bh, "Stage 1", C_PHASE1,
         line2="7 DQN variants", line3="× Baseline CNN")

# Funnel annotation
label(ax, 4.65, 6.2, "4 pass")

# Stage 2
s2 = box(ax, 6.5, 6.2, bw, bh, "Stage 2", C_PHASE1,
         line2="4 DQN × 6 CNN", line3="= 24 combinations")

label(ax, 8.35, 6.2, "14 pass")

# Stage 3
s3 = box(ax, 10.2, 6.2, bw, bh, "Stage 3", C_PHASE1,
         line2="10 combos on", line3="S32–S33 gate")

label(ax, 12.05, 6.2, "5 pass")

# Arrows between stages
arr(ax, (s1[0] + bw/2, s1[1]), (s2[0] - bw/2, s2[1]))
arr(ax, (s2[0] + bw/2, s2[1]), (s3[0] - bw/2, s3[1]))

# Variance estimation (below main pipeline)
var = box(ax, 6.5, 4.65, 2.8, 0.70, "Escape Velocity", C_EXTRA,
          line2="10 seeds × 7 variants")

# Arrow from Stage 1 down to variance
arr_curved(ax, (s1[0] + 0.4, s1[1] - bh/2), (var[0] - 1.2, var[1] + 0.05),
           rad=-0.15, color=C_EXTRA["border"], linewidth=1.4, linestyle="--")

# Result of Phase 1
p1_result = box(ax, 12.5, 4.65, 2.0, 0.70, "5 Agents", C_RESULT,
                line2="+ GRU exploratory")

arr(ax, (s3[0] + bw/2 - 0.4, s3[1] - bh/2 + 0.1), (p1_result[0] - 0.8, p1_result[1] + 0.35),
    color=C_RESULT["border"], linewidth=1.4)

# ── PHASE 2 BOXES (y ~ 0.7 - 3.3) ───────────────────────────────────────

# Train-only grid
train = box(ax, 4.0, 2.5, 3.0, 0.85, "Train Operations", C_PHASE2,
            line2="6 loads × 3 fills", line3="6 DQN agents + heuristic")

# Mixed-op stress test
stress = box(ax, 9.0, 2.5, 3.0, 0.85, "Stress Test", C_PHASE2,
             line2="3 loads × 2 mixes", line3="Baseline+Baseline only")

# Arrow from Phase 1 result down to Phase 2
arr(ax, (p1_result[0] - 0.5, p1_result[1] - 0.35),
    (train[0] + 1.2, train[1] + 0.43),
    color=C_RESULT["border"], linewidth=1.6)

arr(ax, (train[0] + 1.5, train[1]), (stress[0] - 1.5, stress[1]),
    color=C_PHASE2["border"], linewidth=1.4)

# RQ labels
rq1 = box(ax, 2.8, 4.65, 1.6, 0.55, "RQ1 + RQ3", C_GREY, fontsize=8)
arr(ax, (rq1[0] + 0.8, rq1[1] + 0.1), (s1[0] - 0.3, s1[1] - bh/2),
    color="#999999", linewidth=1.0, linestyle=":")

rq2 = box(ax, 4.0, 1.3, 1.2, 0.45, "RQ2", C_GREY, fontsize=8)
arr(ax, (rq2[0], rq2[1] + 0.23), (train[0], train[1] - 0.43),
    color="#999999", linewidth=1.0, linestyle=":")

# ── Curriculum input arrow ────────────────────────────────────────────────

# Tutorial curriculum feeding into Phase 1
curr = box(ax, 2.8, 6.95, 2.2, 0.40, "Tutorial Curriculum", C_GREY, fontsize=7.5)
arr(ax, (curr[0], curr[1] - 0.20), (s1[0], s1[1] + bh/2),
    color="#999999", linewidth=1.2, linestyle="--")

# ── Legend ─────────────────────────────────────────────────────────────────

lx, ly = 1.0, 0.15
items = [
    ("Phase 1: Tutorial Screening", C_PHASE1["box"]),
    ("Phase 2: Scalability Eval", C_PHASE2["box"]),
    ("Exploratory", C_EXTRA["box"]),
    ("Research Questions", C_GREY["box"]),
]
for i, (lbl, col) in enumerate(items):
    x = lx + i * 3.2
    rect = FancyBboxPatch(
        (x, ly - 0.08), 0.24, 0.20,
        boxstyle="round,pad=0.03", facecolor=col, edgecolor="white",
        linewidth=0.8, zorder=5,
    )
    ax.add_patch(rect)
    ax.text(x + 0.38, ly + 0.02, lbl, fontsize=7, va="center", color="#333333", zorder=5)

# ── Save ───────────────────────────────────────────────────────────────────

fig.tight_layout(pad=0.3)
fig.savefig(OUT, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved -> {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
