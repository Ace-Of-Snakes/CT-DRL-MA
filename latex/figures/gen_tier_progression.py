#!/usr/bin/env python3
"""Generate a tier progression staircase diagram for Chapter 7 (Curriculum Training).

Shows the 13 mastery-gated tiers (0-12) as ascending steps with colour coding
by difficulty, scenario counts, and theme labels.

Output: latex/figures/tier_progression.png  (200 DPI)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# -- Tier data ----------------------------------------------------------------
TIERS = [
    (0,  "Primitives",                "S1\u2013S4",        4),
    (1,  "Two-action chains",         "S5,S6,S26,S27",    4),
    (2,  "Full chains",               "S7,S8",             2),
    (3,  "Restack + load",            "S9,S10",            2),
    (4,  "Random generalisation",     "S11,S12",           2),
    (5,  "Multi-step hard",           "S13,S14",           2),
    (6,  "Terminal truck",            "S22\u2013S25",      4),
    (7,  "Multi-vehicle",             "S15,S16",           2),
    (8,  "Bidirectional train",       "S17",               1),
    (9,  "Concurrent ops",            "S18,S19",           2),
    (10, "Full complexity",           "S20,S21",           2),
    (11, "Full train ops",            "S28,S29",           2),
    (12, "Cross-modal orchestration", "S30,S31",           2),
    (13, "Operational readiness",    "S32,S33",           2),
]

N = len(TIERS)

# -- Colour gradient: green -> yellow -> orange -> red -------------------------
cmap = LinearSegmentedColormap.from_list(
    "difficulty",
    ["#4CAF50", "#8BC34A", "#CDDC39", "#FFC107", "#FF9800", "#F44336"],
)
colours = [cmap(i / (N - 1)) for i in range(N)]

# -- Figure setup --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 7.5))
ax.set_xlim(-0.5, N + 1.5)
ax.set_ylim(-1.2, N + 2.0)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")

# Staircase geometry
STEP_W = 0.85
STEP_H = 0.85

for i, (tier, theme, scenarios, n_scen) in enumerate(TIERS):
    x0 = i * STEP_W
    y0 = i * STEP_H

    # -- Draw step block -------------------------------------------------------
    tread = mpatches.FancyBboxPatch(
        (x0, y0),
        STEP_W,
        STEP_H,
        boxstyle="round,pad=0.04",
        facecolor=colours[i],
        edgecolor="white",
        linewidth=1.5,
        zorder=2,
    )
    ax.add_patch(tread)

    # -- Tier number (centred in step) -----------------------------------------
    ax.text(
        x0 + STEP_W / 2,
        y0 + STEP_H / 2 + 0.08,
        f"T{tier}",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="white",
        path_effects=[pe.withStroke(linewidth=2.5, foreground="#333333")],
        zorder=3,
    )

    # -- Scenario count (bottom of step) ---------------------------------------
    ax.text(
        x0 + STEP_W / 2,
        y0 + 0.15,
        f"{n_scen} scen.",
        ha="center",
        va="center",
        fontsize=6.5,
        color="white",
        path_effects=[pe.withStroke(linewidth=1.8, foreground="#555555")],
        zorder=3,
    )

    # -- Theme label (rotated, above step) -------------------------------------
    ax.text(
        x0 + STEP_W / 2,
        y0 + STEP_H + 0.18,
        theme,
        ha="center",
        va="bottom",
        fontsize=7,
        fontstyle="italic",
        color="#333333",
        rotation=55,
        rotation_mode="anchor",
        zorder=3,
    )

    # -- Scenario IDs (below step) ---------------------------------------------
    ax.text(
        x0 + STEP_W / 2,
        y0 - 0.18,
        scenarios,
        ha="center",
        va="top",
        fontsize=5.5,
        color="#666666",
        rotation=55,
        rotation_mode="anchor",
        zorder=3,
    )

    # -- Mastery-gating arrow to next tier -------------------------------------
    if i < N - 1:
        x_arrow_start = x0 + STEP_W - 0.02
        y_arrow_start = y0 + STEP_H / 2
        x_arrow_end = x0 + STEP_W + 0.12
        y_arrow_end = y0 + STEP_H / 2 + STEP_H * 0.55

        ax.annotate(
            "",
            xy=(x_arrow_end, y_arrow_end),
            xytext=(x_arrow_start, y_arrow_start),
            arrowprops=dict(
                arrowstyle="->,head_width=0.18,head_length=0.12",
                color="#888888",
                lw=1.2,
                connectionstyle="arc3,rad=0.15",
            ),
            zorder=1,
        )

# -- Title ---------------------------------------------------------------------
ax.text(
    N * STEP_W / 2,
    N * STEP_H + 1.5,
    "Tutorial Curriculum: Mastery-Gated Tier Progression",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold",
    color="#222222",
)

# -- Difficulty colour bar legend ----------------------------------------------
cbar_ax = inset_axes(
    ax,
    width="30%",
    height="2.5%",
    loc="lower right",
    borderpad=2.0,
)
gradient = np.linspace(0, 1, 256).reshape(1, -1)
cbar_ax.imshow(gradient, aspect="auto", cmap=cmap)
cbar_ax.set_xticks([0, 255])
cbar_ax.set_xticklabels(["Easy", "Hard"], fontsize=8)
cbar_ax.set_yticks([])
cbar_ax.set_title("Difficulty", fontsize=8, pad=3)

# -- Mastery gate legend entry -------------------------------------------------
ax.annotate(
    "  = mastery gate",
    xy=(N * STEP_W - 1.0, -0.6),
    fontsize=8,
    color="#888888",
    va="center",
)
ax.annotate(
    "",
    xy=(N * STEP_W - 1.0, -0.6),
    xytext=(N * STEP_W - 2.0, -0.6),
    arrowprops=dict(
        arrowstyle="->,head_width=0.18,head_length=0.12",
        color="#888888",
        lw=1.2,
    ),
    zorder=1,
)

# -- Save ----------------------------------------------------------------------
out_path = "/home/franko/CT-DRL-MA/latex/figures/tier_progression.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_path}")
