#!/usr/bin/env python3
"""Generate system architecture / class dependency diagram for Chapter 4.

Produces a layered block diagram with color-coded layers and dependency arrows.
Pure matplotlib implementation (no graphviz/networkx).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────

FIG_W, FIG_H = 14, 10
DPI = 200
OUT = Path(__file__).parent / "system_architecture.png"

# Layer colours (background fill, border, text)
COLORS = {
    "training":    {"bg": "#D6E4F0", "border": "#4A7FB5", "box": "#4A7FB5",  "text": "#FFFFFF"},
    "environment": {"bg": "#D9EDDB", "border": "#5AA861", "box": "#5AA861",  "text": "#FFFFFF"},
    "agent":       {"bg": "#FDEBD0", "border": "#D68A2E", "box": "#D68A2E",  "text": "#FFFFFF"},
    "data":        {"bg": "#E8E8E8", "border": "#888888", "box": "#888888",  "text": "#FFFFFF"},
}

# Arrow style
ARROW_KW = dict(
    arrowstyle="-|>",
    connectionstyle="arc3,rad=0.0",
    linewidth=1.3,
    mutation_scale=14,
    color="#333333",
    shrinkA=4,
    shrinkB=4,
)

ANNOTATION_KW = dict(
    fontsize=7.5,
    fontstyle="italic",
    color="#555555",
    ha="center",
    va="center",
    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
)


# ── Helper functions ───────────────────────────────────────────────────────

def draw_layer_bg(ax, y_bot, height, label, color_key):
    """Draw a coloured background band for one layer."""
    c = COLORS[color_key]
    rect = FancyBboxPatch(
        (0.3, y_bot), 13.4, height,
        boxstyle="round,pad=0.12",
        facecolor=c["bg"], edgecolor=c["border"],
        linewidth=1.8, alpha=0.55, zorder=0,
    )
    ax.add_patch(rect)
    ax.text(
        0.65, y_bot + height / 2, label,
        fontsize=9, fontweight="bold", color=c["border"],
        rotation=90, ha="center", va="center", zorder=1,
    )


def draw_box(ax, cx, cy, w, h, label, color_key, fontsize=9, sublabel=None):
    """Draw a rounded box centred at (cx, cy)."""
    c = COLORS[color_key]
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.10",
        facecolor=c["box"], edgecolor="white",
        linewidth=1.2, zorder=3,
    )
    ax.add_patch(box)
    if sublabel:
        ax.text(cx, cy + 0.12, label, fontsize=fontsize, fontweight="bold",
                color=c["text"], ha="center", va="center", zorder=4)
        ax.text(cx, cy - 0.16, sublabel, fontsize=6.5,
                color=c["text"], ha="center", va="center", zorder=4, alpha=0.9)
    else:
        ax.text(cx, cy, label, fontsize=fontsize, fontweight="bold",
                color=c["text"], ha="center", va="center", zorder=4)
    return (cx, cy)


def arrow(ax, start, end, **extra_kw):
    """Draw a dependency arrow from start (x,y) to end (x,y)."""
    kw = {**ARROW_KW, **extra_kw}
    a = FancyArrowPatch(start, end, **kw, zorder=2)
    ax.add_patch(a)
    return a


def curved_arrow(ax, start, end, rad=0.15, **extra_kw):
    """Draw a curved dependency arrow."""
    kw = {**ARROW_KW, **extra_kw}
    kw["connectionstyle"] = f"arc3,rad={rad}"
    a = FancyArrowPatch(start, end, **kw, zorder=2)
    ax.add_patch(a)
    return a


# ── Main figure ────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.set_aspect("equal")
ax.axis("off")

# Title
ax.text(
    FIG_W / 2, FIG_H - 0.35, "System Architecture",
    fontsize=16, fontweight="bold", ha="center", va="top", color="#222222",
)

# ── Layer backgrounds ──────────────────────────────────────────────────────
# Y layout (bottom to top): Data 0.3-2.0, Agent 2.3-4.2, Env 4.5-7.3, Training 7.6-9.3

draw_layer_bg(ax, 0.3,  1.7, "Data",        "data")
draw_layer_bg(ax, 2.3,  1.9, "Agent",       "agent")
draw_layer_bg(ax, 4.5,  2.8, "Environment", "environment")
draw_layer_bg(ax, 7.6,  1.7, "Training",    "training")

# ── DATA LAYER (y ~ 0.7 - 1.7) ────────────────────────────────────────────

bw, bh = 2.2, 0.55  # box width / height

scn = draw_box(ax, 4.5, 1.15, 2.6, bh, "Scenarios", "data",
               sublabel="31 scenarios, 13 tiers")
trn_sched = draw_box(ax, 8.5, 1.15, 2.0, bh, "TrainScheduler", "data")
trk_gen   = draw_box(ax, 11.5, 1.15, 2.0, bh, "TruckGenerator", "data")

# ── AGENT LAYER (y ~ 2.7 - 3.8) ───────────────────────────────────────────

agent = draw_box(ax, 5.0, 3.55, 2.4, 0.65, "DQN Agent", "agent", fontsize=10)

qnet = draw_box(ax, 9.0, 3.85, 3.2, 0.55, "QNetwork", "agent",
                sublabel="CNN + SourceHead + DestHead")
replay = draw_box(ax, 9.0, 2.95, 2.4, 0.50, "ReplayBuffer", "agent")

# Agent -> QNetwork
arrow(ax, (agent[0] + 1.2, agent[1] + 0.10), (qnet[0] - 1.6, qnet[1]))

# Agent -> ReplayBuffer
arrow(ax, (agent[0] + 1.2, agent[1] - 0.12), (replay[0] - 1.2, replay[1]))

# ── ENVIRONMENT LAYER (y ~ 4.9 - 6.9) ─────────────────────────────────────

env = draw_box(ax, 5.0, 6.55, 2.6, 0.70, "TerminalEnv", "environment", fontsize=11)

state_enc   = draw_box(ax, 2.5, 5.55, 2.2, 0.55, "StateEncoder", "environment")
reward_eng  = draw_box(ax, 5.0, 5.55, 2.0, 0.55, "RewardEngine", "environment")
fac_coord   = draw_box(ax, 8.0, 5.55, 2.8, 0.55, "FacilityCoordinator", "environment")
crane_ctrl  = draw_box(ax, 11.5, 5.55, 2.2, 0.55, "CraneController", "environment")

# Sub-facilities under FacilityCoordinator
storage = draw_box(ax, 6.5, 4.85, 1.8, 0.45, "StorageYard", "environment", fontsize=7.5)
rail    = draw_box(ax, 8.5, 4.85, 1.5, 0.45, "RailYard", "environment", fontsize=7.5)
parking = draw_box(ax, 10.5, 4.85, 1.8, 0.45, "ParkingArea", "environment", fontsize=7.5)

# TerminalEnv -> sub-components
arrow(ax, (env[0] - 0.9, env[1] - 0.35), (state_enc[0] + 0.5, state_enc[1] + 0.28))
arrow(ax, (env[0], env[1] - 0.35), (reward_eng[0], reward_eng[1] + 0.28))
arrow(ax, (env[0] + 0.9, env[1] - 0.35), (fac_coord[0] - 0.7, fac_coord[1] + 0.28))
arrow(ax, (env[0] + 1.3, env[1] - 0.20), (crane_ctrl[0] - 0.8, crane_ctrl[1] + 0.28),
      connectionstyle="arc3,rad=-0.10")

# FacilityCoordinator -> facilities
arrow(ax, (fac_coord[0] - 0.8, fac_coord[1] - 0.28), (storage[0] + 0.4, storage[1] + 0.23))
arrow(ax, (fac_coord[0], fac_coord[1] - 0.28), (rail[0], rail[1] + 0.23))
arrow(ax, (fac_coord[0] + 0.8, fac_coord[1] - 0.28), (parking[0] - 0.4, parking[1] + 0.23))

# ── TRAINING LAYER (y ~ 7.9 - 9.0) ────────────────────────────────────────

curr = draw_box(ax, 5.0, 8.45, 2.8, 0.65, "CurriculumTrainer", "training", fontsize=10)
tut  = draw_box(ax, 9.5, 8.45, 2.4, 0.55, "TutorialRunner", "training")

# CurriculumTrainer -> TutorialRunner
arrow(ax, (curr[0] + 1.4, curr[1]), (tut[0] - 1.2, tut[1]))

# ── CROSS-LAYER ARROWS ────────────────────────────────────────────────────

# CurriculumTrainer -> TerminalEnv
arrow(ax, (curr[0] - 0.5, curr[1] - 0.33), (env[0] - 0.5, env[1] + 0.35),
      color="#4A7FB5", linewidth=1.6)

# CurriculumTrainer -> Agent
curved_arrow(ax, (curr[0] - 1.2, curr[1] - 0.33), (agent[0] - 0.8, agent[1] + 0.33),
             rad=0.15, color="#4A7FB5", linewidth=1.6)

# TutorialRunner -> TerminalEnv
arrow(ax, (tut[0] - 0.3, tut[1] - 0.28), (env[0] + 1.0, env[1] + 0.35),
      color="#4A7FB5", linewidth=1.4)

# TutorialRunner -> Scenarios
curved_arrow(ax, (tut[0] - 0.5, tut[1] - 0.28), (scn[0] + 0.8, scn[1] + 0.28),
             rad=0.25, color="#4A7FB5", linewidth=1.4)

# StateEncoder -> Agent  (state tensor)
curved_arrow(ax, (state_enc[0] - 0.2, state_enc[1] - 0.28),
             (agent[0] - 1.0, agent[1] + 0.33),
             rad=0.20, color="#5AA861", linewidth=1.8, linestyle="--")
# Annotation: state tensor shape
ax.text(1.85, 4.45, "state tensor\n(14, 15, 1160, 5)",
        **ANNOTATION_KW)

# Agent -> TerminalEnv  (action)
curved_arrow(ax, (agent[0] + 0.5, agent[1] + 0.33),
             (env[0] + 0.5, env[1] - 0.35),
             rad=-0.15, color="#D68A2E", linewidth=1.8, linestyle="--")
# Annotation: action tuple
ax.text(6.35, 4.55, "action\n(source, dest)",
        **ANNOTATION_KW)

# TerminalEnv -> Agent  (masks)
curved_arrow(ax, (env[0] - 1.0, env[1] - 0.35),
             (agent[0] - 0.5, agent[1] + 0.33),
             rad=-0.12, color="#5AA861", linewidth=1.2, linestyle=":")
ax.text(3.2, 4.55, "masks",
        fontsize=7, fontstyle="italic", color="#555555", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))

# TrainScheduler / TruckGenerator -> TerminalEnv
curved_arrow(ax, (trn_sched[0], trn_sched[1] + 0.28),
             (env[0] + 1.0, env[1] - 0.35),
             rad=-0.30, color="#888888", linewidth=1.2)
curved_arrow(ax, (trk_gen[0], trk_gen[1] + 0.28),
             (crane_ctrl[0], crane_ctrl[1] - 0.28),
             rad=-0.20, color="#888888", linewidth=1.2)

# ── LEGEND ─────────────────────────────────────────────────────────────────

legend_x, legend_y = 1.2, 9.55
legend_items = [
    ("Training Layer",    COLORS["training"]["box"]),
    ("Environment Layer", COLORS["environment"]["box"]),
    ("Agent Layer",       COLORS["agent"]["box"]),
    ("Data Layer",        COLORS["data"]["box"]),
]
for i, (lbl, col) in enumerate(legend_items):
    x = legend_x + i * 2.8
    rect = FancyBboxPatch(
        (x, legend_y - 0.12), 0.28, 0.24,
        boxstyle="round,pad=0.03", facecolor=col, edgecolor="white",
        linewidth=0.8, zorder=5,
    )
    ax.add_patch(rect)
    ax.text(x + 0.42, legend_y, lbl, fontsize=7.5, va="center", color="#333333", zorder=5)

# Dashed-line legend entries
dash_x = legend_x + len(legend_items) * 2.8 + 0.3
ax.annotate("", xy=(dash_x + 0.5, legend_y), xytext=(dash_x, legend_y),
            arrowprops=dict(arrowstyle="-|>", linestyle="--", color="#333333",
                            linewidth=1.2, mutation_scale=12))
ax.text(dash_x + 0.65, legend_y, "data flow", fontsize=7.5, va="center", color="#333333")

# ── Save ───────────────────────────────────────────────────────────────────

fig.tight_layout(pad=0.3)
fig.savefig(OUT, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved -> {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
