#!/usr/bin/env python3
"""Generate a clean action-selection pipeline diagram for Chapter 6.

Top-to-bottom flow with two clearly separated stages:
  Stage 1: State → CNN → Source Head → mask → ε-greedy → src_pos (or IDLE)
  Stage 2: src context → Dest Head → mask → ε-greedy → dest_pos → Resolve → Action

Output: latex/figures/action_selection_pipeline.png at 200 DPI.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# Colors matching thesis style
CLR_INPUT = "#5B7DAD"       # blue-grey — inputs
CLR_CNN = "#3A6FA0"         # darker blue — CNN backbone
CLR_SOURCE = "#E8A030"      # orange — source stage
CLR_DEST = "#6A5ACD"        # purple — destination stage
CLR_MASK = "#888888"        # grey — masks
CLR_OUTPUT = "#B03030"      # red — final output
CLR_IDLE = "#999999"        # grey — IDLE path
CLR_DECISION = "#2A8C4A"    # green — ε-greedy decisions
CLR_FEAT = "#3AADA8"        # teal — feature extraction
CLR_BG_S1 = "#FFF5E6"      # warm tint — Stage 1 background
CLR_BG_S2 = "#EEE8F8"      # cool tint — Stage 2 background


def box(ax, x, y, w, h, text, subtext, color, text_color="white",
        fontsize=10, subsize=7):
    """Draw a rounded box with title and subtitle."""
    rect = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.15", facecolor=color, edgecolor="white",
        linewidth=1.5, zorder=5)
    ax.add_patch(rect)
    if subtext:
        ax.text(x, y + 0.15, text, fontsize=fontsize, fontweight="bold",
                fontfamily="serif", color=text_color, ha="center", va="center",
                zorder=6)
        ax.text(x, y - 0.25, subtext, fontsize=subsize, fontfamily="serif",
                color=text_color, ha="center", va="center", zorder=6,
                alpha=0.85)
    else:
        ax.text(x, y, text, fontsize=fontsize, fontweight="bold",
                fontfamily="serif", color=text_color, ha="center", va="center",
                zorder=6)


def arrow(ax, x1, y1, x2, y2, color="#555555", style="-",
          connectionstyle="arc3,rad=0"):
    """Draw a simple arrow."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->,head_length=0.3,head_width=0.2",
                    color=color, lw=1.5, linestyle=style,
                    connectionstyle=connectionstyle),
                zorder=4)


def label(ax, x, y, text, color="#555555", fontsize=7, ha="center"):
    """Small annotation label on an arrow."""
    ax.text(x, y, text, fontsize=fontsize, fontfamily="serif",
            color=color, ha=ha, va="center", zorder=7,
            fontstyle="italic")


def draw_pipeline():
    fig, ax = plt.subplots(figsize=(12, 16))

    # =====================================================================
    # Stage 1 background
    # =====================================================================
    ax.add_patch(FancyBboxPatch(
        (-4.8, 1.8), 9.6, 10.5,
        boxstyle="round,pad=0.3", facecolor=CLR_BG_S1, edgecolor="#D0C0A0",
        linewidth=1.5, zorder=0, alpha=0.6))
    ax.text(-4.2, 11.8, "Stage 1: Source Selection",
            fontsize=12, fontweight="bold", fontfamily="serif",
            color="#A07020", va="center")

    # Stage 2 background
    ax.add_patch(FancyBboxPatch(
        (-4.8, -7.2), 9.6, 9.2,
        boxstyle="round,pad=0.3", facecolor=CLR_BG_S2, edgecolor="#B0A0D0",
        linewidth=1.5, zorder=0, alpha=0.6))
    ax.text(-4.2, 1.5, "Stage 2: Destination Selection",
            fontsize=12, fontweight="bold", fontfamily="serif",
            color="#6040A0", va="center")

    # =====================================================================
    # Stage 1: Source Selection (top to bottom)
    # =====================================================================

    # State Tensor
    box(ax, 0, 11, 3.0, 0.8, "State Tensor",
        "(14, 15, 1160, 5)", CLR_INPUT)

    # CNN Backbone
    box(ax, 0, 9.5, 3.0, 0.8, "CNN Backbone",
        "4–6 Conv3D layers", CLR_CNN)
    arrow(ax, 0, 10.6, 0, 9.9)

    # Feature Map
    box(ax, 0, 8.0, 3.0, 0.8, "Feature Map",
        "(64, 15, 290, 5)", CLR_FEAT)
    arrow(ax, 0, 9.1, 0, 8.4)

    # Source Head
    box(ax, 0, 6.3, 3.0, 0.8, "Source Head",
        "Conv1×1 → Q_src (21,750+1)", CLR_SOURCE)
    arrow(ax, 0, 7.6, 0, 6.7)
    label(ax, 0.3, 7.15, "feat_map")

    # Source Mask (to the left)
    box(ax, -3.2, 6.3, 2.2, 0.7, "Source Mask",
        "(15, 290, 5)", CLR_MASK)
    arrow(ax, -2.1, 6.3, -1.5, 6.3)
    label(ax, -1.8, 6.55, "mask → −∞")

    # IDLE MLP (to the right)
    box(ax, 3.2, 6.3, 2.0, 0.7, "IDLE MLP",
        "bias = −5.0", CLR_IDLE)
    arrow(ax, 1.5, 6.3, 2.2, 6.3)
    label(ax, 1.85, 6.55, "append Q_idle")

    # ε-greedy
    box(ax, 0, 4.6, 2.5, 0.7, "ε-greedy",
        "argmax or random", CLR_DECISION)
    arrow(ax, 0, 5.9, 0, 4.95)

    # Decision diamond — IDLE or position?
    # IDLE branch (left)
    box(ax, -2.8, 3.2, 2.0, 0.7, "IDLE Result",
        "no crane move", CLR_IDLE)
    arrow(ax, -0.8, 4.25, -2.0, 3.55, color=CLR_IDLE, style="--")
    label(ax, -1.8, 4.1, "idx = IDLE", color=CLR_IDLE)

    # src_pos (right → continues to Stage 2)
    box(ax, 0, 3.2, 2.2, 0.7, "src_pos",
        "source position", CLR_SOURCE, text_color="white")
    arrow(ax, 0, 4.25, 0, 3.55)
    label(ax, 0.3, 3.9, "idx < 21,750")

    # =====================================================================
    # Stage 2: Destination Selection
    # =====================================================================

    # Source Feature extraction
    box(ax, 2.8, 0.8, 2.2, 0.7, "Source Feat",
        "feat_map[src_pos]", CLR_FEAT)
    arrow(ax, 0.6, 2.85, 2.0, 1.15, color=CLR_SOURCE,
          connectionstyle="arc3,rad=0.2")
    label(ax, 2.0, 2.2, "extract\nat position", color=CLR_SOURCE)

    # Global Feature
    box(ax, -2.8, 0.8, 2.2, 0.7, "Global Feat",
        "OccupiedPooling (128)", CLR_FEAT)
    # Arrow from Feature Map down to Global Feat
    arrow(ax, -0.8, 7.6, -2.8, 1.15, color=CLR_FEAT,
          connectionstyle="arc3,rad=0.25")
    label(ax, -2.8, 4.5, "max+mean→FC", color=CLR_FEAT, ha="center")

    # dest_mask_fn (to the left)
    box(ax, -3.5, -0.8, 2.4, 0.7, "dest_mask_fn",
        "context-dependent", CLR_MASK)
    arrow(ax, 0, 2.85, -3.0, -0.45, color=CLR_MASK,
          connectionstyle="arc3,rad=0.3")
    label(ax, -2.5, 1.2, "src_pos →\n(row, split, tier)", color=CLR_MASK)

    # Dest Mask
    box(ax, -1.5, -2.0, 2.2, 0.7, "Dest Mask",
        "(15, 290, 5)", CLR_MASK)
    arrow(ax, -3.5, -1.15, -2.0, -1.65, color=CLR_MASK)

    # Dest Head
    box(ax, 0, -3.5, 3.5, 0.9, "Dest Head",
        "Query + Additive Fusion → Q_dst (21,750)", CLR_DEST)
    # Arrows into dest head
    arrow(ax, 2.8, 0.45, 1.2, -3.1, color=CLR_FEAT,
          connectionstyle="arc3,rad=-0.15")  # source feat
    label(ax, 2.8, -1.2, "source (64)", color=CLR_FEAT)
    arrow(ax, -2.8, 0.45, -1.2, -3.1, color=CLR_FEAT,
          connectionstyle="arc3,rad=0.15")  # global feat
    label(ax, -2.8, -1.2, "global (128)", color=CLR_FEAT)
    arrow(ax, -1.0, -2.35, -0.5, -3.05, color=CLR_MASK)  # dest mask
    label(ax, -1.2, -2.7, "mask → −∞", color=CLR_MASK)

    # ε-greedy (Stage 2)
    box(ax, 0, -5.0, 2.5, 0.7, "ε-greedy",
        "argmax or random", CLR_DECISION)
    arrow(ax, 0, -3.95, 0, -4.65)

    # Resolve
    box(ax, 0, -6.2, 3.2, 0.8, "Resolve",
        "dest_pos → region → move_type", CLR_OUTPUT,
        text_color="white")
    arrow(ax, 0, -5.35, 0, -5.8)

    # =====================================================================
    # Axes
    # =====================================================================
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-7.5, 12.5)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    fig = draw_pipeline()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "action_selection_pipeline.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_path}")
