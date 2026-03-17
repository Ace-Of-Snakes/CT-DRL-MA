#!/usr/bin/env python3
"""Generate a bird's-eye terminal layout diagram for the master's thesis (Chapter 4).

The terminal grid:
  - 58 bays (X-axis)
  - 15 rows (Y-axis, bottom to top in diagram):
      Rows 13-14 (display 0-1):   Queue (external truck queue)
      Rows 8-12  (display 2-6):   Yard (storage blocks)
      Row  7     (display 7):     Parking (truck staging)
      Rows 0-6   (display 8-14):  Rail (7 rail tracks)
  - 2 RMG cranes with 4-bay overlap zone
  - Special zones: reefer (outer bays), dangerous goods (centre), swap bodies (row 1)

Output: latex/figures/terminal_layout.png at 200 DPI.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
N_BAYS = 58
N_ROWS = 15

# Display row ranges (bottom-up in diagram)
QUEUE_ROWS = (0, 1)       # grid rows 13-14
YARD_ROWS = (2, 6)        # grid rows 8-12
PARKING_ROW = 7           # grid row 7
RAIL_ROWS = (8, 14)       # grid rows 0-6

# Crane zones (0-based bay indices)
CRANE1_BAYS = (0, 30)     # bays 1-31
CRANE2_BAYS = (27, 57)    # bays 28-58
# Overlap: bays 28-31 (4 bays)

# Special zones (yard rows are 1-based in config, mapped to display rows)
# Yard row 1 (1-based) = display row 2 (bottom yard row)
# Yard row 2 = display 3, row 3 = display 4, row 4 = display 5, row 5 = display 6
SWAP_BODY_DISPLAY_ROW = 6                    # yard row 1 (nearest to parking)
REEFER_BAYS = (0, 57)                        # bay 1 and bay 58 (0-based)
DG_CENTER_BAY = 28                           # bay 29 (0-based), center of 58 bays
DG_HALFWIDTH = 3                             # ±3 bays from centre
DG_DISPLAY_ROWS = (2, 3, 4)                  # yard rows 5, 4, 3 (centre rows, away from parking)

# Colors
CLR_RAIL = "#C8D8E8"
CLR_PARKING = "#E8D8C8"
CLR_YARD = "#D8E8D0"
CLR_QUEUE = "#E0D8E8"
CLR_OVERLAP = "#FFE0A0"
CLR_CRANE1 = "#2070B0"
CLR_CRANE2 = "#B03030"
CLR_CONTAINER_20 = "#4A90D9"
CLR_CONTAINER_40 = "#D94A4A"
CLR_TRAIN = "#5A5A5A"
CLR_TRUCK = "#E8A030"
CLR_GRID = "#CCCCCC"
CLR_REGION_BORDER = "#888888"

# Special zone colors (subtle tints over the yard green)
CLR_REEFER = "#C0E0E8"       # cool blue-green tint
CLR_DG = "#E8D0C0"           # warm amber tint
CLR_SWAP_BODY = "#D0D0E0"    # light lavender tint


def draw_terminal():
    fig, ax = plt.subplots(figsize=(16, 9.5))

    # ----- Background region fills -----
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, RAIL_ROWS[0] - 0.5), N_BAYS, RAIL_ROWS[1] - RAIL_ROWS[0] + 1,
        boxstyle="square,pad=0", facecolor=CLR_RAIL, edgecolor="none", zorder=0))

    ax.add_patch(mpatches.FancyBboxPatch(
        (0, PARKING_ROW - 0.5), N_BAYS, 1,
        boxstyle="square,pad=0", facecolor=CLR_PARKING, edgecolor="none", zorder=0))

    ax.add_patch(mpatches.FancyBboxPatch(
        (0, YARD_ROWS[0] - 0.5), N_BAYS, YARD_ROWS[1] - YARD_ROWS[0] + 1,
        boxstyle="square,pad=0", facecolor=CLR_YARD, edgecolor="none", zorder=0))

    ax.add_patch(mpatches.FancyBboxPatch(
        (0, QUEUE_ROWS[0] - 0.5), N_BAYS, QUEUE_ROWS[1] - QUEUE_ROWS[0] + 1,
        boxstyle="square,pad=0", facecolor=CLR_QUEUE, edgecolor="none", zorder=0))

    # ----- Special zone overlays (on top of yard background) -----
    # Swap body row (yard row 1 = display row 2, full width)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, SWAP_BODY_DISPLAY_ROW - 0.5), N_BAYS, 1,
        boxstyle="square,pad=0", facecolor=CLR_SWAP_BODY, edgecolor="none",
        alpha=0.7, zorder=0.5))

    # Reefer bays (bay 1 and bay 58, all yard rows)
    for bay_idx in REEFER_BAYS:
        ax.add_patch(mpatches.FancyBboxPatch(
            (bay_idx, YARD_ROWS[0] - 0.5), 1, YARD_ROWS[1] - YARD_ROWS[0] + 1,
            boxstyle="square,pad=0", facecolor=CLR_REEFER, edgecolor="none",
            alpha=0.7, zorder=0.5))

    # Dangerous goods zone (centre bays, yard rows 3-5 = display rows 4-6)
    dg_start = DG_CENTER_BAY - DG_HALFWIDTH
    dg_width = 2 * DG_HALFWIDTH + 1
    dg_row_min = min(DG_DISPLAY_ROWS)
    dg_row_max = max(DG_DISPLAY_ROWS)
    ax.add_patch(mpatches.FancyBboxPatch(
        (dg_start, dg_row_min - 0.5), dg_width, dg_row_max - dg_row_min + 1,
        boxstyle="square,pad=0", facecolor=CLR_DG, edgecolor="none",
        alpha=0.7, zorder=0.5))

    # ----- Crane zone overlays -----
    # Overlap zone highlight (draw first so crane borders are on top)
    overlap_start = CRANE2_BAYS[0]
    overlap_end = CRANE1_BAYS[1]
    overlap_rect = mpatches.FancyBboxPatch(
        (overlap_start, -0.5), overlap_end - overlap_start + 1, N_ROWS,
        boxstyle="square,pad=0", facecolor=CLR_OVERLAP, alpha=0.35,
        edgecolor="#C0A000", linewidth=1.5, linestyle="-", zorder=1)
    ax.add_patch(overlap_rect)

    # Crane 1 zone
    ax.add_patch(mpatches.FancyBboxPatch(
        (CRANE1_BAYS[0], -0.5), CRANE1_BAYS[1] - CRANE1_BAYS[0] + 1, N_ROWS,
        boxstyle="square,pad=0", facecolor="none",
        edgecolor=CLR_CRANE1, linewidth=2.5, linestyle="--", zorder=3))

    # Crane 2 zone
    ax.add_patch(mpatches.FancyBboxPatch(
        (CRANE2_BAYS[0], -0.5), CRANE2_BAYS[1] - CRANE2_BAYS[0] + 1, N_ROWS,
        boxstyle="square,pad=0", facecolor="none",
        edgecolor=CLR_CRANE2, linewidth=2.5, linestyle="--", zorder=3))

    # ----- Grid lines -----
    for r in range(N_ROWS + 1):
        lw = 0.3
        clr = CLR_GRID
        if r in (0, 2, 7, 8, 15):
            lw = 1.2
            clr = CLR_REGION_BORDER
        ax.axhline(r - 0.5, color=clr, linewidth=lw, zorder=2)

    for b in range(N_BAYS + 1):
        lw = 0.2
        clr = CLR_GRID
        if b % 10 == 0:
            lw = 0.6
            clr = "#AAAAAA"
        ax.axvline(b, color=clr, linewidth=lw, zorder=2)

    # ----- Region labels (right side) -----
    label_x = N_BAYS + 1.0
    label_props = dict(fontsize=10, fontweight="bold", va="center", ha="left",
                       fontfamily="serif")

    ax.text(label_x, (RAIL_ROWS[0] + RAIL_ROWS[1]) / 2,
            "Rail\n(rows 0\u20136)", color="#2A5080", **label_props)
    ax.text(label_x, PARKING_ROW,
            "Parking\n(row 7)", color="#806030", **label_props)
    ax.text(label_x, (YARD_ROWS[0] + YARD_ROWS[1]) / 2,
            "Yard\n(rows 8\u201312)", color="#306830", **label_props)
    ax.text(label_x, (QUEUE_ROWS[0] + QUEUE_ROWS[1]) / 2,
            "Queue\n(rows 13\u201314)", color="#503068", **label_props)

    # ----- Row numbers (left side) -----
    for r in range(N_ROWS):
        grid_row = N_ROWS - 1 - r
        ax.text(-0.8, r, str(grid_row), fontsize=6.5, fontfamily="serif",
                ha="right", va="center", color="#666666")

    ax.text(-1.8, N_ROWS / 2 - 0.5, "Row", fontsize=9, fontfamily="serif",
            ha="center", va="center", color="#666666", rotation=90)

    # ----- Bay axis labels (bottom) -----
    for b in range(0, N_BAYS, 5):
        ax.text(b + 0.5, -1.2, str(b + 1), fontsize=6.5, fontfamily="serif",
                ha="center", va="top", color="#666666")
    ax.text(N_BAYS - 0.5, -1.2, str(N_BAYS), fontsize=6.5, fontfamily="serif",
            ha="center", va="top", color="#666666")
    ax.text(N_BAYS / 2, -2.1, "Bay", fontsize=9, fontfamily="serif",
            ha="center", va="top", color="#666666")

    # ----- Crane labels (top) -----
    ax.text((CRANE1_BAYS[0] + CRANE1_BAYS[1] + 1) / 2, N_ROWS + 0.2,
            "Crane 1", fontsize=11, fontweight="bold", fontfamily="serif",
            ha="center", va="bottom", color=CLR_CRANE1)
    ax.text((CRANE2_BAYS[0] + CRANE2_BAYS[1] + 1) / 2, N_ROWS + 0.2,
            "Crane 2", fontsize=11, fontweight="bold", fontfamily="serif",
            ha="center", va="bottom", color=CLR_CRANE2)

    # Overlap label with arrow
    ax.text((overlap_start + overlap_end + 1) / 2, N_ROWS + 1.5,
            "Overlap\n(4 bays)", fontsize=8, fontfamily="serif",
            ha="center", va="bottom", color="#907000", fontstyle="italic")
    ax.annotate("", xy=((overlap_start + overlap_end + 1) / 2, N_ROWS - 0.1),
                xytext=((overlap_start + overlap_end + 1) / 2, N_ROWS + 1.4),
                arrowprops=dict(arrowstyle="->", color="#907000", lw=1.2))

    # ----- Example containers in yard -----
    def draw_container(bay, row, width_bays, color, label=None):
        rect = mpatches.FancyBboxPatch(
            (bay + 0.1, row - 0.35), width_bays - 0.2, 0.7,
            boxstyle="round,pad=0.05", facecolor=color, edgecolor="white",
            linewidth=1.0, zorder=5)
        ax.add_patch(rect)
        if label:
            ax.text(bay + width_bays / 2, row, label, fontsize=5.5,
                    fontfamily="serif", ha="center", va="center",
                    color="white", fontweight="bold", zorder=6)

    # 20ft containers (1 bay wide)
    draw_container(5, 4, 1, CLR_CONTAINER_20, "20ft")
    draw_container(6, 4, 1, CLR_CONTAINER_20)
    draw_container(10, 3, 1, CLR_CONTAINER_20, "20ft")
    draw_container(22, 5, 1, CLR_CONTAINER_20)
    draw_container(40, 3, 1, CLR_CONTAINER_20)
    draw_container(41, 3, 1, CLR_CONTAINER_20, "20ft")

    # 40ft containers (2 bays wide)
    draw_container(14, 4, 2, CLR_CONTAINER_40, "40ft")
    draw_container(20, 4, 2, CLR_CONTAINER_40, "40ft")
    draw_container(33, 5, 2, CLR_CONTAINER_40)
    draw_container(45, 4, 2, CLR_CONTAINER_40, "40ft")
    draw_container(50, 3, 2, CLR_CONTAINER_40)

    # Containers on rail (being loaded/unloaded)
    draw_container(8, 10, 2, CLR_CONTAINER_40)
    draw_container(12, 10, 1, CLR_CONTAINER_20)

    # ----- Train on rail track (display row 10) -----
    train_row = 10
    train_start = 3
    train_length = 18

    # Locomotive
    loco = mpatches.FancyBboxPatch(
        (train_start, train_row - 0.40), 2, 0.80,
        boxstyle="round,pad=0.08", facecolor="#404040", edgecolor="#222222",
        linewidth=1.2, zorder=4)
    ax.add_patch(loco)
    ax.text(train_start + 1, train_row, "Loco", fontsize=5, fontfamily="serif",
            ha="center", va="center", color="white", fontweight="bold", zorder=6)

    # Flatcars
    for i in range(train_start + 2, train_start + train_length, 2):
        car = mpatches.FancyBboxPatch(
            (i + 0.05, train_row - 0.35), 1.9, 0.70,
            boxstyle="round,pad=0.05", facecolor="#787878", edgecolor="#555555",
            linewidth=0.8, zorder=4)
        ax.add_patch(car)

    # Rail track lines (parallel rails for all 7 tracks)
    for r in range(RAIL_ROWS[0], RAIL_ROWS[1] + 1):
        ax.plot([0, N_BAYS], [r - 0.15, r - 0.15], color="#888888",
                linewidth=0.5, zorder=1, linestyle="-")
        ax.plot([0, N_BAYS], [r + 0.15, r + 0.15], color="#888888",
                linewidth=0.5, zorder=1, linestyle="-")

    # ----- Trucks in parking -----
    def draw_truck(bay, row):
        body = mpatches.FancyBboxPatch(
            (bay + 0.15, row - 0.30), 1.7, 0.60,
            boxstyle="round,pad=0.06", facecolor=CLR_TRUCK, edgecolor="#A07020",
            linewidth=1.0, zorder=5)
        ax.add_patch(body)
        cab = mpatches.FancyBboxPatch(
            (bay - 0.3, row - 0.22), 0.5, 0.44,
            boxstyle="round,pad=0.04", facecolor="#D08020", edgecolor="#A07020",
            linewidth=0.8, zorder=5)
        ax.add_patch(cab)

    draw_truck(15, PARKING_ROW)
    ax.text(16.0, PARKING_ROW, "Truck", fontsize=5, fontfamily="serif",
            ha="center", va="center", color="white", fontweight="bold", zorder=6)
    draw_truck(38, PARKING_ROW)

    # ----- Trucks in queue -----
    for qbay in [4, 8, 12, 25, 30, 48]:
        qrow = 0 if qbay % 8 < 4 else 1
        qbody = mpatches.FancyBboxPatch(
            (qbay + 0.15, qrow - 0.25), 1.5, 0.50,
            boxstyle="round,pad=0.04", facecolor="#C89828", edgecolor="#A07020",
            linewidth=0.6, zorder=5, alpha=0.6)
        ax.add_patch(qbody)

    # ----- Legend -----
    legend_elements = [
        mpatches.Patch(facecolor=CLR_CONTAINER_20, edgecolor="white",
                       label="20ft container"),
        mpatches.Patch(facecolor=CLR_CONTAINER_40, edgecolor="white",
                       label="40ft container"),
        mpatches.Patch(facecolor="#787878", edgecolor="#555555",
                       label="Train (flatcars)"),
        mpatches.Patch(facecolor=CLR_TRUCK, edgecolor="#A07020",
                       label="Truck"),
        mpatches.Patch(facecolor=CLR_OVERLAP, edgecolor="#C0A000", alpha=0.5,
                       label="Crane overlap zone"),
        plt.Line2D([0], [0], color=CLR_CRANE1, linewidth=2.5, linestyle="--",
                   label="Crane 1 zone"),
        plt.Line2D([0], [0], color=CLR_CRANE2, linewidth=2.5, linestyle="--",
                   label="Crane 2 zone"),
        mpatches.Patch(facecolor=CLR_REEFER, edgecolor="#999999", alpha=0.7,
                       label="Reefer zone (outer bays)"),
        mpatches.Patch(facecolor=CLR_DG, edgecolor="#999999", alpha=0.7,
                       label="Dangerous goods zone (centre)"),
        mpatches.Patch(facecolor=CLR_SWAP_BODY, edgecolor="#999999", alpha=0.7,
                       label="Swap body row"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7.5,
              framealpha=0.9, edgecolor="#CCCCCC", fancybox=True,
              prop={"family": "serif"}, ncol=1)

    # ----- Axes styling -----
    ax.set_xlim(-2.5, N_BAYS + 6.5)
    ax.set_ylim(-2.5, N_ROWS + 6.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(N_BAYS / 2, N_ROWS + 2.2,
            "Container Terminal Layout (Bird's-Eye View)",
            fontsize=14, fontweight="bold", fontfamily="serif",
            ha="center", va="bottom")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    fig = draw_terminal()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "terminal_layout.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_path}")
