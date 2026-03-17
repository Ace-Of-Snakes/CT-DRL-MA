#!/usr/bin/env python3
"""Generate move-types overlay on the terminal layout diagram (Chapter 4).

Uses the same visual language as terminal_layout.png (Figure 4.1) — same
region colors, grid, and proportions — but focuses on a representative section
and overlays elegant directional arrows for all nine move types plus IDLE.

Output: latex/figures/move_types_layout.png at 200 DPI.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
N_BAYS = 58
N_ROWS = 15

QUEUE_ROWS = (0, 1)
YARD_ROWS = (2, 6)
PARKING_ROW = 7
RAIL_ROWS = (8, 14)

# Colors — same as gen_terminal_layout.py
CLR_RAIL = "#C8D8E8"
CLR_PARKING = "#E8D8C8"
CLR_YARD = "#D8E8D0"
CLR_QUEUE = "#E0D8E8"
CLR_GRID = "#CCCCCC"
CLR_REGION_BORDER = "#888888"

# Arrow colors
CLR_IMPORT = "#1B7A3D"
CLR_EXPORT = "#B03030"
CLR_PARK = "#6A5ACD"
CLR_RESTACK = "#D07020"
CLR_DIRECT = "#8B008B"
CLR_IDLE = "#999999"


def draw_arrow(ax, start, end, color, linestyle="-", lw=1.8,
               connectionstyle="arc3,rad=0"):
    """Draw a clean annotate-based arrow."""
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(
                    arrowstyle="->,head_length=0.5,head_width=0.3",
                    color=color, lw=lw, linestyle=linestyle,
                    connectionstyle=connectionstyle,
                    shrinkA=1, shrinkB=1),
                zorder=10)


def draw_label(ax, x, y, text, color):
    """Draw a label with white background box."""
    ax.text(x, y, text, fontsize=7, fontfamily="monospace",
            fontweight="bold", color=color, ha="center", va="center",
            zorder=11,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=color, alpha=0.92, linewidth=0.6))


def draw_move_types():
    fig, ax = plt.subplots(figsize=(16, 9))

    view_left = 1
    view_right = 50

    # ----- Background region fills -----
    for y0, h, clr in [
        (RAIL_ROWS[0] - 0.5,  RAIL_ROWS[1] - RAIL_ROWS[0] + 1,  CLR_RAIL),
        (PARKING_ROW - 0.5,   1,                                  CLR_PARKING),
        (YARD_ROWS[0] - 0.5,  YARD_ROWS[1] - YARD_ROWS[0] + 1,   CLR_YARD),
        (QUEUE_ROWS[0] - 0.5, QUEUE_ROWS[1] - QUEUE_ROWS[0] + 1,  CLR_QUEUE),
    ]:
        ax.add_patch(mpatches.FancyBboxPatch(
            (view_left, y0), view_right - view_left, h,
            boxstyle="square,pad=0", facecolor=clr, edgecolor="none", zorder=0))

    # ----- Region boundary lines -----
    for r in (0, 2, 7, 8, 15):
        ax.axhline(r - 0.5, color=CLR_REGION_BORDER, linewidth=1.0, zorder=2)

    # ----- Light grid -----
    for r in range(N_ROWS + 1):
        if r not in (0, 2, 7, 8, 15):
            ax.axhline(r - 0.5, color=CLR_GRID, linewidth=0.25, zorder=1)

    # ----- Rail track lines -----
    for r in range(RAIL_ROWS[0], RAIL_ROWS[1] + 1):
        ax.plot([view_left, view_right], [r - 0.12, r - 0.12],
                color="#999999", linewidth=0.4, zorder=1)
        ax.plot([view_left, view_right], [r + 0.12, r + 0.12],
                color="#999999", linewidth=0.4, zorder=1)

    # ----- Region labels (right side) -----
    label_x = view_right + 0.8
    lp = dict(fontsize=11, fontweight="bold", va="center", ha="left",
              fontfamily="serif")
    ax.text(label_x, (RAIL_ROWS[0] + RAIL_ROWS[1]) / 2,
            "Rail\n(rows 0–6)", color="#2A5080", **lp)
    ax.text(label_x, PARKING_ROW,
            "Parking\n(row 7)", color="#806030", **lp)
    ax.text(label_x, (YARD_ROWS[0] + YARD_ROWS[1]) / 2,
            "Yard\n(rows 8–12)", color="#306830", **lp)
    ax.text(label_x, (QUEUE_ROWS[0] + QUEUE_ROWS[1]) / 2,
            "Queue\n(rows 13–14)", color="#503068", **lp)

    # ----- Row numbers (left side) -----
    for r in range(N_ROWS):
        grid_row = N_ROWS - 1 - r
        ax.text(view_left - 0.5, r, str(grid_row), fontsize=6.5,
                fontfamily="serif", ha="right", va="center", color="#666666")

    # =====================================================================
    # MOVE TYPE ARROWS
    # =====================================================================
    queue_y = 0.5
    parking_y = 7.0
    yard_lo = 2.3
    yard_hi = 5.7
    yard_mid = 4.0
    rail_lo = 8.7

    # --- Left group: truck-side logistics ---

    # 1. PARK_TRUCK: Queue → Parking
    px = 7
    draw_arrow(ax, (px, queue_y + 0.2), (px, parking_y - 0.2), CLR_PARK)
    draw_label(ax, px - 3.3, (queue_y + parking_y) / 2, "PARK_TRUCK", CLR_PARK)

    # 2. TRUCK_TO_YARD: Parking → Yard
    tx = 13
    draw_arrow(ax, (tx, parking_y - 0.2), (tx, yard_hi + 0.2), CLR_IMPORT)
    draw_label(ax, tx - 4.0, (parking_y + yard_hi) / 2, "TRUCK_TO_YARD", CLR_IMPORT)

    # 3. YARD_TO_TRUCK: Yard → Parking
    yx = 19
    draw_arrow(ax, (yx, yard_hi + 0.2), (yx, parking_y - 0.2), CLR_EXPORT)
    draw_label(ax, yx + 4.0, (parking_y + yard_hi) / 2, "YARD_TO_TRUCK", CLR_EXPORT)

    # --- Centre: yard-internal ---

    # 4. YARD_TO_YARD: restack (arc above)
    yy1, yy2 = 22, 28
    draw_arrow(ax, (yy1, yard_hi), (yy2, yard_hi), CLR_RESTACK,
               connectionstyle="arc3,rad=-0.35")
    draw_label(ax, (yy1 + yy2) / 2, parking_y + 0.3, "YARD_TO_YARD", CLR_RESTACK)

    # 5. IDLE: small circular arrow
    idle_x = 25
    idle_y = yard_lo + 0.3
    draw_arrow(ax, (idle_x - 0.8, idle_y), (idle_x + 0.8, idle_y), CLR_IDLE,
               connectionstyle="arc3,rad=-1.0", linestyle=":", lw=1.2)
    draw_label(ax, idle_x, idle_y - 1.4, "IDLE", CLR_IDLE)

    # --- Right group: rail-side logistics ---

    # 6. TRAIN_TO_YARD: Rail → Yard
    rx1 = 31
    draw_arrow(ax, (rx1, rail_lo - 0.2), (rx1, yard_lo - 0.2), CLR_IMPORT)
    draw_label(ax, rx1 - 4.0, (rail_lo + yard_lo) / 2, "TRAIN_TO_YARD", CLR_IMPORT)

    # 7. YARD_TO_TRAIN: Yard → Rail
    rx2 = 37
    draw_arrow(ax, (rx2, yard_lo - 0.2), (rx2, rail_lo - 0.2), CLR_EXPORT)
    draw_label(ax, rx2 + 4.0, (rail_lo + yard_lo) / 2, "YARD_TO_TRAIN", CLR_EXPORT)

    # --- Far right: direct transfers (bypass yard, curves outward) ---

    # 8. TRAIN_TO_TRUCK: Rail → Parking (dashed, curves right to show bypass)
    dx1 = 42
    draw_arrow(ax, (dx1, rail_lo + 1.5), (dx1 - 1, parking_y + 0.2),
               CLR_DIRECT, linestyle="--", connectionstyle="arc3,rad=0.30", lw=1.8)
    draw_label(ax, dx1 + 4.0, rail_lo + 1.0, "TRAIN_TO_TRUCK", CLR_DIRECT)

    # 9. TRUCK_TO_TRAIN: Parking → Rail (dashed, curves right to show bypass)
    dx2 = 46
    draw_arrow(ax, (dx2 - 1, parking_y + 0.2), (dx2, rail_lo + 1.5),
               CLR_DIRECT, linestyle="--", connectionstyle="arc3,rad=0.30", lw=1.8)
    draw_label(ax, dx2 + 4.0, parking_y + 0.8, "TRUCK_TO_TRAIN", CLR_DIRECT)

    # ----- Legend -----
    legend_elements = [
        plt.Line2D([0], [0], color=CLR_IMPORT, linewidth=1.5,
                   label="Import (inbound to yard)"),
        plt.Line2D([0], [0], color=CLR_EXPORT, linewidth=1.5,
                   label="Export (outbound from yard)"),
        plt.Line2D([0], [0], color=CLR_PARK, linewidth=1.5,
                   label="Truck parking"),
        plt.Line2D([0], [0], color=CLR_RESTACK, linewidth=1.5,
                   label="Yard restack"),
        plt.Line2D([0], [0], color=CLR_DIRECT, linewidth=1.5, linestyle="--",
                   label="Direct transfer (bypass yard)"),
        plt.Line2D([0], [0], color=CLR_IDLE, linewidth=1.2, linestyle=":",
                   label="IDLE (no movement)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8.5,
              framealpha=0.92, edgecolor="#CCCCCC", fancybox=True,
              prop={"family": "serif"}, ncol=2)

    # ----- Axes styling -----
    ax.set_xlim(view_left - 1.5, view_right + 12.0)
    ax.set_ylim(-1.8, N_ROWS + 0.8)
    # No equal aspect — allow vertical stretch for readability
    ax.axis("off")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    fig = draw_move_types()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "move_types_layout.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_path}")
