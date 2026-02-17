#!/usr/bin/env python3
"""Crane movement visualiser — renders an animated terminal top-down view.

Reads the CSV files produced by ``MoveCSVLogger`` and draws animated arrows
showing container moves over time.  Each move type gets a distinct colour.

Usage::

    python tools/visualise_crane_movements.py \
        --moves outputs/phase2/deeper_munchausen/logs/moves_day0.csv \
        --events outputs/phase2/deeper_munchausen/logs/events_day0.csv \
        --output crane_animation.gif

Or in interactive mode (no --output flag) to view a matplotlib window.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

# ── Terminal layout constants ────────────────────────────────────────────
# These must match the unified state encoder layout:
#   Rows 0..6   Rail tracks
#   Row  7      Parking
#   Rows 8..12  Yard
#   Rows 13..14 Queue

N_TRACKS = 7
N_YARD_ROWS = 5
N_BAYS = 58

REGION_Y: Dict[str, float] = {}
# Rail tracks at the top
for t in range(N_TRACKS):
    REGION_Y[f"RAIL_{t}"] = N_TRACKS - t  # top-down
REGION_Y["PARKING"] = -1.0
for r in range(N_YARD_ROWS):
    REGION_Y[f"YARD_{r}"] = -(2.0 + r)
REGION_Y["QUEUE"] = -(2.0 + N_YARD_ROWS + 1)

# ── Move type colours ────────────────────────────────────────────────────

MOVE_COLOURS: Dict[str, str] = {
    "TRAIN_TO_YARD": "#3498db",           # blue (import)
    "YARD_TO_TRUCK": "#2ecc71",           # green (pickup delivery)
    "YARD_TO_TRAIN": "#e67e22",           # orange (export)
    "TRUCK_TO_YARD": "#f39c12",           # gold (delivery unload)
    "YARD_TO_YARD": "#95a5a6",            # grey (restack)
    "YARD_TO_TERMINAL_TRUCK": "#9b59b6",  # purple (terminal truck)
    "PARK_TRUCK": "#1abc9c",              # teal (parking)
    "TRAIN_TO_TRUCK": "#e74c3c",          # red (direct transfer)
    "TRUCK_TO_TRAIN": "#c0392b",          # dark red
}


@dataclass
class MoveRecord:
    """One parsed row from the moves CSV."""
    timestamp: datetime
    move_type: str
    container_id: str
    src_row: int
    src_bay: int
    dst_row: int
    dst_bay: int
    crane_time_s: float


def _row_to_y(row: int, region: str) -> float:
    """Convert unified row index to plot Y coordinate."""
    if region == "RAIL":
        return N_TRACKS - row
    elif region == "PARKING":
        return -1.0
    elif region == "YARD":
        return -(2.0 + (row - 8))
    elif region == "QUEUE":
        return -(2.0 + N_YARD_ROWS + 1)
    return 0.0


def _parse_moves(path: str) -> List[MoveRecord]:
    """Read moves CSV into list of MoveRecord."""
    records = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = datetime.fromisoformat(row["timestamp"])
            except (ValueError, KeyError):
                continue
            records.append(MoveRecord(
                timestamp=ts,
                move_type=row.get("move_type", ""),
                container_id=row.get("container_id", ""),
                src_row=int(row.get("src_row", -1)),
                src_bay=int(row.get("src_bay", -1)),
                dst_row=int(row.get("dst_row", -1)),
                dst_bay=int(row.get("dst_bay", -1)),
                crane_time_s=float(row.get("crane_time_s", 0)),
            ))
    return records


def _build_frames(
    moves: List[MoveRecord],
    frame_minutes: float = 1.0,
) -> List[List[MoveRecord]]:
    """Group moves into time frames."""
    if not moves:
        return []

    t0 = moves[0].timestamp
    t_end = moves[-1].timestamp
    total_minutes = (t_end - t0).total_seconds() / 60.0
    n_frames = max(1, int(total_minutes / frame_minutes) + 1)

    frames: List[List[MoveRecord]] = [[] for _ in range(n_frames)]
    for m in moves:
        dt = (m.timestamp - t0).total_seconds() / 60.0
        idx = min(int(dt / frame_minutes), n_frames - 1)
        frames[idx].append(m)

    return frames


def create_animation(
    moves: List[MoveRecord],
    frame_minutes: float = 1.0,
    output_path: Optional[str] = None,
    fps: int = 10,
    figsize: Tuple[float, float] = (16, 8),
) -> None:
    """Create and display/save the crane movement animation."""
    frames = _build_frames(moves, frame_minutes)
    if not frames:
        print("No moves to visualise.")
        return

    fig, ax = plt.subplots(figsize=figsize)

    # Draw terminal layout background
    _draw_terminal_background(ax)

    # Title and time display
    title_text = ax.set_title("Crane Movements", fontsize=14)
    time_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes,
        fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Collect active arrows for animation
    active_arrows: List = []

    def update(frame_idx):
        nonlocal active_arrows
        # Clear old arrows
        for arrow in active_arrows:
            arrow.remove()
        active_arrows.clear()

        # Draw moves in this frame and recent frames (fade window)
        fade_window = 5  # show moves from last N frames
        start = max(0, frame_idx - fade_window)
        for fi in range(start, frame_idx + 1):
            if fi >= len(frames):
                break
            alpha = 0.3 + 0.7 * (1.0 - (frame_idx - fi) / (fade_window + 1))
            for m in frames[fi]:
                if m.src_bay < 0 or m.dst_bay < 0:
                    continue
                src_x = m.src_bay
                dst_x = m.dst_bay
                src_y = _row_to_y(m.src_row, _region_of_row(m.src_row))
                dst_y = _row_to_y(m.dst_row, _region_of_row(m.dst_row))

                colour = MOVE_COLOURS.get(m.move_type, "#333333")
                arrow = ax.annotate(
                    "",
                    xy=(dst_x, dst_y),
                    xytext=(src_x, src_y),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=colour,
                        alpha=alpha,
                        lw=1.5,
                    ),
                )
                active_arrows.append(arrow)

        # Update time display
        if moves and frame_idx < len(frames):
            t = moves[0].timestamp
            from datetime import timedelta
            t += timedelta(minutes=frame_idx * frame_minutes)
            time_text.set_text(f"Time: {t.strftime('%H:%M')}")
            title_text.set_text(
                f"Crane Movements — Frame {frame_idx + 1}/{len(frames)}"
            )

        return active_arrows + [time_text, title_text]

    anim = FuncAnimation(
        fig, update,
        frames=len(frames),
        interval=1000 // fps,
        blit=False,
    )

    if output_path:
        print(f"Saving animation to {output_path} ({len(frames)} frames)...")
        anim.save(output_path, writer="pillow", fps=fps, dpi=100)
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close(fig)


def _region_of_row(row: int) -> str:
    """Determine region name from unified row index."""
    if 0 <= row < N_TRACKS:
        return "RAIL"
    elif row == N_TRACKS:
        return "PARKING"
    elif N_TRACKS + 1 <= row < N_TRACKS + 1 + N_YARD_ROWS:
        return "YARD"
    else:
        return "QUEUE"


def _draw_terminal_background(ax) -> None:
    """Draw the terminal layout as background rectangles."""
    # Rail tracks
    for t in range(N_TRACKS):
        y = N_TRACKS - t - 0.4
        rect = mpatches.Rectangle(
            (0, y), N_BAYS, 0.8,
            linewidth=0.5, edgecolor="#bdc3c7",
            facecolor="#ecf0f1", alpha=0.5,
        )
        ax.add_patch(rect)
        ax.text(-2, y + 0.4, f"Track {t}", fontsize=7, va="center")

    # Parking lane
    rect = mpatches.Rectangle(
        (0, -1.4), N_BAYS, 0.8,
        linewidth=0.5, edgecolor="#bdc3c7",
        facecolor="#fadbd8", alpha=0.5,
    )
    ax.add_patch(rect)
    ax.text(-2, -1.0, "Parking", fontsize=7, va="center")

    # Yard rows
    for r in range(N_YARD_ROWS):
        y = -(2.0 + r) - 0.4
        rect = mpatches.Rectangle(
            (0, y), N_BAYS, 0.8,
            linewidth=0.5, edgecolor="#bdc3c7",
            facecolor="#d5f5e3", alpha=0.5,
        )
        ax.add_patch(rect)
        ax.text(-2, y + 0.4, f"Yard {r}", fontsize=7, va="center")

    ax.set_xlim(-4, N_BAYS + 2)
    ax.set_ylim(-(2.0 + N_YARD_ROWS + 2), N_TRACKS + 1)
    ax.set_xlabel("Bay", fontsize=10)
    ax.set_ylabel("Region", fontsize=10)
    ax.set_aspect("auto")

    # Legend
    legend_patches = [
        mpatches.Patch(color=c, label=mt.replace("_", " ").title())
        for mt, c in MOVE_COLOURS.items()
    ]
    ax.legend(
        handles=legend_patches, loc="lower right",
        fontsize=7, ncol=3,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualise crane movements from CSV logs",
    )
    parser.add_argument(
        "--moves", required=True,
        help="Path to moves CSV file",
    )
    parser.add_argument(
        "--events", default=None,
        help="Path to vehicle events CSV file (optional, for annotations)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output path for GIF (omit for interactive window)",
    )
    parser.add_argument(
        "--frame-minutes", type=float, default=1.0,
        help="Simulated minutes per animation frame (default: 1)",
    )
    parser.add_argument(
        "--fps", type=int, default=10,
        help="Frames per second for GIF output (default: 10)",
    )
    args = parser.parse_args()

    moves = _parse_moves(args.moves)
    print(f"Loaded {len(moves)} moves from {args.moves}")

    if not moves:
        print("No moves to visualise.")
        return

    create_animation(
        moves,
        frame_minutes=args.frame_minutes,
        output_path=args.output,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
