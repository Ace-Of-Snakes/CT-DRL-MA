# simulation/training/scenarios/_visualization.py
"""Visualization bridge: convert TutorialResult move logs to animated terminal views.

Uses the existing ``tools/visualise_crane_movements.py`` engine to render
top-down animated arrows for container moves within a single scenario.

Usage in Jupyter::

    from simulation.training.scenarios._visualization import visualize_scenario
    result = runner.run_scenario(scenario)
    visualize_scenario(result)          # inline HTML animation
    visualize_scenario(result, inline=False, save_path="s1.gif")
"""
from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from matplotlib.animation import FuncAnimation

from simulation.training.scenarios._base import MoveRecord, TutorialResult, TUTORIAL_TIME


def visualize_scenario(
    result: TutorialResult,
    inline: bool = True,
    save_path: Optional[str] = None,
    fps: int = 2,
    figsize: tuple = (16, 8),
):
    """Render a top-down terminal animation for one scenario's moves.

    Converts ``TutorialResult.move_log`` into the format expected by
    ``visualise_crane_movements.create_animation()`` and either displays
    inline in Jupyter or saves to file.

    Args:
        result: The tutorial result containing move_log with spatial data.
        inline: If True (default), return an IPython HTML widget for
            inline display in Jupyter notebooks.
        save_path: If provided, save the animation as a GIF to this path.
        fps: Frames per second for the animation.
        figsize: Figure size tuple for matplotlib.

    Returns:
        IPython HTML widget if ``inline=True``, else the FuncAnimation object.
    """
    # Import the visualiser's types and functions
    from tools.visualise_crane_movements import (
        MoveRecord as VisMoveRecord,
        create_animation,
        _build_frames,
        _draw_terminal_background,
        _region_of_row,
        _row_to_y,
        MOVE_COLOURS,
    )

    # Convert our MoveRecords to the visualiser's format.
    # Spread moves across artificial timestamps (1 minute apart per step)
    # so each move gets its own animation frame.
    vis_records: list[VisMoveRecord] = []
    for m in result.move_log:
        if m.src_row is None or m.dst_row is None:
            continue
        if m.src_bay is None or m.dst_bay is None:
            continue
        # Skip invalid spatial data
        if m.src_row < 0 and m.dst_row < 0:
            continue

        # Each move_num gets a distinct timestamp so they animate sequentially
        fake_ts = TUTORIAL_TIME + timedelta(minutes=m.move_num)
        vis_records.append(VisMoveRecord(
            timestamp=fake_ts,
            move_type=m.move_type,
            container_id=m.container_id,
            src_row=m.src_row,
            src_bay=m.src_bay,
            dst_row=m.dst_row,
            dst_bay=m.dst_bay,
            crane_time_s=m.time_s,
        ))

    if not vis_records:
        print(f"No spatial move data to visualize for S{result.scenario_id}: {result.name}")
        return None

    if save_path:
        create_animation(
            vis_records,
            frame_minutes=1.0,
            output_path=save_path,
            fps=fps,
            figsize=figsize,
        )
        return None

    # For inline display, build the animation manually (create_animation
    # calls plt.show() or saves, but we need the FuncAnimation object).
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.animation import FuncAnimation as FA

    frames = _build_frames(vis_records, frame_minutes=1.0)
    if not frames:
        print("No frames to visualize.")
        return None

    fig, ax = plt.subplots(figsize=figsize)
    _draw_terminal_background(ax)

    title_text = ax.set_title(
        f"S{result.scenario_id}: {result.name} "
        f"({'PASS' if result.passed else 'FAIL'})",
        fontsize=14,
    )
    info_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes,
        fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    active_arrows: list = []

    def update(frame_idx):
        nonlocal active_arrows
        for arrow in active_arrows:
            arrow.remove()
        active_arrows.clear()

        # Show moves with a fade window
        fade_window = 3
        start = max(0, frame_idx - fade_window)
        for fi in range(start, frame_idx + 1):
            if fi >= len(frames):
                break
            alpha = 0.3 + 0.7 * (1.0 - (frame_idx - fi) / (fade_window + 1))
            for mv in frames[fi]:
                if mv.src_bay < 0 or mv.dst_bay < 0:
                    continue
                src_y = _row_to_y(mv.src_row, _region_of_row(mv.src_row))
                dst_y = _row_to_y(mv.dst_row, _region_of_row(mv.dst_row))
                colour = MOVE_COLOURS.get(mv.move_type, "#333333")
                arrow = ax.annotate(
                    "",
                    xy=(mv.dst_bay, dst_y),
                    xytext=(mv.src_bay, src_y),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=colour,
                        alpha=alpha,
                        lw=2.0,
                    ),
                )
                active_arrows.append(arrow)

        # Count moves shown so far
        moves_shown = sum(len(frames[i]) for i in range(min(frame_idx + 1, len(frames))))
        info_text.set_text(
            f"Move {moves_shown}/{len(vis_records)} | "
            f"R={result.total_reward:.2f}"
        )
        return active_arrows + [info_text, title_text]

    anim = FA(
        fig, update,
        frames=len(frames),
        interval=1000 // fps,
        blit=False,
    )

    if inline:
        plt.close(fig)  # Prevent double display in Jupyter
        try:
            from IPython.display import HTML
            return HTML(anim.to_jshtml())
        except ImportError:
            print("IPython not available — returning FuncAnimation object")
            return anim

    return anim
