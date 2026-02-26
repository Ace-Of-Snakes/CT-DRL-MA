# demo_optimized_storage_usage.py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from simulation.core.facilities.yard import StorageYard, PlacementResult
from simulation.core.containers.container import Container
from simulation.core.facilities.constants import CONTAINER_LENGTH_TO_SUB_BAYS
from simulation.core.enums import Direction, GoodsType


def make_container(container_id: str, length_ft: int, goods_type: GoodsType = GoodsType.REGULAR) -> Container:
    return Container(
        container_id=container_id,
        direction=Direction.IMPORT,
        container_type=f"{length_ft}ft",
        arrival_date=datetime.now(),
        departure_date=datetime.now(),
        goods_type=goods_type,
        length_ft=length_ft,
        length_m=length_ft * 0.3048,
        width_m=2.44,
        height_m=2.59,
        is_swap_body=False,
        is_trailer=False,
    )


def draw_yard(yard: StorageYard, title: str = "", show_ids: bool = True):
    """
    Draw containers as rectangles in split space.
    Y-axis stacks tiers (top tier at top) and rows within each tier.
    """
    fig, ax = plt.subplots(figsize=(12, 4 + yard.n_tiers))
    # Vertical layout: groups of rows per tier, top tier at top
    block_h = 1.0
    gap = 0.3
    rows_per_tier = yard.n_rows
    total_height = yard.n_tiers * (rows_per_tier * block_h + gap) - gap

    # Guide lines: bay boundaries
    for b in range(yard.n_bays + 1):
        x = b * yard.split_factor
        ax.axvline(x, color="#dddddd", lw=0.8, zorder=0)

    # Draw each container from the records
    for rec in yard.iter_records():
        cid = rec.container.container_id
        row = rec.placement.row
        tier = rec.placement.tier
        abs_start = rec.placement.bay * yard.split_factor + rec.placement.start_split
        # y coordinate: top tier at top
        tier_top = total_height - tier * (rows_per_tier * block_h + gap)
        y = tier_top - (row + 1) * block_h
        color = "#4c78a8" if rec.is_accessible else "#f58518"
        ax.add_patch(plt.Rectangle(
            (abs_start, y), rec.n_splits, block_h,
            facecolor=color, alpha=0.7, edgecolor="k", linewidth=1.0
        ))
        if show_ids:
            ax.text(abs_start + 0.1, y + 0.05, cid, fontsize=8, color="white")

    ax.set_xlim(0, yard.total_splits)
    ax.set_ylim(-0.5, total_height + 0.5)
    ax.set_xlabel(f"Split index (split_factor={yard.split_factor}; total_splits={yard.total_splits})")
    ax.set_ylabel("Rows per tier (top tier at top)")
    ax.set_title(title if title else "Yard state")
    # Bay tick labels every bay
    xticks = [b * yard.split_factor for b in range(yard.n_bays + 1)]
    ax.set_xticks(xticks)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.show()


def draw_candidate_placement(yard: StorageYard, placement, length_ft: int, title: str = ""):
    """
    Draw a single candidate placement as a dashed rectangle (does not alter yard).
    """
    fig, ax = plt.subplots(figsize=(12, 4 + yard.n_tiers))
    block_h = 1.0
    gap = 0.3
    rows_per_tier = yard.n_rows
    total_height = yard.n_tiers * (rows_per_tier * block_h + gap) - gap

    # Guide lines
    for b in range(yard.n_bays + 1):
        x = b * yard.split_factor
        ax.axvline(x, color="#eeeeee", lw=0.8, zorder=0)

    # Existing containers in light fill
    for rec in yard.iter_records():
        row = rec.placement.row
        tier = rec.placement.tier
        abs_start = rec.placement.bay * yard.split_factor + rec.placement.start_split
        tier_top = total_height - tier * (rows_per_tier * block_h + gap)
        y = tier_top - (row + 1) * block_h
        ax.add_patch(plt.Rectangle(
            (abs_start, y), rec.n_splits, block_h,
            facecolor="#bbbbbb", alpha=0.4, edgecolor="k", linewidth=0.5
        ))

    n_splits = CONTAINER_LENGTH_TO_SUB_BAYS[length_ft]
    # Candidate rectangle
    if placement is not None:
        p = placement
        abs_start = p.bay * yard.split_factor + p.start_split
        tier_top = total_height - p.tier * (rows_per_tier * block_h + gap)
        y = tier_top - (p.row + 1) * block_h
        ax.add_patch(plt.Rectangle(
            (abs_start, y), n_splits, block_h,
            facecolor="none", edgecolor="#2ca02c", linestyle="--", linewidth=1.5
        ))
        ax.text(abs_start + 0.1, y + 0.05, f"({p.row},T{p.tier})", fontsize=7, color="#2ca02c")

    ax.set_xlim(0, yard.total_splits)
    ax.set_ylim(-0.5, total_height + 0.5)
    ax.set_xlabel("Split index")
    ax.set_ylabel("Rows per tier")
    ax.set_title(title if title else "Candidate placement (green dashed)")
    xticks = [b * yard.split_factor for b in range(yard.n_bays + 1)]
    ax.set_xticks(xticks)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.show()


def main():
    # Small yard; leave coordinates empty or add a few special slots if you want
    yard = StorageYard(
        n_rows=3,
        n_bays=12,
        n_tiers=3,
        coordinates=[],
        validate=False
    )

    print(f"Yard: rows={yard.n_rows}, bays={yard.n_bays}, tiers={yard.n_tiers}, split_factor={yard.split_factor}")

    # Use all known lengths from constants (e.g., 20, 40)
    available_lengths = sorted(CONTAINER_LENGTH_TO_SUB_BAYS.keys())
    print(f"Available container lengths: {available_lengths}")

    # Visualize empty yard
    draw_yard(yard, title="Initial empty yard")

    # Try placing one container for each length at a nearby target bay with low proximity
    target_bay = 3      # 0-based
    proximity = 1       # low proximity
    placed = []

    for idx, L in enumerate(available_lengths, start=1):
        c = make_container(f"C{L}_{idx:02d}", L, goods_type=GoodsType.REGULAR)
        placement = yard.find_single_placement(c, target_bay=target_bay, max_proximity=proximity)
        found = "found" if placement is not None else "not found"
        print(f"{c.container_id}: placement {found} near bay {target_bay} (+/-{proximity})")

        # Visualize candidate for this container
        draw_candidate_placement(yard, placement, L, title=f"Candidate for {c.container_id} (len={L}ft)")

        if placement is not None:
            print(f"Placing {c.container_id} at row={placement.row}, tier={placement.tier}, bay={placement.bay}, start_split={placement.start_split}")
            yard.add_container(c, placement)
            placed.append((c, placement))
            # Visualize yard after placement
            draw_yard(yard, title=f"Yard after placing {c.container_id}")
        else:
            print(f"No feasible placement for {c.container_id}")

    # Show accessible containers after all placements
    accessible_ids = sorted(rec.container.container_id for rec in yard.iter_accessible())
    print("Accessible containers (top-of-stack):", accessible_ids)

    # Demonstrate removal: remove the last placed container (if any)
    if placed:
        c_last, pos_last = placed[-1]
        print(f"Removing last placed container: {c_last.container_id}")
        yard.remove_container(c_last)
        draw_yard(yard, title=f"Yard after removing {c_last.container_id}")

    # Show container count
    print(f"Container count: {yard.container_count}")


if __name__ == "__main__":
    np.random.seed(7)  # reproducible target choices if you randomize
    main()
