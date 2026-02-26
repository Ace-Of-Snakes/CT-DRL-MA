# same_length_stack_demo.py
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
    fig, ax = plt.subplots(figsize=(12, 4 + yard.n_tiers))
    block_h, gap = 1.0, 0.3
    rows_per_tier = yard.n_rows
    total_height = yard.n_tiers * (rows_per_tier * block_h + gap) - gap

    # Bay guides
    for b in range(yard.n_bays + 1):
        ax.axvline(b * yard.split_factor, color="#e5e5e5", lw=0.8, zorder=0)

    # Draw containers
    for rec in yard.iter_records():
        cid = rec.container.container_id
        row = rec.placement.row
        tier = rec.placement.tier
        abs_start = rec.placement.bay * yard.split_factor + rec.placement.start_split
        tier_top = total_height - tier * (rows_per_tier * block_h + gap)
        y = tier_top - (row + 1) * block_h
        color = "#4c78a8" if rec.is_accessible else "#f58518"
        ax.add_patch(plt.Rectangle((abs_start, y), rec.n_splits, block_h,
                                   facecolor=color, alpha=0.75, edgecolor="k", linewidth=1.0))
        if show_ids:
            ax.text(abs_start + 0.1, y + 0.05, cid, fontsize=8, color="white")

    ax.set_xlim(0, yard.total_splits)
    ax.set_ylim(-0.5, total_height + 0.5)
    ax.set_xlabel(f"Split index (split_factor={yard.split_factor})")
    ax.set_ylabel("Rows per tier (top tier at top)")
    ax.set_title(title or "Yard")
    ax.set_xticks([b * yard.split_factor for b in range(yard.n_bays + 1)])
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()


def draw_candidate_placement(yard: StorageYard, placement, length_ft: int, title: str = ""):
    fig, ax = plt.subplots(figsize=(12, 4 + yard.n_tiers))
    block_h, gap = 1.0, 0.3
    rows_per_tier = yard.n_rows
    total_height = yard.n_tiers * (rows_per_tier * block_h + gap) - gap

    # Guides
    for b in range(yard.n_bays + 1):
        ax.axvline(b * yard.split_factor, color="#eeeeee", lw=0.8, zorder=0)

    # Existing containers (light gray)
    for rec in yard.iter_records():
        row = rec.placement.row
        tier = rec.placement.tier
        abs_start = rec.placement.bay * yard.split_factor + rec.placement.start_split
        tier_top = total_height - tier * (rows_per_tier * block_h + gap)
        y = tier_top - (row + 1) * block_h
        ax.add_patch(plt.Rectangle((abs_start, y), rec.n_splits, block_h,
                                   facecolor="#bbbbbb", alpha=0.4, edgecolor="k", linewidth=0.5))

    n_splits = CONTAINER_LENGTH_TO_SUB_BAYS[length_ft]
    # Candidate rectangle (green dashed)
    if placement is not None:
        p = placement
        abs_start = p.bay * yard.split_factor + p.start_split
        tier_top = total_height - p.tier * (rows_per_tier * block_h + gap)
        y = tier_top - (p.row + 1) * block_h
        ax.add_patch(plt.Rectangle((abs_start, y), n_splits, block_h,
                                   facecolor="none", edgecolor="#2ca02c", linestyle="--", linewidth=1.6))
        ax.text(abs_start + 0.1, y + 0.05, f"r{p.row}/t{p.tier}", fontsize=7, color="#2ca02c")

    ax.set_xlim(0, yard.total_splits)
    ax.set_ylim(-0.5, total_height + 0.5)
    ax.set_xlabel("Split index")
    ax.set_ylabel("Rows per tier")
    ax.set_title(title or "Candidate placement (green dashed)")
    ax.set_xticks([b * yard.split_factor for b in range(yard.n_bays + 1)])
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()


def main_same_length_demo():
    yard = StorageYard(n_rows=3, n_bays=12, n_tiers=4, coordinates=[], validate=False)

    # Prefer 23ft if supported; otherwise fall back to the first available length.
    preferred = 23
    if preferred in CONTAINER_LENGTH_TO_SUB_BAYS:
        L = preferred
    else:
        L = sorted(CONTAINER_LENGTH_TO_SUB_BAYS.keys())[0]
        print(f"Note: {preferred}ft not configured. Using length {L}ft instead.")

    print(f"Using container length: {L}ft (n_splits={CONTAINER_LENGTH_TO_SUB_BAYS[L]})")
    draw_yard(yard, title="Empty yard")

    # 1) Place first container of length L (prefer ground tier via find_single_placement)
    c1 = make_container(f"S{L}_01", L)
    chosen1 = yard.find_single_placement(c1, target_bay=3, max_proximity=2, prefer_lower_tier=True)
    if not chosen1:
        print("No placement found for the first container.")
        return

    yard.add_container(c1, chosen1)
    print(f"Placed {c1.container_id} at row={chosen1.row}, tier={chosen1.tier}, bay={chosen1.bay}, start_split={chosen1.start_split}")
    draw_yard(yard, title=f"After placing {c1.container_id}")

    # 2) Stack second container directly above the first (same row, bay, start_split, tier+1)
    c2 = make_container(f"S{L}_02", L)
    aligned2 = PlacementResult(
        row=chosen1.row,
        bay=chosen1.bay,
        tier=chosen1.tier + 1,
        start_split=chosen1.start_split
    )

    if aligned2.tier < yard.n_tiers:
        yard.add_container(c2, aligned2)
        print(f"Placed {c2.container_id} above {c1.container_id} at row={aligned2.row}, tier={aligned2.tier}, bay={aligned2.bay}, start_split={aligned2.start_split}")
        draw_yard(yard, title=f"After stacking {c2.container_id} above {c1.container_id}")
    else:
        print("Cannot stack second container -- max tier reached.")
        aligned2 = None

    # 3) Try a third one to stack further
    c3 = make_container(f"S{L}_03", L)
    aligned3 = None
    if aligned2 and aligned2.tier + 1 < yard.n_tiers:
        aligned3 = PlacementResult(
            row=aligned2.row,
            bay=aligned2.bay,
            tier=aligned2.tier + 1,
            start_split=aligned2.start_split
        )
        yard.add_container(c3, aligned3)
        print(f"Placed {c3.container_id} above {c2.container_id} at row={aligned3.row}, tier={aligned3.tier}")
        draw_yard(yard, title=f"After stacking {c3.container_id} on top")
    else:
        print("No further aligned above-tier placement found (maybe reached max tier or blocked).")

    # 4) Show accessible containers and then remove the top one
    accessible_ids = sorted(rec.container.container_id for rec in yard.iter_accessible())
    print("Accessible containers:", accessible_ids)

    # Remove topmost if exists (prefer c3, else c2)
    to_remove = None
    for c in [c3, c2, c1]:
        rec = yard.get_record(c.container_id)
        if rec is not None:
            to_remove = c
            break

    if to_remove:
        yard.remove_container(to_remove)
        print(f"Removed top container: {to_remove.container_id}")
        draw_yard(yard, title=f"After removing {to_remove.container_id}")


if __name__ == "__main__":
    main_same_length_demo()
