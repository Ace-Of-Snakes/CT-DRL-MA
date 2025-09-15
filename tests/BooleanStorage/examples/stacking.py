# same_length_stack_demo.py
import matplotlib.pyplot as plt
from datetime import datetime

from simulation.terminal_components.storage.BooleanStorage import BooleanStorageYard, PlacementResult
from simulation.terminal_components.storage_units.Container import Container
from simulation.terminal_components.storage.constants import CONTAINER_LENGTH_TO_SUB_BAYS

def make_container(container_id: str, length_ft: int, goods_type: str = "Regular") -> Container:
    return Container(
        container_id=container_id,
        direction="Import",
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

def draw_yard(yard: BooleanStorageYard, title: str = "", show_ids: bool = True):
    fig, ax = plt.subplots(figsize=(12, 4 + yard.n_tiers))
    block_h, gap = 1.0, 0.3
    rows_per_tier = yard.n_rows
    total_height = yard.n_tiers * (rows_per_tier * block_h + gap) - gap

    # Bay guides
    for b in range(yard.n_bays + 1):
        ax.axvline(b * yard.split_factor, color="#e5e5e5", lw=0.8, zorder=0)

    # Draw containers
    for cid, rec in yard.containers.items():
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

def draw_candidate_placements(yard: BooleanStorageYard, placements, length_ft: int, title: str = ""):
    fig, ax = plt.subplots(figsize=(12, 4 + yard.n_tiers))
    block_h, gap = 1.0, 0.3
    rows_per_tier = yard.n_rows
    total_height = yard.n_tiers * (rows_per_tier * block_h + gap) - gap

    # Guides
    for b in range(yard.n_bays + 1):
        ax.axvline(b * yard.split_factor, color="#eeeeee", lw=0.8, zorder=0)

    # Existing containers (light gray)
    for cid, rec in yard.containers.items():
        row = rec.placement.row
        tier = rec.placement.tier
        abs_start = rec.placement.bay * yard.split_factor + rec.placement.start_split
        tier_top = total_height - tier * (rows_per_tier * block_h + gap)
        y = tier_top - (row + 1) * block_h
        ax.add_patch(plt.Rectangle((abs_start, y), rec.n_splits, block_h,
                                   facecolor="#bbbbbb", alpha=0.4, edgecolor="k", linewidth=0.5))

    n_splits = CONTAINER_LENGTH_TO_SUB_BAYS[length_ft]
    # Candidate rectangles (green dashed)
    for p in placements:
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
    ax.set_title(title or "Candidate placements (green dashed)")
    ax.set_xticks([b * yard.split_factor for b in range(yard.n_bays + 1)])
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()

def main_same_length_demo():
    yard = BooleanStorageYard(n_rows=3, n_bays=12, n_tiers=4, coordinates=[], validate=False)

    # Prefer 23ft if supported; otherwise fall back to the first available length.
    preferred = 23
    if preferred in CONTAINER_LENGTH_TO_SUB_BAYS:
        L = preferred
    else:
        L = sorted(CONTAINER_LENGTH_TO_SUB_BAYS.keys())[0]
        print(f"Note: {preferred}ft not configured. Using length {L}ft instead.")

    print(f"Using container length: {L}ft (n_splits={CONTAINER_LENGTH_TO_SUB_BAYS[L]})")
    draw_yard(yard, title="Empty yard")

    # 1) Place first container of length L (prefer ground tier)
    c1 = make_container(f"S{L}_01", L)
    placements1 = yard.search_placement_all_tiers(c1, target_bay=3, max_proximity=2)
    if not placements1:
        print("No placement found for the first container.")
        return

    # Prefer a ground-tier placement to enable stacking test
    chosen1 = next((p for p in placements1 if p.tier == 0), placements1[0])
    yard.add_container(c1, chosen1)
    print(f"Placed {c1.container_id} at row={chosen1.row}, tier={chosen1.tier}, bay={chosen1.bay}, start_split={chosen1.start_split}")
    draw_yard(yard, title=f"After placing {c1.container_id}")

    # 2) Search placements for a second container of the same length
    c2 = make_container(f"S{L}_02", L)
    placements2 = yard.search_placement_all_tiers(c2, target_bay=chosen1.bay, max_proximity=1)
    above_candidates = [p for p in placements2 if p.tier > 0]
    print(f"Second {L}ft: total placements={len(placements2)}, above-tier placements={len(above_candidates)}")

    draw_candidate_placements(yard, placements2[:30], L,
                              title=f"Candidates for {c2.container_id} (same length)")

    # 3) Place the second container directly above the first (same-length, aligned)
    aligned2 = next(
        (p for p in placements2
         if p.row == chosen1.row
         and p.tier == chosen1.tier + 1
         and p.bay == chosen1.bay
         and p.start_split == chosen1.start_split),
        None
    )
    if aligned2:
        yard.add_container(c2, aligned2)
        print(f"Placed {c2.container_id} above {c1.container_id} at row={aligned2.row}, tier={aligned2.tier}, bay={aligned2.bay}, start_split={aligned2.start_split}")
        draw_yard(yard, title=f"After stacking {c2.container_id} above {c1.container_id}")
    else:
        print("No aligned above-tier placement found for the second container; cannot stack.")

    # 4) Try a third one to stack further
    c3 = make_container(f"S{L}_03", L)
    placements3 = yard.search_placement_all_tiers(c3, target_bay=chosen1.bay, max_proximity=1)
    aligned3 = None
    if aligned2:
        aligned3 = next(
            (p for p in placements3
             if p.row == aligned2.row
             and p.tier == aligned2.tier + 1
             and p.bay == aligned2.bay
             and p.start_split == aligned2.start_split),
            None
        )
    print(f"Third {L}ft: total placements={len(placements3)}, above-tier placements={len([p for p in placements3 if p.tier > 0])}")

    if aligned3:
        yard.add_container(c3, aligned3)
        print(f"Placed {c3.container_id} above {c2.container_id} at row={aligned3.row}, tier={aligned3.tier}")
        draw_yard(yard, title=f"After stacking {c3.container_id} on top")
    else:
        print("No further aligned above-tier placement found (maybe reached max tier or blocked).")

    # 5) Show accessible containers and then remove the top one
    print("Accessible containers:", sorted(list(yard.accessible_containers)))

    # Remove topmost if exists (prefer c3, else c2)
    to_remove = None
    for c in [c3, c2, c1]:
        if c.container_id in yard.containers:
            to_remove = c
            break

    if to_remove:
        rec = yard.containers[to_remove.container_id].placement
        yard.remove_container(to_remove)
        print(f"Removed top container: {to_remove.container_id}")
        draw_yard(yard, title=f"After removing {to_remove.container_id}")

if __name__ == "__main__":
    main_same_length_demo()