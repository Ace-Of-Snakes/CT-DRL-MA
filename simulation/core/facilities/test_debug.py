#!/usr/bin/env python3
"""Debug occupancy - check constants and trace."""
import numpy as np
from dataclasses import dataclass


from simulation.core.facilities.yard import OptimizedStorageYard, PlacementResult
from simulation.core.facilities.constants import BAY_SPLIT_FACTOR, CONTAINER_LENGTH_TO_SUB_BAYS

@dataclass
class MockContainer:
    container_id: str
    length_ft: float = 40.0
    goods_type: str = "Regular"
    is_swap_body: bool = False
    is_trailer: bool = False

print("=== CONSTANTS ===")
print(f"BAY_SPLIT_FACTOR: {BAY_SPLIT_FACTOR}")
print(f"CONTAINER_LENGTH_TO_SUB_BAYS: {CONTAINER_LENGTH_TO_SUB_BAYS}")

yard = OptimizedStorageYard(n_rows=2, n_bays=5, n_tiers=3, coordinates=[])
print(f"\nYard: {yard.n_rows} rows, {yard.n_bays} bays, {yard.n_tiers} tiers")
print(f"Total splits: {yard.total_splits}")
print(f"Occupancy shape: {yard.occupancy_mask.shape}")

# Place A
cA = MockContainer("A", length_ft=40.0)
n_splits_A = yard.container_length_map.get(cA.length_ft)
print(f"\n=== Placing A (40ft -> {n_splits_A} splits) at bay=1, start_split=0 ===")
placement_A = PlacementResult(row=0, bay=1, tier=0, start_split=0)
print(f"Placement abs_start: {placement_A.abs_start}")
yard.add_container(cA, placement_A)
print(f"Tier 0, Row 0: {yard.occupancy_mask[0, 0, :40].astype(int)}")

# Stack B
cB = MockContainer("B", length_ft=40.0)
print(f"\n=== Stacking B at bay=1, tier=1 ===")
placement_B = PlacementResult(row=0, bay=1, tier=1, start_split=0)
print(f"Placement abs_start: {placement_B.abs_start}")
yard.add_container(cB, placement_B)
print(f"Tier 1, Row 0: {yard.occupancy_mask[1, 0, :40].astype(int)}")

# Move B
print(f"\n=== Moving B to bay=3, tier=0 ===")
dest = PlacementResult(row=0, bay=3, tier=0, start_split=0)
print(f"Destination abs_start: {dest.abs_start}")
yard.move_container("B", dest)
print(f"Tier 0, Row 0: {yard.occupancy_mask[0, 0, :50].astype(int)}")
print(f"Tier 1, Row 0: {yard.occupancy_mask[1, 0, :50].astype(int)}")

# Check B's new position
rec_B = yard.get_record("B")
print(f"\nB's new placement: bay={rec_B.placement.bay}, abs_start={rec_B.placement.abs_start}, tier={rec_B.placement.tier}")