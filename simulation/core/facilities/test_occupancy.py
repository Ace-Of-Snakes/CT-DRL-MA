#!/usr/bin/env python3
"""Quick occupancy matrix test with stacking and moves."""
import numpy as np
from simulation.core.facilities.yard import OptimizedStorageYard, PlacementResult
from dataclasses import dataclass

@dataclass
class MockContainer:
    container_id: str
    length_ft: float = 40.0
    goods_type: str = "Regular"
    is_swap_body: bool = False
    is_trailer: bool = False

def show(yard, label):
    """Print occupancy for tiers 0-2, rows 0-1, all splits."""
    print(f"\n=== {label} ===")
    for t in range(min(3, yard.n_tiers)):
        print(f"T{t}: ", end="")
        for r in range(min(2, yard.n_rows)):
            row_str = ''.join(['█' if yard.occupancy_mask[t, r, s] else '·' for s in range(yard.total_splits)])
            print(f"R{r}[{row_str}]", end=" ")
        print()

yard = OptimizedStorageYard(n_rows=3, n_bays=5, n_tiers=4, coordinates=[])

show(yard, "Empty yard")

# Place container A at tier 0
cA = MockContainer("A")
yard.add_container(cA, PlacementResult(row=0, bay=1, tier=0, start_split=0))
show(yard, "After placing A (tier 0, bay 1)")

# Stack container B on top of A
cB = MockContainer("B")
yard.add_container(cB, PlacementResult(row=0, bay=1, tier=1, start_split=0))
show(yard, "After stacking B on A (tier 1)")

# Place container C elsewhere
cC = MockContainer("C")
yard.add_container(cC, PlacementResult(row=1, bay=2, tier=0, start_split=0))
show(yard, "After placing C (row 1, tier 0, bay 2)")

# Check accessibility
print(f"\nAccessibility: A={yard._accessible_mask[yard._id_to_idx['A']]}, "
      f"B={yard._accessible_mask[yard._id_to_idx['B']]}, "
      f"C={yard._accessible_mask[yard._id_to_idx['C']]}")

# Move B from bay 1 to bay 3
yard.move_container("B", PlacementResult(row=0, bay=3, tier=0, start_split=0))
show(yard, "After moving B to bay 3 (tier 0)")

# Check accessibility again
print(f"\nAccessibility: A={yard._accessible_mask[yard._id_to_idx['A']]}, "
      f"B={yard._accessible_mask[yard._id_to_idx['B']]}, "
      f"C={yard._accessible_mask[yard._id_to_idx['C']]}")