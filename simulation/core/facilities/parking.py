# simulation/core/facilities/parking.py
"""
Grid-based parking area with O(1) spatial lookups.

Uses the same pattern as StorageYard and RailYard:
- occupancy_flat: 1D bool array over total_splits
- position_grid:  1D int32 array → record index
- ParkingRecord:  truck + placement info

Bay-level rule: one truck per bay (entire bay marked occupied).
"""
import numpy as np
from typing import Optional, List, Tuple, Iterator, Dict
from dataclasses import dataclass

from simulation.core.vehicles.truck import Truck
from simulation.core.facilities.base import EMPTY_SLOT
from simulation.core.facilities.constants import BAY_SPLIT_FACTOR

MAX_PARKED_TRUCKS: int = 200


@dataclass(slots=True)
class ParkingSpot:
    """Spot identifier as integers."""
    bay: int
    split: int

    def to_string(self, prefix: str = "P") -> str:
        """Convert to string format for compatibility."""
        return f"{prefix}_{self.bay}_{self.split}"

    @staticmethod
    def from_string(spot_str: str) -> Optional['ParkingSpot']:
        """Parse string spot name."""
        try:
            parts = spot_str.split("_")
            if len(parts) < 3:
                return None
            return ParkingSpot(bay=int(parts[-2]), split=int(parts[-1]))
        except (ValueError, IndexError):
            return None


@dataclass(slots=True)
class ParkingRecord:
    """A truck parked in the parking area."""
    truck: Truck
    bay: int
    abs_split: int       # bay * split_factor (canonical position)
    idx: int             # internal index in _records


class ParkingArea:
    """
    High-performance parking area using numpy arrays.

    Grid-based with same pattern as StorageYard / RailYard:
    - O(1) check/set via occupancy arrays
    - O(1) truck lookup via position_grid
    - Records with placement info for iteration
    """

    __slots__ = (
        'n_bays', 'split_factor', 'total_splits', 'prefix',
        'occupied', 'truck_ids', '_truck_spots',
        'occupancy_flat', 'position_grid',
        '_records', '_id_to_idx', '_free_indices',
    )

    def __init__(
        self,
        n_bays: int,
        split_factor: int = BAY_SPLIT_FACTOR,
        prefix: str = "P",
    ):
        self.n_bays = n_bays
        self.split_factor = split_factor
        self.total_splits = n_bays * split_factor
        self.prefix = prefix

        # Bay-level storage (existing)
        self.occupied = np.zeros((n_bays, split_factor), dtype=bool)
        self.truck_ids: np.ndarray = np.empty(
            (n_bays, split_factor), dtype=object,
        )
        self.truck_ids.fill(None)
        self._truck_spots: Dict[str, Tuple[int, int]] = {}

        # Flat spatial grid (new — matches yard/rail pattern)
        self.occupancy_flat = np.zeros(self.total_splits, dtype=bool)
        self.position_grid = np.full(
            self.total_splits, EMPTY_SLOT, dtype=np.int32,
        )

        # Record storage
        self._records: List[Optional[ParkingRecord]] = [
            None
        ] * MAX_PARKED_TRUCKS
        self._id_to_idx: Dict[str, int] = {}
        self._free_indices: List[int] = list(
            range(MAX_PARKED_TRUCKS - 1, -1, -1)
        )

    # ══════════════════════════════════════════════════════════════════
    # Core operations
    # ══════════════════════════════════════════════════════════════════

    def is_free(self, bay: int, split: int = 0) -> bool:
        """Check if bay is free (split kept for API compat but ignored)."""
        if not (0 <= bay < self.n_bays):
            return False
        return not self.occupied[bay].any()

    def allocate(self, truck: Truck, bay: int, split: int) -> bool:
        """Allocate a bay to a truck.

        One truck occupies an entire bay (all splits are marked occupied)
        so that no second truck can park in the same bay.  The *split*
        argument is recorded for backward-compat but the whole bay is
        claimed.  Returns True if successful.
        """
        if not (0 <= bay < self.n_bays):
            return False
        if self.occupied[bay].any():
            return False

        # Bay-level occupancy (existing)
        self.occupied[bay, :] = True
        self.truck_ids[bay, 0] = truck.truck_id
        self._truck_spots[truck.truck_id] = (bay, 0)

        # Update truck's parking spot (always split 0 — canonical)
        truck.parking_spot = ParkingSpot(bay=bay, split=0)

        # Flat grid + record (new)
        abs_split = bay * self.split_factor
        self.occupancy_flat[abs_split] = True

        if self._free_indices:
            idx = self._free_indices.pop()
            rec = ParkingRecord(
                truck=truck, bay=bay, abs_split=abs_split, idx=idx,
            )
            self._records[idx] = rec
            self._id_to_idx[truck.truck_id] = idx
            self.position_grid[abs_split] = idx

        return True

    def release(self, truck: Truck) -> bool:
        """Release a truck's bay (frees all splits)."""
        pos = self._truck_spots.get(truck.truck_id)
        if pos is None:
            return False

        bay, _split = pos

        # Bay-level cleanup (existing)
        self.occupied[bay, :] = False
        self.truck_ids[bay, :] = None
        del self._truck_spots[truck.truck_id]
        truck.parking_spot = None

        # Flat grid + record cleanup (new)
        abs_split = bay * self.split_factor
        self.occupancy_flat[abs_split] = False

        idx = self._id_to_idx.pop(truck.truck_id, None)
        if idx is not None:
            self.position_grid[abs_split] = EMPTY_SLOT
            self._records[idx] = None
            self._free_indices.append(idx)

        return True

    # ══════════════════════════════════════════════════════════════════
    # O(1) spatial lookups (new)
    # ══════════════════════════════════════════════════════════════════

    def get_truck_at(self, split: int) -> Optional[Truck]:
        """O(1) truck lookup at absolute split position."""
        if split < 0 or split >= self.total_splits:
            return None
        idx = self.position_grid[split]
        if idx == EMPTY_SLOT:
            return None
        rec = self._records[idx]
        return rec.truck if rec is not None else None

    def get_record_at(self, split: int) -> Optional[ParkingRecord]:
        """O(1) full record lookup at absolute split position."""
        if split < 0 or split >= self.total_splits:
            return None
        idx = self.position_grid[split]
        if idx == EMPTY_SLOT:
            return None
        return self._records[idx]

    def get_record_by_id(self, truck_id: str) -> Optional[ParkingRecord]:
        """O(1) record lookup by truck_id."""
        idx = self._id_to_idx.get(truck_id)
        if idx is None:
            return None
        return self._records[idx]

    # ══════════════════════════════════════════════════════════════════
    # Iteration
    # ══════════════════════════════════════════════════════════════════

    def iter_records(self) -> Iterator[ParkingRecord]:
        """Iterate over all parked truck records."""
        for tid, idx in self._id_to_idx.items():
            rec = self._records[idx]
            if rec is not None:
                yield rec

    # ══════════════════════════════════════════════════════════════════
    # Queries (existing API preserved)
    # ══════════════════════════════════════════════════════════════════

    def get_truck_spot(self, truck_id: str) -> Optional[Tuple[int, int]]:
        """Get (bay, split) for a truck."""
        return self._truck_spots.get(truck_id)

    def iter_free(self) -> Iterator[ParkingSpot]:
        """Iterate over free spots."""
        rows, cols = np.where(~self.occupied)
        for bay, split in zip(rows, cols):
            yield ParkingSpot(bay=int(bay), split=int(split))

    def iter_free_in_bay_range(
        self,
        bay_lo: int,
        bay_hi: int,
    ) -> Iterator[ParkingSpot]:
        """Iterate free spots in bay range — O(range_size)."""
        bay_lo = max(0, bay_lo)
        bay_hi = min(self.n_bays - 1, bay_hi)

        if bay_lo > bay_hi:
            bay_lo, bay_hi = bay_hi, bay_lo

        slice_occ = self.occupied[bay_lo:bay_hi + 1, :]
        rows, cols = np.where(~slice_occ)

        for rel_bay, split in zip(rows, cols):
            yield ParkingSpot(bay=int(bay_lo + rel_bay), split=int(split))

    # ══════════════════════════════════════════════════════════════════
    # State export (for RL)
    # ══════════════════════════════════════════════════════════════════

    def get_occupancy_array(self) -> np.ndarray:
        """Get occupancy as (n_bays, split_factor) bool array."""
        return self.occupied.copy()

    def get_occupancy_flat(self) -> np.ndarray:
        """Get 1D split-level occupancy (total_splits,) bool."""
        return self.occupancy_flat
