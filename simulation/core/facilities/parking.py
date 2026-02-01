# simulation/core/facilities/parking_optimized.py
"""
Optimized parking area with array-based storage.
Key changes:
- Bool array instead of string sets
- O(1) spot operations
- O(range) for bay range queries
- No string parsing in hot paths
"""
import numpy as np
from typing import Optional, List, Tuple, Iterator
from dataclasses import dataclass
from simulation.core.vehicles.truck import Truck


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


class OptimizedParkingArea:
    """
    High-performance parking area using numpy arrays.
    
    Optimizations:
    - Bool array for occupancy: O(1) check/set
    - Slicing for bay range queries: O(range_size)
    - No string parsing in hot paths
    - Truck tracking via separate array
    """
    
    __slots__ = (
        'n_bays', 'split_factor', 'prefix',
        'occupied', 'truck_ids', '_truck_spots'
    )
    
    def __init__(
        self,
        n_bays: int,
        split_factor: int,
        prefix: str = "P"
    ):
        """
        Initialize parking area.
        
        Args:
            n_bays: Number of bays
            split_factor: Splits per bay
            prefix: String prefix for spot names
        """
        self.n_bays = n_bays
        self.split_factor = split_factor
        self.prefix = prefix
        
        # Primary storage: (bays, splits) bool array
        self.occupied = np.zeros((n_bays, split_factor), dtype=bool)
        
        # Track which truck is in which spot: -1 for empty
        # Using object array for truck_id strings (or could use int indices)
        self.truck_ids: np.ndarray = np.empty((n_bays, split_factor), dtype=object)
        self.truck_ids.fill(None)
        
        # Reverse lookup: truck_id -> (bay, split)
        self._truck_spots: dict[str, Tuple[int, int]] = {}
    
    # ========== Core Operations ==========
    
    def is_free(self, bay: int, split: int) -> bool:
        """Check if spot is free - O(1)."""
        if not (0 <= bay < self.n_bays and 0 <= split < self.split_factor):
            return False
        return not self.occupied[bay, split]
    
    def allocate(self, truck: Truck, bay: int, split: int) -> bool:
        """
        Allocate spot to truck - O(1).
        Returns True if successful.
        """
        if not self.is_free(bay, split):
            return False
        
        self.occupied[bay, split] = True
        self.truck_ids[bay, split] = truck.truck_id
        self._truck_spots[truck.truck_id] = (bay, split)
        
        # Update truck's parking spot (string for compatibility)
        truck.parking_spot = f"{self.prefix}_{bay}_{split}"
        
        return True

    def release(self, truck: Truck) -> bool:
        """
        Release truck's spot - O(1).
        """
        pos = self._truck_spots.get(truck.truck_id)
        if pos is None:
            return False
        
        bay, split = pos
        self.occupied[bay, split] = False
        self.truck_ids[bay, split] = None
        del self._truck_spots[truck.truck_id]
        truck.parking_spot = None
        
        return True

    # ========== Queries ==========
    
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
        bay_hi: int
    ) -> Iterator[ParkingSpot]:
        """
        Iterate free spots in bay range - O(range_size).
        Much faster than legacy string-based iteration.
        """
        bay_lo = max(0, bay_lo)
        bay_hi = min(self.n_bays - 1, bay_hi)
        
        if bay_lo > bay_hi:
            bay_lo, bay_hi = bay_hi, bay_lo
        
        # Slice and find free positions
        slice_occ = self.occupied[bay_lo:bay_hi + 1, :]
        rows, cols = np.where(~slice_occ)
        
        for rel_bay, split in zip(rows, cols):
            yield ParkingSpot(bay=int(bay_lo + rel_bay), split=int(split))
    
    # ========== State Export (for RL) ==========
    
    def get_occupancy_array(self) -> np.ndarray:
        """Get occupancy as (n_bays, split_factor) bool array."""
        return self.occupied.copy()
