# simulation/core/facilities/parking.py
from typing import Optional, Set, List
from simulation.core.vehicles.truck import Truck

class ParkingArea:
    """
    O(1) allocator with bay-aware spots to enforce ±2 bay parking constraint.
    Spot IDs must encode bay and split as f"P_{bay}_{split}" (0-based).
    """
    def __init__(self, spots: Set[str]):
        self.free: Set[str] = set(spots)
        self.used: Set[str] = set()

    @staticmethod
    def make_grid(n_bays: int, split_factor: int, prefix: str = "P") -> Set[str]:
        return {f"{prefix}_{bay}_{split}" for bay in range(n_bays) for split in range(split_factor)}

    @staticmethod
    def spot_bay(spot: str) -> Optional[int]:
        # expect "P_{bay}_{split}"
        try:
            _, bay, _ = spot.split("_")
            return int(bay)
        except Exception:
            return None

    def iter_free(self) -> List[str]:
        return list(self.free)

    def iter_free_in_bay_range(self, bay_lo: int, bay_hi: int) -> List[str]:
        out = []
        for s in self.free:
            b = self.spot_bay(s)
            if b is not None and bay_lo <= b <= bay_hi:
                out.append(s)
        return out

    def allocate(self, truck: Truck, spot: str) -> bool:
        if spot not in self.free:
            return False
        self.free.remove(spot)
        self.used.add(spot)
        truck.parking_spot = spot
        return True

    def release(self, truck: Truck) -> bool:
        spot = truck.parking_spot
        if not spot or spot not in self.used:
            return False
        self.used.remove(spot)
        self.free.add(spot)
        truck.parking_spot = None
        return True