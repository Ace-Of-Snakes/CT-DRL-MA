# simulation/core/facilities/parking.py
from typing import Optional, Set, List
from simulation.core.vehicles.truck import Truck

class ParkingArea:
    """
    Parking spots are named as: P_{MODULE}_{BAY}_{SPLIT}
    MODULE can contain underscores; BAY and SPLIT are the last two underscore-separated tokens.
    BAY is 0-based (0..n_bays-1), SPLIT is 0-based (0..split_factor-1).
    """
    def __init__(self, spots: Set[str]):
        self.free: Set[str] = set(spots)
        self.used: Set[str] = set()

    @staticmethod
    def make_grid(n_bays: int, split_factor: int, prefix: str = "P") -> Set[str]:
        # prefix may include module, e.g., "P_M1" -> "P_M1_{bay}_{split}"
        return {f"{prefix}_{bay}_{split}" for bay in range(n_bays) for split in range(split_factor)}

    @staticmethod
    def spot_bay(spot: str) -> Optional[int]:
        # robust: take last two underscore tokens as bay/split
        try:
            parts = spot.split("_")
            if len(parts) < 3:
                return None
            return int(parts[-2])
        except Exception:
            return None

    def iter_free(self) -> List[str]:
        return list(self.free)

    def iter_free_in_bay_range(self, bay_lo: int, bay_hi: int) -> List[str]:
        # linear scan over free set; parser must be robust to module prefixes
        out: List[str] = []
        if bay_lo > bay_hi:
            bay_lo, bay_hi = bay_hi, bay_lo
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