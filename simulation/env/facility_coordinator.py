# simulation/env/facility_coordinator.py
"""
Unified coordinate wrapper for all terminal facilities.

Maps unified (row, split, tier) coordinates to the correct facility and
local coordinates. Provides O(1) entity lookups and shared yard-dest
validation — eliminating duplicated coordinate translation in resolvers.
"""
from typing import Optional, Tuple, Dict

from simulation.core.containers.container import Container
from simulation.core.vehicles.truck import Truck
from simulation.core.vehicles.train import Train
from simulation.core.facilities.yard import (
    StorageYard, ContainerRecord, PlacementResult,
)
from simulation.core.facilities.railyard import (
    RailYard, RailContainerRecord,
)
from simulation.core.facilities.parking import (
    ParkingArea, ParkingRecord,
)
from simulation.rl.multihead_dqn.config import Dims


class FacilityCoordinator:
    """Thin wrapper dispatching unified-coordinate lookups to facilities.

    All three facilities (yard, rail, parking) use local coordinates.
    This class translates unified row/split/tier to the right facility
    and returns results in a uniform way.
    """

    __slots__ = ('yard', 'rail', 'parking', 'dims')

    def __init__(
        self,
        yard: StorageYard,
        rail: RailYard,
        parking: ParkingArea,
        dims: Dims,
    ):
        self.yard = yard
        self.rail = rail
        self.parking = parking
        self.dims = dims

    # ══════════════════════════════════════════════════════════════════
    # Container lookups
    # ══════════════════════════════════════════════════════════════════

    def get_yard_record(
        self, row: int, split: int, tier: int,
    ) -> Optional[ContainerRecord]:
        """Yard record at unified coords. Returns None if out of range."""
        d = self.dims
        yard_row = row - d.yard_row_start
        if not (0 <= yard_row < d.n_yard_rows):
            return None
        return self.yard.get_record_at(yard_row, tier, split)

    def get_rail_record(
        self, track_row: int, split: int,
    ) -> Optional[RailContainerRecord]:
        """Rail record at unified coords. track_row IS the track index."""
        if track_row < 0 or track_row >= self.dims.rail_row_end:
            return None
        return self.rail.get_record_at(track_row, split)

    def get_parking_record(
        self, split: int,
    ) -> Optional[ParkingRecord]:
        """Parking record at absolute split."""
        return self.parking.get_record_at(split)

    def get_truck_at(self, split: int) -> Optional[Truck]:
        """O(1) truck lookup at absolute parking split."""
        return self.parking.get_truck_at(split)

    def get_train_on_track(
        self, track_id: int, trains: Dict[str, Train],
    ) -> Optional[Train]:
        """Find the train assigned to a track. O(1) via rail dict."""
        train_ids = self.rail.get_trains_on_track(track_id)
        for tid in train_ids:
            train = trains.get(tid)
            if train is not None:
                return train
        return None

    # ══════════════════════════════════════════════════════════════════
    # Yard destination validation
    # ══════════════════════════════════════════════════════════════════

    def validate_yard_dest(
        self,
        row: int,
        split: int,
        tier: int,
        n_splits: int,
    ) -> Optional[PlacementResult]:
        """Validate unified dest coords → yard PlacementResult.

        Checks:
        - Row is within yard region
        - Span fits within total_splits
        - Destination splits are all free
        - Tier support exists (tier > 0 needs occupied tier below)

        Returns PlacementResult in yard-local coordinates or None.
        """
        d = self.dims
        yard_row = row - d.yard_row_start
        if not (0 <= yard_row < d.n_yard_rows):
            return None
        if n_splits <= 0 or split + n_splits > d.n_splits:
            return None

        occ = self.yard.occupancy_mask
        if occ[tier, yard_row, split:split + n_splits].any():
            return None
        if tier > 0:
            if not occ[tier - 1, yard_row, split:split + n_splits].all():
                return None

        sf = self.yard.split_factor
        return PlacementResult(
            row=yard_row,
            bay=split // sf,
            tier=tier,
            start_split=split % sf,
        )
