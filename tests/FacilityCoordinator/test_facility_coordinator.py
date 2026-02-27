"""Tests for FacilityCoordinator unified coordinate wrapper.

Validates that FacilityCoordinator correctly dispatches unified (row, split, tier)
lookups to the appropriate facility and returns results, plus validates
yard destination coords.
"""
import unittest
import numpy as np
from datetime import datetime

from simulation.core.containers.container import Container
from simulation.core.vehicles.train import Train
from simulation.core.vehicles.truck import Truck
from simulation.core.enums import Direction, GoodsType
from simulation.core.facilities.yard import StorageYard, PlacementResult
from simulation.core.facilities.railyard import RailYard, RailSlot
from simulation.core.facilities.parking import ParkingArea
from simulation.env.facility_coordinator import FacilityCoordinator
from simulation.rl.multihead_dqn.config import Dims
from simulation.core.facilities.constants import BAY_SPLIT_FACTOR


class TestFacilityCoordinator(unittest.TestCase):
    """Tests for FacilityCoordinator."""

    def setUp(self):
        n_rows = 5
        n_bays = 10
        n_tiers = 5
        sf = BAY_SPLIT_FACTOR

        coordinates = [
            (1, 1, "r"), (2, 1, "dg"), (3, 1, "sb_t"),
        ]

        self.yard = StorageYard(
            n_rows=n_rows, n_bays=n_bays, n_tiers=n_tiers,
            coordinates=coordinates,
        )
        self.rail = RailYard(n_tracks=4, n_bays=n_bays, split_factor=sf)
        self.parking = ParkingArea(n_bays=n_bays, split_factor=sf)
        self.dims = Dims(
            n_tracks=4,
            n_parking_rows=1,
            n_yard_rows=n_rows,
            n_queue_rows=2,
            n_bays=n_bays,
            split_factor=sf,
            n_tiers=n_tiers,
        )
        self.coord = FacilityCoordinator(
            self.yard, self.rail, self.parking, self.dims,
        )
        self.now = datetime(2025, 6, 15, 10, 0, 0)

    # ── Helpers ──────────────────────────────────────────────────────

    def _make_container(
        self,
        cid: str = "C001",
        length_ft: int = 40,
        direction: Direction = Direction.IMPORT,
    ) -> Container:
        return Container(
            container_id=cid,
            direction=direction,
            container_type=f"{length_ft}ft",
            arrival_date=self.now,
            departure_date=self.now,
            goods_type=GoodsType.REGULAR,
            length_ft=length_ft,
            length_m=length_ft * 0.3048,
            width_m=2.44,
            height_m=2.59,
        )

    def _make_train_with_containers(
        self,
        train_id: str = "T1",
        n_wagons: int = 2,
    ) -> Train:
        train = Train(train_id=train_id, num_wagons=n_wagons)
        for w in range(n_wagons):
            c = self._make_container(f"{train_id}_C{w:03d}")
            train.add_container(c, wagon_index=w)
        return train

    def _place_in_yard(self, cid: str, row: int = 0, bay: int = 5):
        """Place a container in the yard at a specific position."""
        c = self._make_container(cid)
        placement = self.yard.find_single_placement(c, target_bay=bay, max_proximity=3)
        if placement:
            self.yard.add_container(c, placement)
            return c, placement
        return None, None

    # ── Yard record lookups ──────────────────────────────────────────

    def test_get_yard_record_valid(self):
        """get_yard_record returns record for occupied yard position."""
        c, pl = self._place_in_yard("YC1", bay=5)
        self.assertIsNotNone(c)

        # Unified coords: yard starts at row = dims.yard_row_start
        unified_row = self.dims.yard_row_start + pl.row
        rec = self.coord.get_yard_record(unified_row, pl.abs_start, pl.tier)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.container.container_id, "YC1")

    def test_get_yard_record_empty(self):
        """get_yard_record returns None for empty position."""
        rec = self.coord.get_yard_record(self.dims.yard_row_start, 0, 0)
        self.assertIsNone(rec)

    def test_get_yard_record_rail_row(self):
        """get_yard_record returns None for rail-region row."""
        rec = self.coord.get_yard_record(0, 0, 0)  # row 0 is rail
        self.assertIsNone(rec)

    # ── Rail record lookups ──────────────────────────────────────────

    def test_get_rail_record_valid(self):
        """get_rail_record returns record for occupied rail position."""
        train = self._make_train_with_containers("T1", n_wagons=1)
        self.rail.slot_train(train, RailSlot(track_id=0, anchor_bay=2))

        rec = self.coord.get_rail_record(0, 2 * BAY_SPLIT_FACTOR)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.container.container_id, "T1_C000")

    def test_get_rail_record_empty(self):
        """get_rail_record returns None for empty rail position."""
        rec = self.coord.get_rail_record(0, 0)
        self.assertIsNone(rec)

    # ── Parking record lookups ───────────────────────────────────────

    def test_get_parking_record_valid(self):
        """get_parking_record returns record for parked truck."""
        tk = Truck(truck_id="TK1", arrival_time=self.now)
        self.parking.allocate(tk, bay=3, split=0)

        rec = self.coord.get_parking_record(3 * BAY_SPLIT_FACTOR)
        self.assertIsNotNone(rec)
        self.assertIs(rec.truck, tk)

    def test_get_parking_record_empty(self):
        """get_parking_record returns None for empty parking split."""
        rec = self.coord.get_parking_record(0)
        self.assertIsNone(rec)

    # ── Truck lookup ─────────────────────────────────────────────────

    def test_get_truck_at(self):
        """get_truck_at returns truck at parking split."""
        tk = Truck(truck_id="TK1", arrival_time=self.now)
        self.parking.allocate(tk, bay=5, split=0)

        result = self.coord.get_truck_at(5 * BAY_SPLIT_FACTOR)
        self.assertIs(result, tk)

    def test_get_truck_at_empty(self):
        """get_truck_at returns None for empty parking split."""
        self.assertIsNone(self.coord.get_truck_at(0))

    # ── Train lookup ─────────────────────────────────────────────────

    def test_get_train_on_track(self):
        """get_train_on_track returns train assigned to track."""
        train = self._make_train_with_containers("T1", n_wagons=1)
        self.rail.slot_train(train, RailSlot(track_id=2, anchor_bay=0))

        trains = {"T1": train}
        result = self.coord.get_train_on_track(2, trains)
        self.assertIs(result, train)

    def test_get_train_on_track_empty(self):
        """get_train_on_track returns None for empty track."""
        result = self.coord.get_train_on_track(0, {})
        self.assertIsNone(result)

    # ── validate_yard_dest ───────────────────────────────────────────

    def test_validate_yard_dest_empty_yard(self):
        """Valid dest in empty yard returns PlacementResult."""
        unified_row = self.dims.yard_row_start  # first yard row
        split = 0
        tier = 0
        n_splits = 20  # 40ft container

        result = self.coord.validate_yard_dest(unified_row, split, tier, n_splits)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, PlacementResult)
        self.assertEqual(result.row, 0)  # yard-local row
        self.assertEqual(result.tier, 0)

    def test_validate_yard_dest_occupied_rejected(self):
        """Dest overlapping an occupied position is rejected."""
        # Place a container at yard row 0, bay 5
        c, pl = self._place_in_yard("BLOCK", bay=5)
        self.assertIsNotNone(c)

        unified_row = self.dims.yard_row_start + pl.row
        # Try to place at the same spot
        result = self.coord.validate_yard_dest(
            unified_row, pl.abs_start, pl.tier, 20,
        )
        self.assertIsNone(result)

    def test_validate_yard_dest_tier_support_required(self):
        """Tier > 0 needs support (occupied tier below)."""
        unified_row = self.dims.yard_row_start
        split = 0
        n_splits = 20

        # Tier 1 without anything below → rejected
        result = self.coord.validate_yard_dest(unified_row, split, 1, n_splits)
        self.assertIsNone(result)

    def test_validate_yard_dest_tier_with_support(self):
        """Tier 1 succeeds when tier 0 is occupied underneath."""
        # Place container at tier 0 first
        c = self._make_container("BASE")
        placement = PlacementResult(row=0, bay=0, tier=0, start_split=0)
        self.yard.add_container(c, placement)

        unified_row = self.dims.yard_row_start
        # Now tier 1 should be valid above the same position
        result = self.coord.validate_yard_dest(unified_row, 0, 1, 20)
        self.assertIsNotNone(result)
        self.assertEqual(result.tier, 1)

    def test_validate_yard_dest_rail_row_rejected(self):
        """validate_yard_dest for a rail-region row returns None."""
        result = self.coord.validate_yard_dest(0, 0, 0, 20)
        self.assertIsNone(result)

    def test_validate_yard_dest_split_overflow(self):
        """Split + n_splits exceeding total_splits is rejected."""
        unified_row = self.dims.yard_row_start
        # total_splits = 10 * 20 = 200, try placing at split 195 with n_splits 20
        result = self.coord.validate_yard_dest(
            unified_row, 195, 0, 20,
        )
        self.assertIsNone(result)

    def test_validate_yard_dest_zero_splits_rejected(self):
        """n_splits <= 0 is rejected."""
        unified_row = self.dims.yard_row_start
        result = self.coord.validate_yard_dest(unified_row, 0, 0, 0)
        self.assertIsNone(result)
        result = self.coord.validate_yard_dest(unified_row, 0, 0, -1)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
