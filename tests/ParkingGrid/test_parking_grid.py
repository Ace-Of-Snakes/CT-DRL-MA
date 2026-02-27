"""Tests for ParkingArea grid-based spatial tracking.

Validates that the enhanced ParkingArea maintains correct flat spatial
grids (occupancy_flat, position_grid) alongside the existing bay-level
arrays, and exposes O(1) lookups via get_truck_at / get_record_at.
"""
import unittest
import numpy as np
from datetime import datetime
from unittest.mock import Mock

from simulation.core.vehicles.truck import Truck
from simulation.core.containers.container import Container
from simulation.core.enums import Direction, GoodsType, TruckStatus
from simulation.core.facilities.parking import (
    ParkingArea, ParkingRecord, ParkingSpot,
)
from simulation.core.facilities.base import EMPTY_SLOT
from simulation.core.facilities.constants import BAY_SPLIT_FACTOR


class TestParkingGrid(unittest.TestCase):
    """Tests for ParkingArea grid arrays and record management."""

    def setUp(self):
        self.parking = ParkingArea(n_bays=10, split_factor=BAY_SPLIT_FACTOR)
        self.now = datetime(2025, 6, 15, 10, 0, 0)

    # ── Helpers ──────────────────────────────────────────────────────

    def _make_truck(self, truck_id: str = "TK001") -> Truck:
        return Truck(truck_id=truck_id, arrival_time=self.now)

    def _make_container(self, cid: str = "C001") -> Container:
        return Container(
            container_id=cid,
            direction=Direction.EXPORT,
            container_type="40ft",
            arrival_date=self.now,
            departure_date=self.now,
            goods_type=GoodsType.REGULAR,
            length_ft=40,
            length_m=12.2,
            width_m=2.44,
            height_m=2.59,
        )

    # ── Initial state ────────────────────────────────────────────────

    def test_initial_state_empty(self):
        """New parking area has empty grids."""
        self.assertFalse(self.parking.occupancy_flat.any())
        self.assertTrue((self.parking.position_grid == EMPTY_SLOT).all())
        self.assertEqual(list(self.parking.iter_records()), [])

    def test_grid_dimensions(self):
        """Flat grid dimensions match n_bays * split_factor."""
        ts = 10 * BAY_SPLIT_FACTOR
        self.assertEqual(self.parking.occupancy_flat.shape, (ts,))
        self.assertEqual(self.parking.position_grid.shape, (ts,))

    # ── allocate ─────────────────────────────────────────────────────

    def test_allocate_updates_flat_grid(self):
        """Allocating a truck marks occupancy_flat and position_grid."""
        tk = self._make_truck("TK1")
        ok = self.parking.allocate(tk, bay=3, split=0)
        self.assertTrue(ok)

        abs_split = 3 * BAY_SPLIT_FACTOR
        self.assertTrue(self.parking.occupancy_flat[abs_split])
        self.assertNotEqual(self.parking.position_grid[abs_split], EMPTY_SLOT)

    def test_allocate_creates_record(self):
        """Allocating creates a ParkingRecord with correct fields."""
        tk = self._make_truck("TK1")
        self.parking.allocate(tk, bay=5, split=0)

        records = list(self.parking.iter_records())
        self.assertEqual(len(records), 1)

        rec = records[0]
        self.assertIsInstance(rec, ParkingRecord)
        self.assertIs(rec.truck, tk)
        self.assertEqual(rec.bay, 5)
        self.assertEqual(rec.abs_split, 5 * BAY_SPLIT_FACTOR)

    def test_allocate_same_bay_rejected(self):
        """Second truck in the same bay is rejected (bay-level rule)."""
        tk1 = self._make_truck("TK1")
        tk2 = self._make_truck("TK2")

        self.assertTrue(self.parking.allocate(tk1, bay=0, split=0))
        self.assertFalse(self.parking.allocate(tk2, bay=0, split=0))

    def test_allocate_out_of_range(self):
        """Allocating outside bay range fails."""
        tk = self._make_truck("TK1")
        self.assertFalse(self.parking.allocate(tk, bay=-1, split=0))
        self.assertFalse(self.parking.allocate(tk, bay=10, split=0))

    def test_allocate_sets_parking_spot(self):
        """Allocating sets the truck's parking_spot attribute."""
        tk = self._make_truck("TK1")
        self.parking.allocate(tk, bay=7, split=0)

        self.assertIsNotNone(tk.parking_spot)
        self.assertIsInstance(tk.parking_spot, ParkingSpot)
        self.assertEqual(tk.parking_spot.bay, 7)

    # ── release ──────────────────────────────────────────────────────

    def test_release_clears_grid(self):
        """Releasing a truck clears flat grid and record."""
        tk = self._make_truck("TK1")
        self.parking.allocate(tk, bay=2, split=0)

        abs_split = 2 * BAY_SPLIT_FACTOR
        self.assertTrue(self.parking.occupancy_flat[abs_split])

        ok = self.parking.release(tk)
        self.assertTrue(ok)

        self.assertFalse(self.parking.occupancy_flat[abs_split])
        self.assertEqual(self.parking.position_grid[abs_split], EMPTY_SLOT)
        self.assertEqual(list(self.parking.iter_records()), [])

    def test_release_clears_parking_spot(self):
        """Releasing clears the truck's parking_spot."""
        tk = self._make_truck("TK1")
        self.parking.allocate(tk, bay=4, split=0)
        self.parking.release(tk)
        self.assertIsNone(tk.parking_spot)

    def test_release_unknown_truck(self):
        """Releasing an unparked truck returns False."""
        tk = self._make_truck("GHOST")
        self.assertFalse(self.parking.release(tk))

    # ── O(1) lookups ─────────────────────────────────────────────────

    def test_get_truck_at(self):
        """get_truck_at returns correct truck at occupied split."""
        tk = self._make_truck("TK1")
        self.parking.allocate(tk, bay=6, split=0)

        result = self.parking.get_truck_at(6 * BAY_SPLIT_FACTOR)
        self.assertIs(result, tk)

    def test_get_truck_at_empty(self):
        """get_truck_at returns None for empty position."""
        self.assertIsNone(self.parking.get_truck_at(0))

    def test_get_truck_at_out_of_range(self):
        """get_truck_at returns None for out-of-range splits."""
        self.assertIsNone(self.parking.get_truck_at(-1))
        self.assertIsNone(self.parking.get_truck_at(9999))

    def test_get_record_at(self):
        """get_record_at returns full ParkingRecord."""
        tk = self._make_truck("TK1")
        self.parking.allocate(tk, bay=1, split=0)

        rec = self.parking.get_record_at(1 * BAY_SPLIT_FACTOR)
        self.assertIsNotNone(rec)
        self.assertIs(rec.truck, tk)
        self.assertEqual(rec.bay, 1)

    def test_get_record_by_id(self):
        """get_record_by_id returns record by truck_id."""
        tk = self._make_truck("TK1")
        self.parking.allocate(tk, bay=0, split=0)

        rec = self.parking.get_record_by_id("TK1")
        self.assertIsNotNone(rec)
        self.assertIs(rec.truck, tk)

    def test_get_record_by_id_missing(self):
        """get_record_by_id returns None for unknown truck."""
        self.assertIsNone(self.parking.get_record_by_id("UNKNOWN"))

    # ── iter_records ─────────────────────────────────────────────────

    def test_iter_records_multiple_trucks(self):
        """iter_records yields all parked trucks."""
        for i in range(5):
            tk = self._make_truck(f"TK{i}")
            self.parking.allocate(tk, bay=i, split=0)

        records = list(self.parking.iter_records())
        self.assertEqual(len(records), 5)

        ids = {r.truck.truck_id for r in records}
        for i in range(5):
            self.assertIn(f"TK{i}", ids)

    def test_iter_records_after_release(self):
        """iter_records excludes released trucks."""
        tk1 = self._make_truck("TK1")
        tk2 = self._make_truck("TK2")
        self.parking.allocate(tk1, bay=0, split=0)
        self.parking.allocate(tk2, bay=1, split=0)

        self.parking.release(tk1)

        records = list(self.parking.iter_records())
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].truck.truck_id, "TK2")

    # ── Existing API preserved ───────────────────────────────────────

    def test_is_free(self):
        """is_free checks bay-level occupancy."""
        self.assertTrue(self.parking.is_free(0))
        tk = self._make_truck("TK1")
        self.parking.allocate(tk, bay=0, split=0)
        self.assertFalse(self.parking.is_free(0))

    def test_get_truck_spot(self):
        """get_truck_spot returns (bay, split) for parked truck."""
        tk = self._make_truck("TK1")
        self.parking.allocate(tk, bay=8, split=0)

        spot = self.parking.get_truck_spot("TK1")
        self.assertEqual(spot, (8, 0))

    def test_get_occupancy_flat(self):
        """get_occupancy_flat returns reference to flat occupancy."""
        flat = self.parking.get_occupancy_flat()
        self.assertIs(flat, self.parking.occupancy_flat)

    def test_bay_level_occupancy_marks_entire_bay(self):
        """Allocating marks entire bay as occupied (one truck per bay)."""
        tk = self._make_truck("TK1")
        self.parking.allocate(tk, bay=3, split=0)
        self.assertTrue(self.parking.occupied[3].all())

    # ── Allocate/release cycle ───────────────────────────────────────

    def test_allocate_release_cycle(self):
        """Bay can be reused after release."""
        tk1 = self._make_truck("TK1")
        self.parking.allocate(tk1, bay=0, split=0)
        self.parking.release(tk1)

        tk2 = self._make_truck("TK2")
        ok = self.parking.allocate(tk2, bay=0, split=0)
        self.assertTrue(ok)

        rec = self.parking.get_record_at(0)
        self.assertIsNotNone(rec)
        self.assertIs(rec.truck, tk2)


if __name__ == "__main__":
    unittest.main()
