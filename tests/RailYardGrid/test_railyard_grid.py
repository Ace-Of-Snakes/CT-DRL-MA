"""Tests for RailYard grid-based spatial tracking.

Validates that the rewritten RailYard maintains correct spatial grids
(occupancy_mask, position_grid) synchronised with train/wagon state,
and exposes O(1) lookups via get_container_at / get_record_at.
"""
import unittest
import numpy as np
from datetime import datetime

from simulation.core.containers.container import Container
from simulation.core.vehicles.train import Train
from simulation.core.enums import Direction, GoodsType
from simulation.core.facilities.railyard import (
    RailYard, RailSlot, RailContainerRecord, _container_n_splits,
)
from simulation.core.facilities.base import EMPTY_SLOT
from simulation.core.facilities.constants import BAY_SPLIT_FACTOR


class TestRailYardGrid(unittest.TestCase):
    """Tests for RailYard grid arrays and record management."""

    def setUp(self):
        self.rail = RailYard(n_tracks=4, n_bays=10, split_factor=BAY_SPLIT_FACTOR)
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
        n_wagons: int = 3,
        containers_per_wagon: int = 1,
        length_ft: int = 40,
    ) -> Train:
        """Create a train and load containers onto its wagons."""
        train = Train(train_id=train_id, num_wagons=n_wagons)
        idx = 0
        for w_idx in range(n_wagons):
            for _ in range(containers_per_wagon):
                c = self._make_container(
                    f"{train_id}_C{idx:03d}", length_ft=length_ft,
                )
                train.add_container(c, wagon_index=w_idx)
                idx += 1
        return train

    # ── Initialisation ───────────────────────────────────────────────

    def test_initial_state_empty(self):
        """Empty rail yard has all-zero occupancy and EMPTY position grid."""
        self.assertFalse(self.rail.occupancy_mask.any())
        self.assertTrue((self.rail.position_grid == EMPTY_SLOT).all())
        self.assertEqual(self.rail.container_count(), 0)
        self.assertEqual(list(self.rail.iter_records()), [])

    def test_grid_dimensions(self):
        """Grid shapes match constructor args."""
        ts = 10 * BAY_SPLIT_FACTOR
        self.assertEqual(self.rail.occupancy_mask.shape, (4, ts))
        self.assertEqual(self.rail.position_grid.shape, (4, ts))

    # ── slot_train populates grid ────────────────────────────────────

    def test_slot_train_populates_grid(self):
        """Slotting a train with containers fills occupancy and position grids."""
        train = self._make_train_with_containers("T1", n_wagons=2, containers_per_wagon=1)
        slot = RailSlot(track_id=0, anchor_bay=2)
        self.rail.slot_train(train, slot)

        # Track 0 should have occupancy at wagons 2 and 3 (anchor + wagon_idx)
        sf = BAY_SPLIT_FACTOR
        for w in range(2):
            bay = 2 + w
            s0 = bay * sf
            # 40ft container occupies ceil(40 / 2.0) = 20 splits
            n_splits = _container_n_splits(
                self._make_container(length_ft=40), sf,
            )
            self.assertTrue(
                self.rail.occupancy_mask[0, s0:s0 + n_splits].all(),
                f"Wagon {w} at bay {bay} should be occupied",
            )

        # Container count matches
        self.assertEqual(self.rail.container_count(), 2)

    def test_slot_train_records_correct(self):
        """Records created by slot_train have correct track, split, train_id."""
        train = self._make_train_with_containers("T1", n_wagons=1, containers_per_wagon=1)
        slot = RailSlot(track_id=1, anchor_bay=5)
        self.rail.slot_train(train, slot)

        records = list(self.rail.iter_records())
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec.track, 1)
        self.assertEqual(rec.train_id, "T1")
        self.assertEqual(rec.abs_split, 5 * BAY_SPLIT_FACTOR)
        self.assertIsNotNone(rec.container)
        self.assertEqual(rec.container.container_id, "T1_C000")

    # ── O(1) lookups ─────────────────────────────────────────────────

    def test_get_container_at(self):
        """get_container_at returns correct container at occupied split."""
        train = self._make_train_with_containers("T1", n_wagons=1, containers_per_wagon=1)
        slot = RailSlot(track_id=2, anchor_bay=0)
        self.rail.slot_train(train, slot)

        c = self.rail.get_container_at(2, 0)
        self.assertIsNotNone(c)
        self.assertEqual(c.container_id, "T1_C000")

    def test_get_container_at_empty(self):
        """get_container_at returns None for empty position."""
        self.assertIsNone(self.rail.get_container_at(0, 0))

    def test_get_container_at_out_of_range(self):
        """get_container_at returns None for out-of-range coords."""
        self.assertIsNone(self.rail.get_container_at(-1, 0))
        self.assertIsNone(self.rail.get_container_at(99, 0))
        self.assertIsNone(self.rail.get_container_at(0, -1))
        self.assertIsNone(self.rail.get_container_at(0, 9999))

    def test_get_record_at(self):
        """get_record_at returns full record with placement info."""
        train = self._make_train_with_containers("T1", n_wagons=1, containers_per_wagon=1)
        slot = RailSlot(track_id=0, anchor_bay=3)
        self.rail.slot_train(train, slot)

        rec = self.rail.get_record_at(0, 3 * BAY_SPLIT_FACTOR)
        self.assertIsNotNone(rec)
        self.assertIsInstance(rec, RailContainerRecord)
        self.assertEqual(rec.track, 0)
        self.assertEqual(rec.wagon_index, 0)

    def test_get_record_by_id(self):
        """get_record_by_id returns record by container_id."""
        train = self._make_train_with_containers("T1", n_wagons=1, containers_per_wagon=1)
        slot = RailSlot(track_id=0, anchor_bay=0)
        self.rail.slot_train(train, slot)

        rec = self.rail.get_record_by_id("T1_C000")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.container.container_id, "T1_C000")

    def test_get_record_by_id_missing(self):
        """get_record_by_id returns None for unknown container."""
        self.assertIsNone(self.rail.get_record_by_id("NONEXISTENT"))

    # ── sync_train ───────────────────────────────────────────────────

    def test_sync_after_remove_clears_grid(self):
        """After removing a container from a train, sync_train clears it from grid."""
        train = self._make_train_with_containers("T1", n_wagons=2, containers_per_wagon=1)
        slot = RailSlot(track_id=0, anchor_bay=0)
        self.rail.slot_train(train, slot)

        self.assertEqual(self.rail.container_count(), 2)

        # Remove one container from the train
        train.remove_container("T1_C000")
        self.rail.sync_train("T1", train)

        self.assertEqual(self.rail.container_count(), 1)
        # The first wagon's bay should now be empty
        self.assertIsNone(self.rail.get_container_at(0, 0))
        # The second wagon's container should still be there
        self.assertIsNotNone(self.rail.get_container_at(0, 1 * BAY_SPLIT_FACTOR))

    def test_sync_after_add_populates_grid(self):
        """After adding a container to a train, sync_train updates the grid."""
        train = Train(train_id="T1", num_wagons=3)
        slot = RailSlot(track_id=1, anchor_bay=0)
        self.rail.slot_train(train, slot)

        self.assertEqual(self.rail.container_count(), 0)

        # Add container to wagon 0
        c = self._make_container("NEW_C", length_ft=20)
        train.add_container(c, wagon_index=0)
        self.rail.sync_train("T1", train)

        self.assertEqual(self.rail.container_count(), 1)
        self.assertIsNotNone(self.rail.get_container_at(1, 0))

    def test_sync_unslotted_train_is_noop(self):
        """sync_train for an unslotted train does nothing."""
        train = self._make_train_with_containers("T_UNSLT", n_wagons=1)
        self.rail.sync_train("T_UNSLT", train)
        self.assertEqual(self.rail.container_count(), 0)

    # ── release_train ────────────────────────────────────────────────

    def test_release_train_clears_grid(self):
        """Releasing a train clears all its grid entries."""
        train = self._make_train_with_containers("T1", n_wagons=3, containers_per_wagon=1)
        slot = RailSlot(track_id=0, anchor_bay=0)
        self.rail.slot_train(train, slot)

        self.assertEqual(self.rail.container_count(), 3)

        released = self.rail.release_train("T1")
        self.assertIsNotNone(released)
        self.assertEqual(self.rail.container_count(), 0)
        self.assertFalse(self.rail.occupancy_mask.any())

    def test_release_unknown_train(self):
        """Releasing an unslotted train returns None."""
        self.assertIsNone(self.rail.release_train("UNKNOWN"))

    # ── Multiple trains on different tracks ──────────────────────────

    def test_multiple_trains_independent(self):
        """Two trains on different tracks have independent grid entries."""
        t1 = self._make_train_with_containers("T1", n_wagons=2, containers_per_wagon=1)
        t2 = self._make_train_with_containers("T2", n_wagons=2, containers_per_wagon=1)

        self.rail.slot_train(t1, RailSlot(track_id=0, anchor_bay=0))
        self.rail.slot_train(t2, RailSlot(track_id=1, anchor_bay=5))

        self.assertEqual(self.rail.container_count(), 4)

        # Release T1 only
        self.rail.release_train("T1")
        self.assertEqual(self.rail.container_count(), 2)

        # T2 containers should still be there
        for rec in self.rail.iter_records():
            self.assertEqual(rec.train_id, "T2")

    # ── Re-slotting a train to a different track ─────────────────────

    def test_reslot_train_moves_grid(self):
        """Re-slotting a train to a new track moves grid entries."""
        train = self._make_train_with_containers("T1", n_wagons=1, containers_per_wagon=1)
        self.rail.slot_train(train, RailSlot(track_id=0, anchor_bay=0))

        self.assertTrue(self.rail.occupancy_mask[0].any())
        self.assertFalse(self.rail.occupancy_mask[2].any())

        # Re-slot to track 2
        self.rail.slot_train(train, RailSlot(track_id=2, anchor_bay=0))

        self.assertFalse(self.rail.occupancy_mask[0].any())
        self.assertTrue(self.rail.occupancy_mask[2].any())
        self.assertEqual(self.rail.container_count(), 1)

    # ── iter_records ─────────────────────────────────────────────────

    def test_iter_records_all_containers(self):
        """iter_records yields every container across all trains."""
        t1 = self._make_train_with_containers("T1", n_wagons=2, containers_per_wagon=2)
        t2 = self._make_train_with_containers("T2", n_wagons=1, containers_per_wagon=1)

        self.rail.slot_train(t1, RailSlot(track_id=0, anchor_bay=0))
        self.rail.slot_train(t2, RailSlot(track_id=1, anchor_bay=0))

        records = list(self.rail.iter_records())
        self.assertEqual(len(records), 5)  # 4 from T1 + 1 from T2

        cids = {r.container.container_id for r in records}
        self.assertIn("T1_C000", cids)
        self.assertIn("T2_C000", cids)

    # ── Slot queries ─────────────────────────────────────────────────

    def test_get_slot_and_anchor(self):
        """get_slot / get_anchor_bay return correct slot info."""
        train = self._make_train_with_containers("T1", n_wagons=1)
        slot = RailSlot(track_id=3, anchor_bay=7)
        self.rail.slot_train(train, slot)

        self.assertEqual(self.rail.get_anchor_bay("T1"), 7)
        s = self.rail.get_slot("T1")
        self.assertIsNotNone(s)
        self.assertEqual(s.track_id, 3)

    def test_get_trains_on_track(self):
        """get_trains_on_track lists trains on a given track."""
        t1 = self._make_train_with_containers("T1", n_wagons=1)
        t2 = self._make_train_with_containers("T2", n_wagons=1)

        self.rail.slot_train(t1, RailSlot(track_id=0, anchor_bay=0))
        self.rail.slot_train(t2, RailSlot(track_id=0, anchor_bay=5))

        ids = self.rail.get_trains_on_track(0)
        self.assertEqual(sorted(ids), ["T1", "T2"])
        self.assertEqual(self.rail.get_trains_on_track(1), [])

    # ── 20ft container splits ────────────────────────────────────────

    def test_20ft_container_splits(self):
        """20ft containers occupy fewer splits than 40ft."""
        sf = BAY_SPLIT_FACTOR
        n20 = _container_n_splits(self._make_container(length_ft=20), sf)
        n40 = _container_n_splits(self._make_container(length_ft=40), sf)
        self.assertLess(n20, n40)
        # 20ft → ceil(20 / 2.0) = 10 splits
        self.assertEqual(n20, 10)
        # 40ft → ceil(40 / 2.0) = 20 splits
        self.assertEqual(n40, 20)

    def test_two_20ft_on_same_wagon(self):
        """Two 20ft containers on one wagon occupy adjacent split ranges."""
        train = Train(train_id="T1", num_wagons=1)
        c1 = self._make_container("C1", length_ft=20)
        c2 = self._make_container("C2", length_ft=20)
        train.add_container(c1, wagon_index=0)
        train.add_container(c2, wagon_index=0)

        self.rail.slot_train(train, RailSlot(track_id=0, anchor_bay=0))

        self.assertEqual(self.rail.container_count(), 2)
        rec1 = self.rail.get_record_by_id("C1")
        rec2 = self.rail.get_record_by_id("C2")
        self.assertIsNotNone(rec1)
        self.assertIsNotNone(rec2)

        # C1 starts at split 0, C2 starts at split 10 (both on bay 0)
        self.assertEqual(rec1.abs_split, 0)
        self.assertEqual(rec1.n_splits, 10)
        self.assertEqual(rec2.abs_split, 10)
        self.assertEqual(rec2.n_splits, 10)

    # ── Edge cases ───────────────────────────────────────────────────

    def test_wagon_beyond_n_bays_skipped(self):
        """Wagons that extend beyond n_bays are silently skipped."""
        train = self._make_train_with_containers("T1", n_wagons=3, containers_per_wagon=1)
        # Anchor at bay 9, last bay → wagons at 9, 10(skip), 11(skip)
        self.rail.slot_train(train, RailSlot(track_id=0, anchor_bay=9))

        # Only 1 wagon fits (bay 9)
        self.assertEqual(self.rail.container_count(), 1)

    def test_occupancy_mask_reference(self):
        """get_occupancy_mask returns a reference to the internal array."""
        mask = self.rail.get_occupancy_mask()
        self.assertIs(mask, self.rail.occupancy_mask)


if __name__ == "__main__":
    unittest.main()
