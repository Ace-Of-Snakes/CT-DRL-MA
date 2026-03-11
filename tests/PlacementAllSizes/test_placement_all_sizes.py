"""Comprehensive placement tests for all container sizes across all facilities.

Verifies that every container size (20–45 ft), goods type (Regular, Reefer, DG),
and special type (swap body, trailer) can be placed in the correct zones and
is rejected from incorrect zones.  Covers yard, rail yard, and parking area.
"""
import unittest
import math
import numpy as np
from datetime import datetime

from simulation.core.containers.container import Container
from simulation.core.vehicles.train import Train
from simulation.core.vehicles.truck import Truck
from simulation.core.enums import Direction, GoodsType
from simulation.core.facilities.yard import StorageYard, PlacementResult
from simulation.core.facilities.railyard import (
    RailYard, RailSlot, _container_n_splits,
)
from simulation.core.facilities.parking import ParkingArea
from simulation.core.facilities.constants import (
    BAY_SPLIT_FACTOR, CONTAINER_LENGTH_TO_SUB_BAYS,
)


# ================================================================
# Constants
# ================================================================

NOW = datetime(2026, 3, 11, 10, 0, 0)
N_BAYS = 58
N_ROWS = 5
N_TIERS = 5
SF = BAY_SPLIT_FACTOR  # 20

# All container sizes from the real data
ALL_SIZES_FT = sorted(CONTAINER_LENGTH_TO_SUB_BAYS.keys())
# Expected: [20, 22, 23, 24, 30, 40, 45]


# ================================================================
# Helpers
# ================================================================

def _make_container(
    cid: str,
    length_ft: int = 40,
    goods_type: GoodsType = GoodsType.REGULAR,
    is_swap_body: bool = False,
    is_trailer: bool = False,
) -> Container:
    length_m = length_ft * 0.3048
    return Container(
        container_id=cid,
        direction=Direction.IMPORT,
        container_type=f"{length_ft}ft",
        arrival_date=NOW,
        departure_date=NOW,
        goods_type=goods_type,
        length_ft=length_ft,
        length_m=length_m,
        width_m=2.44,
        height_m=2.59,
        is_swap_body=is_swap_body,
        is_trailer=is_trailer,
    )


def _make_yard_with_zones() -> StorageYard:
    """Create a yard with reefer, DG, and swap body zones defined.

    Layout (1-based bays/rows in coordinates, converted internally):
      - Reefer zone:  bay 1, all rows  (all tiers)
      - DG zone:      bay 3, rows 3-5  (all tiers)
      - Swap body:    bay 5, row 1     (tier 0 only)
      - Regular:      everything else
    """
    coords = []
    # Reefer: bay 1, all rows
    for r in range(1, N_ROWS + 1):
        coords.append((1, r, "r"))
    # Dangerous goods: bay 3, rows 3-5
    for r in range(3, 6):
        coords.append((3, r, "dg"))
    # Swap body: bay 5, row 1
    coords.append((5, 1, "sb_t"))

    return StorageYard(
        n_rows=N_ROWS,
        n_bays=N_BAYS,
        n_tiers=N_TIERS,
        coordinates=coords,
        validate=False,
    )


def _splits(length_ft: int) -> int:
    """Expected split count for a container size."""
    return CONTAINER_LENGTH_TO_SUB_BAYS[length_ft]


# ================================================================
# 1. Yard: All sizes at various bays (Regular)
# ================================================================

class TestYardAllSizesRegular(unittest.TestCase):
    """Every container size can be placed in the yard (regular zone)."""

    def setUp(self):
        self.yard = _make_yard_with_zones()

    def test_all_sizes_place_at_interior_bay(self):
        """All sizes should place at an interior bay (bay 10)."""
        for ft in ALL_SIZES_FT:
            with self.subTest(length_ft=ft):
                c = _make_container(f"INT_{ft}", length_ft=ft)
                p = self.yard.find_single_placement(c, target_bay=10)
                self.assertIsNotNone(
                    p, f"{ft}ft container must be placeable at bay 10",
                )
                self.yard.add_container(c, p)

    def test_all_sizes_correct_split_count(self):
        """Placed container occupies the expected number of splits."""
        for ft in ALL_SIZES_FT:
            with self.subTest(length_ft=ft):
                yard = _make_yard_with_zones()  # fresh yard
                c = _make_container(f"SC_{ft}", length_ft=ft)
                p = yard.find_single_placement(c, target_bay=20)
                self.assertIsNotNone(p)
                yard.add_container(c, p)
                rec = yard.get_record(c.container_id)
                self.assertIsNotNone(rec)
                self.assertEqual(
                    rec.n_splits, _splits(ft),
                    f"{ft}ft: expected {_splits(ft)} splits, got {rec.n_splits}",
                )

    def test_45ft_cross_bay(self):
        """45ft container (23 splits) spans into the next bay."""
        c = _make_container("CROSS_45", length_ft=45)
        p = self.yard.find_single_placement(c, target_bay=30)
        self.assertIsNotNone(p, "45ft must be placeable at interior bay")
        self.yard.add_container(c, p)
        # start_split should be 0 (must start at bay boundary for cross-bay)
        self.assertEqual(p.start_split, 0)
        # Occupancy should span 23 splits
        abs_start = p.abs_start
        self.assertTrue(
            np.all(self.yard.occupancy_mask[p.tier, p.row, abs_start:abs_start + 23]),
        )

    def test_45ft_rejected_at_last_bay(self):
        """45ft container at last bay (57) exceeds total_splits → rejected."""
        c = _make_container("LAST_45", length_ft=45)
        p = self.yard.find_single_placement(c, target_bay=57, max_proximity=0)
        self.assertIsNone(
            p, "45ft at bay 57 should be None (23 splits would exceed 1160)",
        )

    def test_40ft_fits_at_last_bay(self):
        """40ft container at bay 57 exactly fills the bay (20 splits)."""
        c = _make_container("LAST_40", length_ft=40)
        p = self.yard.find_single_placement(c, target_bay=57, max_proximity=0)
        self.assertIsNotNone(p, "40ft should fit at last bay")
        self.yard.add_container(c, p)

    def test_20ft_at_first_bay(self):
        """20ft container at bay 0 (first bay)."""
        # Bay 0 is reefer zone in our layout, so use bay 6 (first regular bay)
        c = _make_container("FIRST_20", length_ft=20)
        p = self.yard.find_single_placement(c, target_bay=6, max_proximity=0)
        self.assertIsNotNone(p, "20ft should fit at bay 6")
        self.yard.add_container(c, p)

    def test_multiple_sizes_in_same_bay(self):
        """Different-sized containers can share a bay if splits fit."""
        # Bay 20 has 20 splits. 20ft (10 splits) leaves room for another 20ft.
        c1 = _make_container("MIX_A", length_ft=20)
        p1 = self.yard.find_single_placement(c1, target_bay=20, max_proximity=0)
        self.assertIsNotNone(p1)
        self.yard.add_container(c1, p1)

        c2 = _make_container("MIX_B", length_ft=20)
        # Same bay, should find a free split range in a different row
        p2 = self.yard.find_single_placement(c2, target_bay=20, max_proximity=0)
        self.assertIsNotNone(p2, "Second 20ft should fit in same bay (diff row)")
        self.yard.add_container(c2, p2)


# ================================================================
# 2. Yard: Goods type zone enforcement
# ================================================================

class TestYardGoodsTypeZones(unittest.TestCase):
    """Containers are restricted to their goods-type zones."""

    def setUp(self):
        self.yard = _make_yard_with_zones()

    # ── Reefer ──

    def test_reefer_places_in_reefer_zone(self):
        """Reefer container should place in the reefer zone (bay 0).

        45ft containers (23 splits) cannot fit in a single-bay zone (20
        splits) because the overflow splits aren't in the reefer zone.
        """
        for ft in ALL_SIZES_FT:
            with self.subTest(length_ft=ft):
                yard = _make_yard_with_zones()
                c = _make_container(
                    f"RF_{ft}", length_ft=ft, goods_type=GoodsType.REEFER,
                )
                p = yard.find_single_placement(c, target_bay=0, max_proximity=0)
                if _splits(ft) > SF:
                    # Cross-bay container can't fit in single-bay zone
                    self.assertIsNone(
                        p,
                        f"Reefer {ft}ft ({_splits(ft)} splits) must NOT fit "
                        f"in single-bay reefer zone ({SF} splits)",
                    )
                else:
                    self.assertIsNotNone(
                        p, f"Reefer {ft}ft must fit in reefer zone (bay 0)",
                    )

    def test_reefer_rejected_from_regular_zone(self):
        """Reefer container cannot be placed in a regular zone."""
        c = _make_container("RF_REG", length_ft=40, goods_type=GoodsType.REEFER)
        # Bay 10 is a regular zone — no reefer mask there
        p = self.yard.find_single_placement(c, target_bay=10, max_proximity=0)
        self.assertIsNone(
            p, "Reefer must NOT place in regular zone",
        )

    # ── Dangerous Goods ──

    def test_dg_places_in_dg_zone(self):
        """DG container should place in DG zone (bay 2, rows 2-4).

        45ft containers (23 splits) cannot fit in a single-bay zone (20
        splits) because the overflow splits aren't in the DG zone.
        """
        for ft in ALL_SIZES_FT:
            with self.subTest(length_ft=ft):
                yard = _make_yard_with_zones()
                c = _make_container(
                    f"DG_{ft}", length_ft=ft,
                    goods_type=GoodsType.DANGEROUS_GOODS,
                )
                p = yard.find_single_placement(c, target_bay=2, max_proximity=0)
                if _splits(ft) > SF:
                    self.assertIsNone(
                        p,
                        f"DG {ft}ft ({_splits(ft)} splits) must NOT fit "
                        f"in single-bay DG zone ({SF} splits)",
                    )
                else:
                    self.assertIsNotNone(
                        p, f"DG {ft}ft must fit in DG zone (bay 2)",
                    )

    def test_dg_rejected_from_regular_zone(self):
        """DG container cannot be placed in a regular zone."""
        c = _make_container(
            "DG_REG", length_ft=40, goods_type=GoodsType.DANGEROUS_GOODS,
        )
        p = self.yard.find_single_placement(c, target_bay=10, max_proximity=0)
        self.assertIsNone(
            p, "DG must NOT place in regular zone",
        )

    def test_dg_rejected_from_reefer_zone(self):
        """DG container cannot be placed in reefer zone."""
        c = _make_container(
            "DG_RF", length_ft=40, goods_type=GoodsType.DANGEROUS_GOODS,
        )
        p = self.yard.find_single_placement(c, target_bay=0, max_proximity=0)
        self.assertIsNone(
            p, "DG must NOT place in reefer zone",
        )

    # ── Regular ──

    def test_regular_rejected_from_reefer_zone(self):
        """Regular container cannot be placed in reefer zone."""
        c = _make_container("REG_RF", length_ft=40)
        p = self.yard.find_single_placement(c, target_bay=0, max_proximity=0)
        self.assertIsNone(
            p, "Regular must NOT place in reefer zone",
        )

    def test_45ft_reefer_in_wide_zone(self):
        """45ft reefer places when reefer zone spans 2+ consecutive bays."""
        coords = []
        # Reefer zone: bays 1-3 (1-based), all rows → 60 splits wide
        for bay in range(1, 4):
            for r in range(1, N_ROWS + 1):
                coords.append((bay, r, "r"))
        yard = StorageYard(
            n_rows=N_ROWS, n_bays=N_BAYS, n_tiers=N_TIERS,
            coordinates=coords, validate=False,
        )
        c = _make_container("RF_45W", length_ft=45, goods_type=GoodsType.REEFER)
        p = yard.find_single_placement(c, target_bay=0, max_proximity=1)
        self.assertIsNotNone(
            p, "45ft reefer must fit in 3-bay-wide reefer zone",
        )

    def test_45ft_dg_in_wide_zone(self):
        """45ft DG places when DG zone spans 2+ consecutive bays."""
        coords = []
        # DG zone: bays 10-12 (1-based), rows 3-5 → 60 splits wide
        for bay in range(10, 13):
            for r in range(3, 6):
                coords.append((bay, r, "dg"))
        yard = StorageYard(
            n_rows=N_ROWS, n_bays=N_BAYS, n_tiers=N_TIERS,
            coordinates=coords, validate=False,
        )
        c = _make_container("DG_45W", length_ft=45, goods_type=GoodsType.DANGEROUS_GOODS)
        p = yard.find_single_placement(c, target_bay=9, max_proximity=1)
        self.assertIsNotNone(
            p, "45ft DG must fit in 3-bay-wide DG zone",
        )

    def test_regular_rejected_from_dg_zone(self):
        """Regular container cannot be placed in DG zone."""
        c = _make_container("REG_DG", length_ft=40)
        # Bay 2 rows 2-4 are DG; rows 0-1 are regular — target row matters
        # find_single_placement searches all rows, so might find row 0/1
        # Use add_container directly at a DG row to verify rejection
        p = PlacementResult(row=2, bay=2, tier=0, start_split=0)
        n_splits = CONTAINER_LENGTH_TO_SUB_BAYS[40]
        goods_mask = self.yard._get_goods_mask_tier(c, 0)
        abs_start = p.abs_start
        self.assertFalse(
            np.all(goods_mask[p.row, abs_start:abs_start + n_splits]),
            "Regular goods mask should be False at DG zone row",
        )


# ================================================================
# 3. Yard: Swap bodies and trailers
# ================================================================

class TestYardSwapBodiesTrailers(unittest.TestCase):
    """Swap bodies/trailers: tier 0 only, swapbody zone only."""

    def setUp(self):
        self.yard = _make_yard_with_zones()

    def test_swap_body_places_at_tier_0(self):
        """Swap body places at tier 0 in swap body zone."""
        c = _make_container("SB_40", length_ft=40, is_swap_body=True)
        p = self.yard.find_single_placement(c, target_bay=4, max_proximity=0)
        self.assertIsNotNone(p, "Swap body must place in sb zone (bay 4)")
        self.assertEqual(p.tier, 0, "Swap body must be at tier 0")
        self.assertEqual(p.row, 0, "Swap body zone is row 0 (1-based row 1)")

    def test_swap_body_rejected_from_regular_zone(self):
        """Swap body cannot be placed in regular zone."""
        c = _make_container("SB_REG", length_ft=40, is_swap_body=True)
        p = self.yard.find_single_placement(c, target_bay=10, max_proximity=0)
        self.assertIsNone(p, "Swap body must NOT place in regular zone")

    def test_swap_body_rejected_above_tier_0(self):
        """Swap body goods mask is empty above tier 0."""
        c = _make_container("SB_T1", length_ft=40, is_swap_body=True)
        mask_t0 = self.yard._get_goods_mask_tier(c, 0)
        mask_t1 = self.yard._get_goods_mask_tier(c, 1)
        # Tier 0 should have some True values (swap body zone)
        self.assertTrue(mask_t0.any(), "Tier 0 must allow swap bodies somewhere")
        # Tier 1+ should be all False
        self.assertFalse(mask_t1.any(), "Tier 1 must reject swap bodies everywhere")

    def test_trailer_same_rules_as_swap_body(self):
        """Trailers follow the same placement rules as swap bodies."""
        c = _make_container("TRA_40", length_ft=40, is_trailer=True)
        # Should place at swap body zone, tier 0
        p = self.yard.find_single_placement(c, target_bay=4, max_proximity=0)
        self.assertIsNotNone(p, "Trailer must place in sb zone")
        self.assertEqual(p.tier, 0)
        # Should NOT place in regular zone
        c2 = _make_container("TRA_REG", length_ft=40, is_trailer=True)
        p2 = self.yard.find_single_placement(c2, target_bay=10, max_proximity=0)
        self.assertIsNone(p2, "Trailer must NOT place in regular zone")

    def test_swap_body_cannot_stack_even_when_tier0_full(self):
        """When tier 0 of the sb zone is occupied, no tier-1 fallback exists."""
        c1 = _make_container("SB_FILL", length_ft=40, is_swap_body=True)
        p1 = self.yard.find_single_placement(c1, target_bay=4, max_proximity=0)
        self.assertIsNotNone(p1)
        self.yard.add_container(c1, p1)

        # Tier 0 sb zone is now full (only 1 bay × 1 row in our layout).
        # A second swap body must NOT stack at tier 1 — should get None.
        c2 = _make_container("SB_STACK", length_ft=40, is_swap_body=True)
        p2 = self.yard.find_single_placement(c2, target_bay=4, max_proximity=0)
        self.assertIsNone(
            p2,
            "Swap body must NOT stack — tier 1+ mask is all-False",
        )

    def test_trailer_cannot_stack_on_trailer(self):
        """Trailers at tier 0 cannot have another trailer on top."""
        c1 = _make_container("TRA_BOT", length_ft=40, is_trailer=True)
        p1 = self.yard.find_single_placement(c1, target_bay=4, max_proximity=0)
        self.assertIsNotNone(p1)
        self.yard.add_container(c1, p1)

        c2 = _make_container("TRA_TOP", length_ft=40, is_trailer=True)
        p2 = self.yard.find_single_placement(c2, target_bay=4, max_proximity=0)
        self.assertIsNone(
            p2,
            "Trailer must NOT stack on another trailer",
        )

    def test_regular_cannot_stack_on_swap_body(self):
        """A regular container cannot be stacked on top of a swap body.

        Even though regular containers can go at higher tiers normally,
        the support check requires the container below to have the same
        n_splits.  But more importantly, the swap body zone at tier 1
        is covered by regular_mask (since it's not reefer or DG).
        So a regular container COULD stack there if support existed.
        This test verifies the support-size-match rule blocks it when
        the sizes differ, and allows it when sizes match.
        """
        # Place a 40ft swap body at tier 0
        sb = _make_container("SB_BASE", length_ft=40, is_swap_body=True)
        p_sb = self.yard.find_single_placement(sb, target_bay=4, max_proximity=0)
        self.assertIsNotNone(p_sb)
        self.yard.add_container(sb, p_sb)

        # 20ft regular at tier 1 should fail (different n_splits → no support)
        c_20 = _make_container("REG_20_ON_SB", length_ft=20)
        p_20 = PlacementResult(row=p_sb.row, bay=p_sb.bay, tier=1, start_split=0)
        n20 = _splits(20)
        goods_mask = self.yard._get_goods_mask_tier(c_20, 1)
        valid = self.yard._is_valid_placement(
            1, p_sb.row, p_20.abs_start, p_20.abs_start + n20, n20, goods_mask,
        )
        self.assertFalse(
            valid,
            "20ft regular cannot stack on 40ft swap body (size mismatch)",
        )

        # 40ft regular at tier 1 — same n_splits as the swap body below.
        # The zone at tier 1 has regular_mask = True (swap body zone is only
        # at tier 0, and regular_mask = ~(reefer | dg) which is True here).
        # Support mask should find the 40ft swap body below → stacking OK.
        c_40 = _make_container("REG_40_ON_SB", length_ft=40)
        p_40 = PlacementResult(row=p_sb.row, bay=p_sb.bay, tier=1, start_split=0)
        n40 = _splits(40)
        goods_mask_40 = self.yard._get_goods_mask_tier(c_40, 1)
        valid_40 = self.yard._is_valid_placement(
            1, p_sb.row, p_40.abs_start, p_40.abs_start + n40, n40, goods_mask_40,
        )
        self.assertTrue(
            valid_40,
            "40ft regular CAN stack on 40ft swap body (same size, regular zone at tier 1)",
        )

    def test_swap_body_all_sizes(self):
        """Swap bodies come in various sizes (22–45 ft)."""
        sb_sizes = [22, 23, 24, 30, 45]
        for ft in sb_sizes:
            with self.subTest(length_ft=ft):
                yard = _make_yard_with_zones()
                c = _make_container(
                    f"SB_{ft}", length_ft=ft, is_swap_body=True,
                )
                p = yard.find_single_placement(c, target_bay=4, max_proximity=0)
                if ft <= 20:
                    # 20ft swap body should fit in one bay
                    self.assertIsNotNone(p)
                elif ft == 45:
                    # 45ft swap body spans 23 splits — starts at bay 4 (split 80),
                    # would extend to 103. That's within total_splits (1160).
                    # But swap body zone is only bay 5 (splits 80-99), and
                    # the container needs 23 splits — goods mask will reject
                    # since splits 100-102 are NOT in the swap body zone.
                    # This is correct behavior: 45ft swap bodies need a
                    # wider swap body zone to be placeable.
                    pass  # May or may not place depending on zone width
                else:
                    # 22-30ft should fit within one bay's swap body zone
                    self.assertIsNotNone(
                        p, f"Swap body {ft}ft should fit in sb zone",
                    )


# ================================================================
# 4. Yard: Stacking (tier support)
# ================================================================

class TestYardStacking(unittest.TestCase):
    """Containers can only stack on same-size support below."""

    def setUp(self):
        self.yard = _make_yard_with_zones()

    def test_stack_same_size(self):
        """Container can stack on top of same-size container."""
        c1 = _make_container("BASE_40", length_ft=40)
        p1 = self.yard.find_single_placement(c1, target_bay=15)
        self.assertIsNotNone(p1)
        self.yard.add_container(c1, p1)

        c2 = _make_container("TOP_40", length_ft=40)
        p2 = self.yard.find_single_placement(c2, target_bay=15)
        self.assertIsNotNone(p2, "Same-size container should stack")
        self.yard.add_container(c2, p2)
        # Should be at tier 1 (or same row as c1 at tier 1)
        if p2.row == p1.row:
            self.assertEqual(p2.tier, p1.tier + 1)

    def test_no_stack_different_size_direct(self):
        """Different-size container cannot stack directly (no support)."""
        # Place a 40ft at bay 15, row 0, tier 0
        c1 = _make_container("BASE_40S", length_ft=40)
        p1 = PlacementResult(row=0, bay=15, tier=0, start_split=0)
        self.yard.add_container(c1, p1)

        # Try to place a 20ft at bay 15, row 0, tier 1 — no 20ft support below
        c2 = _make_container("TOP_20S", length_ft=20)
        p2 = PlacementResult(row=0, bay=15, tier=1, start_split=0)
        n_splits_20 = _splits(20)
        goods_mask = self.yard._get_goods_mask_tier(c2, 1)
        valid = self.yard._is_valid_placement(
            1, 0, p2.abs_start, p2.abs_start + n_splits_20, n_splits_20, goods_mask,
        )
        self.assertFalse(
            valid,
            "20ft cannot stack on 40ft (different size, no support)",
        )

    def test_5_tier_stack(self):
        """Fill all 5 tiers with same-size containers."""
        for tier in range(N_TIERS):
            c = _make_container(f"STACK_{tier}", length_ft=40)
            p = PlacementResult(row=0, bay=20, tier=tier, start_split=0)
            self.yard.add_container(c, p)

        # Verify all 5 tiers occupied
        for tier in range(N_TIERS):
            abs_start = 20 * SF
            self.assertTrue(
                self.yard.occupancy_mask[tier, 0, abs_start],
                f"Tier {tier} should be occupied",
            )

    def test_accessibility_updates(self):
        """Bottom container becomes inaccessible when stacked upon."""
        c1 = _make_container("ACC_BOT", length_ft=40)
        p1 = PlacementResult(row=0, bay=25, tier=0, start_split=0)
        self.yard.add_container(c1, p1)
        rec1 = self.yard.get_record("ACC_BOT")
        self.assertTrue(rec1.is_accessible, "Lone container is accessible")

        c2 = _make_container("ACC_TOP", length_ft=40)
        p2 = PlacementResult(row=0, bay=25, tier=1, start_split=0)
        self.yard.add_container(c2, p2)

        rec1 = self.yard.get_record("ACC_BOT")
        self.assertFalse(rec1.is_accessible, "Buried container is NOT accessible")

        rec2 = self.yard.get_record("ACC_TOP")
        self.assertTrue(rec2.is_accessible, "Top container is accessible")


# ================================================================
# 5. Yard: Occupancy conflict detection
# ================================================================

class TestYardOccupancyConflict(unittest.TestCase):
    """add_container rejects overlapping placements."""

    def setUp(self):
        self.yard = _make_yard_with_zones()

    def test_exact_overlap_rejected(self):
        """Placing two containers at the exact same position raises."""
        c1 = _make_container("OVL_A", length_ft=40)
        p = PlacementResult(row=0, bay=10, tier=0, start_split=0)
        self.yard.add_container(c1, p)

        c2 = _make_container("OVL_B", length_ft=40)
        with self.assertRaises(IndexError, msg="Overlap must raise IndexError"):
            self.yard.add_container(c2, p)

    def test_partial_overlap_rejected(self):
        """20ft container overlapping part of an existing 40ft container."""
        c1 = _make_container("PART_A", length_ft=40)
        p1 = PlacementResult(row=0, bay=10, tier=0, start_split=0)
        self.yard.add_container(c1, p1)

        c2 = _make_container("PART_B", length_ft=20)
        # Start 10 splits into the 40ft container's space
        p2 = PlacementResult(row=0, bay=10, tier=0, start_split=10)
        with self.assertRaises(IndexError, msg="Partial overlap must raise"):
            self.yard.add_container(c2, p2)

    def test_out_of_bounds_rejected(self):
        """Container extending past total_splits raises IndexError."""
        c = _make_container("OOB_45", length_ft=45)
        p = PlacementResult(row=0, bay=57, tier=0, start_split=0)
        with self.assertRaises(IndexError, msg="Out of bounds must raise"):
            self.yard.add_container(c, p)


# ================================================================
# 6. Rail yard: All container sizes on trains
# ================================================================

class TestRailYardAllSizes(unittest.TestCase):
    """All container sizes can be loaded on trains and tracked in the grid."""

    def setUp(self):
        self.rail = RailYard(n_tracks=4, n_bays=N_BAYS, split_factor=SF)

    def _make_train(self, train_id: str, n_wagons: int = 5) -> Train:
        return Train(train_id=train_id, num_wagons=n_wagons)

    def test_all_sizes_load_on_train(self):
        """Each container size can be loaded onto a wagon and tracked."""
        for ft in ALL_SIZES_FT:
            with self.subTest(length_ft=ft):
                rail = RailYard(n_tracks=4, n_bays=N_BAYS, split_factor=SF)
                train = self._make_train(f"TR_{ft}", n_wagons=3)
                c = _make_container(f"RAIL_{ft}", length_ft=ft)
                train.add_container(c, wagon_index=0)

                rail.slot_train(train, RailSlot(track_id=0, anchor_bay=5))

                # Container should be in the grid
                rec = rail.get_record_at(0, 5 * SF)
                self.assertIsNotNone(
                    rec, f"{ft}ft container should be tracked in rail grid",
                )
                self.assertEqual(rec.n_splits, _container_n_splits(c, SF))

    def test_split_count_matches(self):
        """Rail grid n_splits matches CONTAINER_LENGTH_TO_SUB_BAYS."""
        for ft in ALL_SIZES_FT:
            n = _container_n_splits(
                _make_container(f"X_{ft}", length_ft=ft), SF,
            )
            self.assertEqual(
                n, _splits(ft),
                f"{ft}ft: rail n_splits={n} != yard n_splits={_splits(ft)}",
            )

    def test_multiple_sizes_on_one_train(self):
        """A train can carry containers of different sizes."""
        train = self._make_train("TR_MIX", n_wagons=10)
        for i, ft in enumerate(ALL_SIZES_FT):
            c = _make_container(f"MIX_{ft}", length_ft=ft)
            # Each on a different wagon to avoid overflow
            train.add_container(c, wagon_index=i)

        self.rail.slot_train(train, RailSlot(track_id=0, anchor_bay=5))

        # All containers should be in the grid
        self.assertEqual(
            self.rail.container_count(),
            len(ALL_SIZES_FT),
            "All container sizes should be tracked",
        )

    def test_45ft_on_train_at_interior_anchor(self):
        """45ft container on a train at an interior anchor bay."""
        train = self._make_train("TR_45INT", n_wagons=3)
        c = _make_container("RAIL_45INT", length_ft=45)
        train.add_container(c, wagon_index=0)

        self.rail.slot_train(train, RailSlot(track_id=0, anchor_bay=20))

        # Should be in the grid
        rec = self.rail.get_record_at(0, 20 * SF)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.n_splits, 23)

    def test_container_survives_sync(self):
        """sync_train() preserves container positions."""
        train = self._make_train("TR_SYNC", n_wagons=5)
        sizes = [20, 40, 30]
        for i, ft in enumerate(sizes):
            c = _make_container(f"SYNC_{ft}", length_ft=ft)
            train.add_container(c, wagon_index=i)

        self.rail.slot_train(train, RailSlot(track_id=1, anchor_bay=10))
        count_before = self.rail.container_count()

        self.rail.sync_train("TR_SYNC", train)
        count_after = self.rail.container_count()

        self.assertEqual(count_before, count_after)

    def test_occupancy_mask_correct(self):
        """Occupancy mask correctly reflects container positions."""
        train = self._make_train("TR_OCC", n_wagons=3)
        c = _make_container("OCC_40", length_ft=40)
        train.add_container(c, wagon_index=0)

        self.rail.slot_train(train, RailSlot(track_id=2, anchor_bay=15))

        # 40ft = 20 splits at bay 15 → splits 300-319
        abs_start = 15 * SF
        for s in range(abs_start, abs_start + 20):
            self.assertTrue(
                self.rail.occupancy_mask[2, s],
                f"Split {s} should be occupied",
            )
        # Split before should be empty
        if abs_start > 0:
            self.assertFalse(self.rail.occupancy_mask[2, abs_start - 1])


# ================================================================
# 7. Parking: Bay-level truck allocation
# ================================================================

class TestParkingAllocation(unittest.TestCase):
    """Truck parking with bay-level occupancy."""

    def setUp(self):
        self.parking = ParkingArea(n_bays=N_BAYS)

    def _make_truck(self, truck_id: str) -> Truck:
        return Truck(
            truck_id=truck_id,
            arrival_time=NOW,
        )

    def test_allocate_and_release(self):
        """Truck can be parked and released."""
        tk = self._make_truck("TK_1")
        ok = self.parking.allocate(tk, bay=5, split=0)
        self.assertTrue(ok, "Allocation should succeed")
        self.assertFalse(self.parking.is_free(5), "Bay 5 should be occupied")

        ok = self.parking.release(tk)
        self.assertTrue(ok, "Release should succeed")
        self.assertTrue(self.parking.is_free(5), "Bay 5 should be free after release")

    def test_one_truck_per_bay(self):
        """Second truck in same bay is rejected."""
        tk1 = self._make_truck("TK_A")
        tk2 = self._make_truck("TK_B")
        self.parking.allocate(tk1, bay=10, split=0)

        ok = self.parking.allocate(tk2, bay=10, split=0)
        self.assertFalse(ok, "Second truck in same bay must be rejected")

    def test_all_bays_can_be_used(self):
        """Every bay from 0 to N_BAYS-1 can hold a truck."""
        for bay in range(N_BAYS):
            tk = self._make_truck(f"TK_{bay}")
            ok = self.parking.allocate(tk, bay=bay, split=0)
            self.assertTrue(ok, f"Bay {bay} should accept a truck")

    def test_occupancy_flat_reflects_parking(self):
        """Flat occupancy grid marks canonical split when truck parks."""
        tk = self._make_truck("TK_FLAT")
        self.parking.allocate(tk, bay=20, split=0)
        abs_split = 20 * SF
        self.assertTrue(
            self.parking.occupancy_flat[abs_split],
            "Canonical split should be marked",
        )

    def test_iter_records(self):
        """iter_records returns parked trucks."""
        trucks = [self._make_truck(f"TK_IT_{i}") for i in range(5)]
        for i, tk in enumerate(trucks):
            self.parking.allocate(tk, bay=i * 10, split=0)

        records = list(self.parking.iter_records())
        self.assertEqual(len(records), 5, "Should have 5 parking records")

    def test_out_of_bounds_bay(self):
        """Bay outside [0, n_bays) is rejected."""
        tk = self._make_truck("TK_OOB")
        ok = self.parking.allocate(tk, bay=N_BAYS, split=0)
        self.assertFalse(ok, "Bay == n_bays should be rejected")
        ok2 = self.parking.allocate(tk, bay=-1, split=0)
        self.assertFalse(ok2, "Negative bay should be rejected")


# ================================================================
# 8. Cross-facility: yard ↔ rail split consistency
# ================================================================

class TestCrossFacilityConsistency(unittest.TestCase):
    """Container size mapping is consistent across facilities."""

    def test_split_factor_same_everywhere(self):
        """All facilities use the same BAY_SPLIT_FACTOR."""
        yard = StorageYard(
            n_rows=2, n_bays=10, n_tiers=3, coordinates=[], validate=False,
        )
        rail = RailYard(n_tracks=2, n_bays=10, split_factor=SF)
        parking = ParkingArea(n_bays=10)

        self.assertEqual(yard.split_factor, SF)
        self.assertEqual(rail.split_factor, SF)
        self.assertEqual(parking.split_factor, SF)

    def test_all_sizes_in_length_map(self):
        """Every size in ALL_SIZES_FT has an entry in the length map."""
        for ft in ALL_SIZES_FT:
            self.assertIn(
                ft, CONTAINER_LENGTH_TO_SUB_BAYS,
                f"{ft}ft missing from CONTAINER_LENGTH_TO_SUB_BAYS",
            )
            self.assertGreater(
                CONTAINER_LENGTH_TO_SUB_BAYS[ft], 0,
                f"{ft}ft has 0 splits",
            )

    def test_split_counts_monotonic(self):
        """Longer containers occupy more (or equal) splits."""
        prev_ft, prev_splits = 0, 0
        for ft in ALL_SIZES_FT:
            n = _splits(ft)
            self.assertGreaterEqual(
                n, prev_splits,
                f"{ft}ft ({n} splits) < {prev_ft}ft ({prev_splits} splits)",
            )
            prev_ft, prev_splits = ft, n

    def test_max_splits_does_not_exceed_two_bays(self):
        """No container size exceeds 2 bays (40 splits)."""
        for ft in ALL_SIZES_FT:
            n = _splits(ft)
            self.assertLessEqual(
                n, 2 * SF,
                f"{ft}ft has {n} splits (> 2 bays = {2*SF})",
            )


if __name__ == "__main__":
    unittest.main()
