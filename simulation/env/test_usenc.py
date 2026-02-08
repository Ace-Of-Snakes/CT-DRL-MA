"""Tests for UnifiedStateEncoder — Phase 1 validation.

Verifies:
  - Tensor shape (C, R_uni, S, T) = (10, 15, 1160, 5)
  - Rail container positions (track row, wagon bay, tier 0)
  - Parking truck positions (parking row, split position, tier 0)
  - Yard container positions (shifted by yard_row_start)
  - Queue truck positions (preferred bay, tier 0)
  - Source mask correctness
  - Region boundary properties
"""
import sys
import os
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Iterator
from collections import OrderedDict
from unittest.mock import MagicMock

# ── Add project to path for imports ──────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock torch (not available in test container)
import types
torch_mock = types.ModuleType("torch")
torch_mock.cuda = MagicMock()
torch_mock.cuda.is_available = MagicMock(return_value=False)
sys.modules["torch"] = torch_mock

from simulation.rl.multihead_dqn.config import UnifiedDims
from simulation.core.enums import TruckStatus, Region
from simulation.env.unified_state_encoder import (
    UnifiedStateEncoder, UnifiedChannelSpec, CH,
    _container_type_value, _container_hash, _direction_value,
)


# ══════════════════════════════════════════════════════════════════════════
# Mock objects
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class MockContainer:
    """Minimal container for testing."""
    container_id: str
    length_ft: float = 20.0
    goods_type: str = "Regular"
    direction: Optional[str] = None
    departure_date: Optional[datetime] = None
    is_swap_body: bool = False
    is_trailer: bool = False


@dataclass
class MockPlacement:
    """Placement result mock."""
    row: int
    bay: int
    tier: int
    start_split: int

    @property
    def abs_start(self) -> int:
        return self.bay * 20 + self.start_split


@dataclass
class MockRecord:
    """Container record mock."""
    container: MockContainer
    placement: MockPlacement
    n_splits: int
    idx: int
    is_accessible: bool = True


@dataclass
class MockParkingSpot:
    """Parking spot mock."""
    bay: int
    split: int


@dataclass
class MockRailSlot:
    """Rail slot mock."""
    track_id: int
    anchor_bay: int


class MockWagon:
    """Wagon mock with OrderedDict containers."""
    def __init__(self, wagon_id: str, containers: Optional[List[MockContainer]] = None):
        self.wagon_id = wagon_id
        self.containers = OrderedDict()
        if containers:
            for c in containers:
                self.containers[c.container_id] = c


class MockTrain:
    """Train mock with wagons and pickup tracking."""
    def __init__(
        self,
        train_id: str,
        wagons: Optional[List[MockWagon]] = None,
        pickup_ids: Optional[Set[str]] = None,
    ):
        self.train_id = train_id
        self.wagons = wagons or []
        self._pickup_ids = pickup_ids or set()
        self.container_locations: Dict[str, object] = {}

    def get_all_pickup_container_ids(self) -> Set[str]:
        return self._pickup_ids


class MockTruck:
    """Truck mock."""
    def __init__(
        self,
        truck_id: str,
        parking_spot=None,
        containers: Optional[List[MockContainer]] = None,
        pickup_ids: Optional[Set[str]] = None,
        arrival_time: Optional[datetime] = None,
        status=TruckStatus.WAITING,
    ):
        self.truck_id = truck_id
        self.parking_spot = parking_spot
        self.containers = containers or []
        self.pickup_container_ids = pickup_ids or set()
        self.arrival_time = arrival_time
        self.status = status


class MockYard:
    """Minimal yard mock."""
    def __init__(self, n_rows=5, n_bays=58, n_tiers=5, split_factor=20):
        self.n_rows = n_rows
        self.n_bays = n_bays
        self.n_tiers = n_tiers
        self.split_factor = split_factor
        self.total_splits = n_bays * split_factor
        self.occupancy_mask = np.zeros((n_tiers, n_rows, self.total_splits), dtype=bool)
        self._records: List[MockRecord] = []
        self._placements: Dict[str, MockPlacement] = {}

    def iter_records(self) -> Iterator[MockRecord]:
        return iter(self._records)

    def get_placement(self, container_id: str):
        return self._placements.get(container_id)

    def add_container(self, container: MockContainer, row: int, bay: int,
                      tier: int, start_split: int = 0, n_splits: int = 10):
        """Helper to add a container to the mock yard."""
        pl = MockPlacement(row=row, bay=bay, tier=tier, start_split=start_split)
        rec = MockRecord(
            container=container, placement=pl,
            n_splits=n_splits, idx=len(self._records), is_accessible=True,
        )
        self._records.append(rec)
        self._placements[container.container_id] = pl
        # Update occupancy mask
        s0 = pl.abs_start
        s1 = min(s0 + n_splits, self.total_splits)
        self.occupancy_mask[tier, row, s0:s1] = True


class MockRail:
    """Minimal rail yard mock."""
    def __init__(self):
        self._train_to_slot: Dict[str, MockRailSlot] = {}

    def get_slot(self, train_id: str):
        return self._train_to_slot.get(train_id)

    def get_anchor_bay(self, train_id: str):
        slot = self._train_to_slot.get(train_id)
        return slot.anchor_bay if slot else None

    def slot_train(self, train_id: str, slot: MockRailSlot):
        self._train_to_slot[train_id] = slot


# ══════════════════════════════════════════════════════════════════════════
# Test helpers
# ══════════════════════════════════════════════════════════════════════════

NOW = datetime(2025, 6, 15, 10, 0, 0)
DIMS = UnifiedDims()  # defaults: 7 tracks, 1 park, 5 yard, 2 queue


def _make_encoder(yard=None, rail=None, parking=None):
    """Create encoder with defaults."""
    y = yard or MockYard()
    r = rail or MockRail()
    return UnifiedStateEncoder(yard=y, rail=r, parking=parking, dims=DIMS)


# ══════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════

class TestUnifiedDims:
    """Test UnifiedDims region boundaries."""

    def test_default_dimensions(self):
        d = UnifiedDims()
        assert d.R_unified == 15
        assert d.n_splits == 1160
        assert d.spatial_shape == (15, 1160, 5)

    def test_rail_rows(self):
        d = UnifiedDims()
        assert d.rail_row_start == 0
        assert d.rail_row_end == 7
        assert list(d.rail_rows) == [0, 1, 2, 3, 4, 5, 6]

    def test_parking_rows(self):
        d = UnifiedDims()
        assert d.parking_row_start == 7
        assert d.parking_row_end == 8
        assert list(d.parking_rows) == [7]

    def test_yard_rows(self):
        d = UnifiedDims()
        assert d.yard_row_start == 8
        assert d.yard_row_end == 13
        assert list(d.yard_rows) == [8, 9, 10, 11, 12]

    def test_queue_rows(self):
        d = UnifiedDims()
        assert d.queue_row_start == 13
        assert d.queue_row_end == 15
        assert list(d.queue_rows) == [13, 14]

    def test_region_of(self):
        d = UnifiedDims()
        assert d.region_of(0) == "RAIL"
        assert d.region_of(6) == "RAIL"
        assert d.region_of(7) == "PARKING"
        assert d.region_of(8) == "YARD"
        assert d.region_of(12) == "YARD"
        assert d.region_of(13) == "QUEUE"
        assert d.region_of(14) == "QUEUE"

    def test_custom_dims(self):
        d = UnifiedDims(n_tracks=3, n_yard_rows=2, n_queue_rows=1)
        assert d.R_unified == 7  # 3 + 1 + 2 + 1
        assert d.yard_row_start == 4  # 3 + 1
        assert d.queue_row_start == 6  # 3 + 1 + 2


class TestTensorShape:
    """Test output tensor dimensions."""

    def test_empty_state_shape(self):
        enc = _make_encoder()
        tensor = enc.encode({}, {}, NOW)
        assert tensor.shape == (10, 15, 1160, 5)
        assert tensor.dtype == np.float32

    def test_empty_state_all_zeros(self):
        enc = _make_encoder()
        tensor = enc.encode({}, {}, NOW)
        assert np.allclose(tensor, 0.0)


class TestRailEncoding:
    """Test rail row encoding of train containers."""

    def test_single_container_on_track(self):
        yard = MockYard()
        rail = MockRail()

        c = MockContainer("C1", length_ft=20.0, direction="Import")
        w0 = MockWagon("W0", [c])
        train = MockTrain("T1", wagons=[w0])
        rail.slot_train("T1", MockRailSlot(track_id=2, anchor_bay=10))

        enc = _make_encoder(yard, rail)
        tensor = enc.encode({"T1": train}, {}, NOW)

        # Should appear at row=2, split=10*20=200, tier=0
        row = 2
        s = 10 * 20  # anchor_bay * split_factor
        assert tensor[CH.OCCUPANCY, row, s, 0] == 1.0
        assert tensor[CH.CONTAINER_START, row, s, 0] == 1.0
        assert tensor[CH.DIRECTION, row, s, 0] == 0.0  # Import
        assert tensor[CH.CONTAINER_HASH, row, s, 0] > 0.0

    def test_multiple_wagons(self):
        yard = MockYard()
        rail = MockRail()

        c1 = MockContainer("C1", length_ft=20.0, direction="Import")
        c2 = MockContainer("C2", length_ft=20.0, direction="Import")
        w0 = MockWagon("W0", [c1])
        w1 = MockWagon("W1", [c2])
        train = MockTrain("T1", wagons=[w0, w1])
        rail.slot_train("T1", MockRailSlot(track_id=0, anchor_bay=5))

        enc = _make_encoder(yard, rail)
        tensor = enc.encode({"T1": train}, {}, NOW)

        # Wagon 0 at bay 5, wagon 1 at bay 6
        assert tensor[CH.OCCUPANCY, 0, 5 * 20, 0] == 1.0
        assert tensor[CH.OCCUPANCY, 0, 6 * 20, 0] == 1.0

    def test_export_container_direction(self):
        yard = MockYard()
        rail = MockRail()

        c = MockContainer("C_exp", length_ft=20.0, direction="Export")
        w0 = MockWagon("W0", [c])
        # Train wants to pick up C_exp (it's already loaded)
        train = MockTrain("T1", wagons=[w0], pickup_ids=set())
        # Container is ON the train, not in pickup_ids → it's import
        rail.slot_train("T1", MockRailSlot(track_id=0, anchor_bay=0))

        enc = _make_encoder(yard, rail)
        tensor = enc.encode({"T1": train}, {}, NOW)

        # Import (on train, not in pickup set)
        assert tensor[CH.DIRECTION, 0, 0, 0] == 0.0

    def test_only_tier_0_used(self):
        """Rail containers should only use tier 0."""
        yard = MockYard()
        rail = MockRail()

        c = MockContainer("C1")
        w0 = MockWagon("W0", [c])
        train = MockTrain("T1", wagons=[w0])
        rail.slot_train("T1", MockRailSlot(track_id=0, anchor_bay=5))

        enc = _make_encoder(yard, rail)
        tensor = enc.encode({"T1": train}, {}, NOW)

        # Tier 0 has data
        assert tensor[CH.OCCUPANCY, 0, 5 * 20, 0] == 1.0
        # Higher tiers should be empty
        for t in range(1, 5):
            assert tensor[CH.OCCUPANCY, 0, 5 * 20, t] == 0.0


class TestParkingEncoding:
    """Test parking row encoding of parked trucks."""

    def test_parked_pickup_truck(self):
        enc = _make_encoder()
        spot = MockParkingSpot(bay=15, split=3)
        tk = MockTruck(
            "TK1", parking_spot=spot,
            pickup_ids={"C99"}, status=TruckStatus.LOADING,
            arrival_time=NOW - timedelta(hours=1),
        )

        tensor = enc.encode({}, {"TK1": tk}, NOW)
        pr = DIMS.parking_row_start  # 7
        s = 15 * 20 + 3

        assert tensor[CH.OCCUPANCY, pr, s, 0] == 1.0
        assert tensor[CH.DIRECTION, pr, s, 0] == 0.0  # pickup = Import
        assert tensor[CH.TRUCK_DEMAND, pr, s, 0] == 1.0
        assert tensor[CH.DEPARTURE_URGENCY, pr, s, 0] > 0.0

    def test_parked_delivery_truck(self):
        enc = _make_encoder()
        c = MockContainer("C_dlv", direction="Export")
        spot = MockParkingSpot(bay=20, split=0)
        tk = MockTruck(
            "TK2", parking_spot=spot,
            containers=[c], status=TruckStatus.LOADING,
        )

        tensor = enc.encode({}, {"TK2": tk}, NOW)
        pr = DIMS.parking_row_start
        s = 20 * 20

        assert tensor[CH.OCCUPANCY, pr, s, 0] == 1.0
        assert tensor[CH.DIRECTION, pr, s, 0] == 1.0  # delivery = Export
        assert tensor[CH.CONTAINER_TYPE, pr, s, 0] > 0.0

    def test_unparked_truck_not_in_parking(self):
        """Trucks without parking spots shouldn't appear in parking row."""
        enc = _make_encoder()
        tk = MockTruck("TK3", parking_spot=None, status=TruckStatus.WAITING)

        tensor = enc.encode({}, {"TK3": tk}, NOW)
        pr = DIMS.parking_row_start
        assert np.all(tensor[CH.OCCUPANCY, pr, :, :] == 0.0)


class TestYardEncoding:
    """Test yard container encoding with row offset."""

    def test_container_at_correct_offset(self):
        yard = MockYard()
        c = MockContainer("CY1", direction="Export", departure_date=NOW + timedelta(days=5))
        yard.add_container(c, row=2, bay=10, tier=0, start_split=5, n_splits=10)

        enc = _make_encoder(yard)
        tensor = enc.encode({}, {}, NOW)

        # Yard row 2 → unified row = yard_row_start + 2 = 10
        r = DIMS.yard_row_start + 2
        s0 = 10 * 20 + 5

        assert tensor[CH.OCCUPANCY, r, s0, 0] == 1.0
        assert tensor[CH.CONTAINER_START, r, s0, 0] == 1.0
        assert tensor[CH.DIRECTION, r, s0, 0] == 1.0  # Export
        assert 0.0 < tensor[CH.DEPARTURE_URGENCY, r, s0, 0] < 1.0

    def test_stacked_containers(self):
        yard = MockYard()
        c0 = MockContainer("CY_T0")
        c1 = MockContainer("CY_T1")
        yard.add_container(c0, row=0, bay=5, tier=0, n_splits=10)
        yard.add_container(c1, row=0, bay=5, tier=1, n_splits=10)

        enc = _make_encoder(yard)
        tensor = enc.encode({}, {}, NOW)

        r = DIMS.yard_row_start  # row 0 → unified row 8
        s = 5 * 20
        assert tensor[CH.OCCUPANCY, r, s, 0] == 1.0
        assert tensor[CH.OCCUPANCY, r, s, 1] == 1.0

    def test_yard_demand_channels(self):
        """Truck and train demand should appear on yard containers."""
        yard = MockYard()
        rail = MockRail()

        c = MockContainer("C_wanted", direction="Export")
        yard.add_container(c, row=0, bay=10, tier=0, n_splits=10)

        # Train that wants this container
        train = MockTrain("T1", wagons=[], pickup_ids={"C_wanted"})
        rail.slot_train("T1", MockRailSlot(track_id=0, anchor_bay=10))

        # Truck that wants this container
        spot = MockParkingSpot(bay=10, split=0)
        tk = MockTruck("TK1", parking_spot=spot, pickup_ids={"C_wanted"})

        enc = _make_encoder(yard, rail)
        tensor = enc.encode({"T1": train}, {"TK1": tk}, NOW)

        r = DIMS.yard_row_start
        s = 10 * 20
        assert tensor[CH.TRUCK_DEMAND, r, s, 0] == 1.0
        assert tensor[CH.TRAIN_DEMAND, r, s, 0] > 0.0


class TestQueueEncoding:
    """Test queue row encoding of waiting trucks."""

    def test_queued_pickup_truck_at_preferred_bay(self):
        yard = MockYard()
        c = MockContainer("C_target")
        yard.add_container(c, row=1, bay=25, tier=0, n_splits=10)

        # Pickup truck wants C_target, which is at bay 25
        tk = MockTruck(
            "TK_Q1", parking_spot=None,
            pickup_ids={"C_target"}, status=TruckStatus.WAITING,
            arrival_time=NOW - timedelta(minutes=30),
        )

        enc = _make_encoder(yard)
        tensor = enc.encode({}, {"TK_Q1": tk}, NOW)

        # Should be in queue row 0 at bay 25
        qr = DIMS.queue_row_start
        s = 25 * 20
        assert tensor[CH.OCCUPANCY, qr, s, 0] == 1.0
        assert tensor[CH.DIRECTION, qr, s, 0] == 0.0  # pickup = Import
        assert tensor[CH.TRUCK_DEMAND, qr, s, 0] == 1.0

    def test_queued_delivery_truck_at_goods_anchor(self):
        yard = MockYard()
        c = MockContainer("C_dlv", goods_type="Reefer")
        tk = MockTruck(
            "TK_Q2", parking_spot=None,
            containers=[c], status=TruckStatus.WAITING,
        )

        enc = _make_encoder(yard)
        tensor = enc.encode({}, {"TK_Q2": tk}, NOW)

        # Reefer anchor is bay 0
        qr = DIMS.queue_row_start
        s = 0  # bay 0 * 20
        assert tensor[CH.OCCUPANCY, qr, s, 0] == 1.0
        assert tensor[CH.DIRECTION, qr, s, 0] == 1.0  # delivery = Export

    def test_two_trucks_stack_to_queue_row_1(self):
        yard = MockYard()
        c1 = MockContainer("C_t1")
        c2 = MockContainer("C_t2")
        yard.add_container(c1, row=0, bay=20, tier=0, n_splits=10)
        yard.add_container(c2, row=0, bay=20, tier=1, n_splits=10)

        tk1 = MockTruck("TK1", pickup_ids={"C_t1"}, status=TruckStatus.WAITING)
        tk2 = MockTruck("TK2", pickup_ids={"C_t2"}, status=TruckStatus.WAITING)

        enc = _make_encoder(yard)
        tensor = enc.encode({}, {"TK1": tk1, "TK2": tk2}, NOW)

        # Both target bay 20, should stack across queue rows
        qr0 = DIMS.queue_row_start
        qr1 = DIMS.queue_row_start + 1
        s = 20 * 20

        occ_q0 = tensor[CH.OCCUPANCY, qr0, s, 0]
        occ_q1 = tensor[CH.OCCUPANCY, qr1, s, 0]
        assert occ_q0 == 1.0
        assert occ_q1 == 1.0

    def test_parked_truck_not_in_queue(self):
        """Parked trucks should NOT appear in queue rows."""
        yard = MockYard()
        spot = MockParkingSpot(bay=10, split=0)
        tk = MockTruck("TK_P", parking_spot=spot, status=TruckStatus.LOADING)

        enc = _make_encoder(yard)
        tensor = enc.encode({}, {"TK_P": tk}, NOW)

        # Queue rows should be empty
        for qr in DIMS.queue_rows:
            assert np.all(tensor[CH.OCCUPANCY, qr, :, :] == 0.0)

    def test_fallback_to_center_bay(self):
        """Truck with no containers or pickup IDs should go to center."""
        yard = MockYard()
        tk = MockTruck("TK_empty", status=TruckStatus.WAITING)

        enc = _make_encoder(yard)
        tensor = enc.encode({}, {"TK_empty": tk}, NOW)

        # Fallback bay = n_bays // 2 = 29
        qr = DIMS.queue_row_start
        s = 29 * 20
        assert tensor[CH.OCCUPANCY, qr, s, 0] == 1.0


class TestSourceMask:
    """Test unified source mask generation."""

    def test_accessible_yard_container(self):
        yard = MockYard()
        c = MockContainer("CY1")
        yard.add_container(c, row=0, bay=5, tier=0, n_splits=10)

        enc = _make_encoder(yard)
        mask = enc.get_source_mask({}, {})

        r = DIMS.yard_row_start
        s = 5 * 20
        assert mask[r, s, 0] is True or mask[r, s, 0] == True

    def test_rail_import_in_source(self):
        yard = MockYard()
        rail = MockRail()

        c = MockContainer("C_imp")
        w = MockWagon("W0", [c])
        train = MockTrain("T1", wagons=[w])
        rail.slot_train("T1", MockRailSlot(track_id=1, anchor_bay=10))

        enc = _make_encoder(yard, rail)
        mask = enc.get_source_mask({"T1": train}, {})

        assert mask[1, 10 * 20, 0] == True

    def test_delivery_truck_in_source(self):
        yard = MockYard()
        c = MockContainer("C_dlv")
        spot = MockParkingSpot(bay=15, split=2)
        tk = MockTruck("TK1", parking_spot=spot, containers=[c])

        enc = _make_encoder(yard)
        mask = enc.get_source_mask({}, {"TK1": tk})

        pr = DIMS.parking_row_start
        s = 15 * 20 + 2
        assert mask[pr, s, 0] == True

    def test_queued_truck_in_source(self):
        yard = MockYard()
        c = MockContainer("C_t")
        yard.add_container(c, row=0, bay=30, tier=0, n_splits=10)

        tk = MockTruck("TK_Q", pickup_ids={"C_t"}, status=TruckStatus.WAITING)

        enc = _make_encoder(yard)
        mask = enc.get_source_mask({}, {"TK_Q": tk})

        # Should be in queue rows at bay 30
        qr = DIMS.queue_row_start
        s = 30 * 20
        assert mask[qr, s, 0] == True

    def test_empty_state_no_sources(self):
        enc = _make_encoder()
        mask = enc.get_source_mask({}, {})
        assert not np.any(mask)


class TestYardValidity:
    """Test yard validity mask with unified padding."""

    def test_validity_shape(self):
        enc = _make_encoder()
        mask = enc.get_yard_validity_mask(n_splits_needed=10)
        assert mask.shape == (15, 1160, 5)

    def test_only_yard_rows_can_be_valid(self):
        enc = _make_encoder()
        mask = enc.get_yard_validity_mask(n_splits_needed=10)

        # Non-yard rows should all be False
        for r in range(DIMS.R_unified):
            if r not in DIMS.yard_rows:
                assert not np.any(mask[r]), f"Non-yard row {r} has True values"

    def test_empty_yard_all_tier0_valid(self):
        enc = _make_encoder()
        mask = enc.get_yard_validity_mask(n_splits_needed=10)

        # Tier 0 in yard rows should have valid positions in an empty yard
        for r in DIMS.yard_rows:
            assert np.any(mask[r, :, 0]), f"Yard row {r} has no valid positions"


# ══════════════════════════════════════════════════════════════════════════
# Run tests
# ══════════════════════════════════════════════════════════════════════════

def run_all():
    """Run all tests and report results."""
    test_classes = [
        TestUnifiedDims,
        TestTensorShape,
        TestRailEncoding,
        TestParkingEncoding,
        TestYardEncoding,
        TestQueueEncoding,
        TestSourceMask,
        TestYardValidity,
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in sorted(methods):
            total += 1
            try:
                getattr(instance, method_name)()
                passed += 1
                print(f"  ✓ {cls.__name__}.{method_name}")
            except Exception as e:
                failed += 1
                errors.append((cls.__name__, method_name, e))
                print(f"  ✗ {cls.__name__}.{method_name}: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")

    if errors:
        print(f"\nFailures:")
        for cls_name, method, err in errors:
            print(f"  {cls_name}.{method}:")
            import traceback
            traceback.print_exception(type(err), err, err.__traceback__)
    else:
        print("All tests passed! ✓")

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)