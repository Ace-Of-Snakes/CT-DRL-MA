# simulation/training/tutorial_scenarios.py
"""Tutorial scenarios for curriculum training.

Ten progressive micro-scenarios that teach the agent individual skills.
Every move is agent-controlled — NO auto-transfer exists.

Progression:
  S1-S3: Single-action primitives (park, import, export)
  S4-S6: Two-action chains (import→export, park→import, park→export)
  S7-S8: Three-action chains (park→import→export)
  S9-S10: Restack under distractors (selective multi-step)
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

from simulation.core.containers.container import Container
from simulation.core.enums import Direction, GoodsType, TrainStatus, TruckStatus
from simulation.core.facilities.parking import ParkingSpot
from simulation.core.facilities.railyard import RailSlot
from simulation.core.facilities.yard import PlacementResult
from simulation.core.vehicles.train import Train
from simulation.core.vehicles.truck import Truck


# ================================================================
# Constants
# ================================================================

TUTORIAL_TIME = datetime(2026, 1, 15, 8, 0, 0)
DEFAULT_DEPARTURE = TUTORIAL_TIME + timedelta(days=5)
CONTAINER_TYPE_40HC = "40' HC"
DEFAULT_LENGTH_FT = 40
DEFAULT_LENGTH_M = 12.2
ANCHOR_BAY = 5
TRAIN_NUM_WAGONS = 5

_SPLIT_FACTOR: Optional[int] = None


def _get_split_factor() -> int:
    global _SPLIT_FACTOR
    if _SPLIT_FACTOR is None:
        from simulation.core.facilities.constants import BAY_SPLIT_FACTOR
        _SPLIT_FACTOR = BAY_SPLIT_FACTOR
    return _SPLIT_FACTOR


# ================================================================
# Result dataclass
# ================================================================

@dataclass
class TutorialResult:
    """Outcome of one tutorial attempt."""
    scenario_id: int
    name: str
    passed: bool
    steps: int
    agent_moves: int
    auto_moves: int
    total_reward: float
    move_log: List[Dict] = field(default_factory=list)

    def __str__(self) -> str:
        tag = "PASS" if self.passed else "FAIL"
        return (f"[{tag}] T{self.scenario_id}: {self.name} "
                f"| steps={self.steps} agent_mv={self.agent_moves} "
                f"auto_mv={self.auto_moves} R={self.total_reward:.2f}")


# ================================================================
# Entity helpers
# ================================================================

def _make_container(
    cid: str,
    direction: Direction = Direction.IMPORT,
    departure: Optional[datetime] = None,
) -> Container:
    """Create a standard 40ft container for tutorials."""
    return Container(
        container_id=cid,
        direction=direction,
        container_type=CONTAINER_TYPE_40HC,
        arrival_date=TUTORIAL_TIME,
        departure_date=departure or DEFAULT_DEPARTURE,
        goods_type=GoodsType.REGULAR,
        length_ft=DEFAULT_LENGTH_FT,
        length_m=DEFAULT_LENGTH_M,
    )


def _make_train(
    tid: str,
    containers: Optional[List[Container]] = None,
    pickup_ids: Optional[List[str]] = None,
    num_wagons: int = TRAIN_NUM_WAGONS,
) -> Train:
    """Create a train and optionally load containers / set pickup ids."""
    train = Train(
        train_id=tid,
        num_wagons=num_wagons,
        arrival_time=TUTORIAL_TIME,
        departure_time=TUTORIAL_TIME + timedelta(hours=8),
    )
    train.status = TrainStatus.WAITING
    for c in (containers or []):
        train.add_container(c)
    for pid in (pickup_ids or []):
        train.add_pickup_container(pid)
    return train


def _make_truck(
    tkid: str,
    containers: Optional[List[Container]] = None,
    pickup_ids: Optional[List[str]] = None,
) -> Truck:
    """Create a truck, optionally with cargo or pickup demand."""
    truck = Truck(truck_id=tkid, arrival_time=TUTORIAL_TIME)
    truck.status = TruckStatus.WAITING
    for c in (containers or []):
        truck.add_container(c)
    for pid in (pickup_ids or []):
        truck.add_pickup_container_id(pid)
    return truck


def _yard_placement(bay: int, row: int, tier: int, start_split: int = 0) -> PlacementResult:
    """Convenience wrapper for PlacementResult."""
    return PlacementResult(row=row, bay=bay, tier=tier, start_split=start_split)


def _place_distractors(env, rng: random.Random, count: int, prefix: str,
                       exclude_bays: Set[int]) -> None:
    """Place N distractor containers at random positions."""
    used: Set[Tuple[int, int]] = set()
    for i in range(count):
        cid = f"{prefix}_D{i}"
        c = _make_container(cid, Direction.IMPORT,
                            departure=TUTORIAL_TIME + timedelta(days=15))
        while True:
            bay = rng.randint(1, env.yard.n_bays - 2)
            row = rng.randint(0, env.yard.n_rows - 1)
            if (bay, row) not in used and bay not in exclude_bays:
                break
        used.add((bay, row))
        env.yard.add_container(c, _yard_placement(bay=bay, row=row, tier=0))


# ================================================================
# Base class
# ================================================================

class TutorialScenario(ABC):
    """Base class for tutorial scenarios."""

    id: int
    name: str
    description: str
    max_steps: int
    expected_moves: Optional[int] = None

    @abstractmethod
    def setup(self, env) -> None:
        """Inject entities into a freshly-reset tutorial env."""

    @abstractmethod
    def check_success(self, env) -> bool:
        """Return True when the tutorial goal has been achieved."""

    def check_pass(self, env, agent_moves: int, auto_moves: int) -> bool:
        """Full pass check: success AND optional move budget."""
        if not self.check_success(env):
            return False
        if self.expected_moves is not None:
            total = agent_moves + auto_moves
            if total > self.expected_moves:
                return False
        return True


# ================================================================
# S1: Park Truck  (SLOT_PARKING)
# ================================================================

class S1_ParkTruck(TutorialScenario):
    """One unparked truck. Agent must park it.

    Teaches: SLOT_PARKING action type.
    Success: truck has a parking_spot.
    """
    id = 1
    name = "park_truck"
    description = "Park an unparked truck"
    max_steps = 15
    expected_moves = None

    def setup(self, env) -> None:
        c = _make_container("S1_C1", Direction.IMPORT)
        env.yard.add_container(c, _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))

        tk = _make_truck("S1_TK1", pickup_ids=[c.container_id])
        env.trucks[tk.truck_id] = tk
        # Truck is NOT parked — agent must use SLOT_PARKING

    def check_success(self, env) -> bool:
        truck = env.trucks.get("S1_TK1")
        if truck is None:
            return True  # departed = full success
        return truck.parking_spot is not None


# ================================================================
# S2: Train Import  (IMPORT_VEHICLE — train→yard)
# ================================================================

class S2_TrainImport(TutorialScenario):
    """Train has one import container. Agent must unload it to yard.

    Teaches: IMPORT_VEHICLE action type for trains.
    Success: container is in yard.
    """
    id = 2
    name = "train_import"
    description = "Agent unloads train to yard"
    max_steps = 15
    expected_moves = None

    def setup(self, env) -> None:
        c = _make_container("S2_C1", Direction.IMPORT)
        tr = _make_train("S2_TR1", containers=[c])
        env.trains[tr.train_id] = tr
        env.rail.slot_train(tr, RailSlot(track_id=0, anchor_bay=ANCHOR_BAY))

    def check_success(self, env) -> bool:
        return env.yard.get_container("S2_C1") is not None


# ================================================================
# S3: Yard Export to Truck  (MOVE_CONTAINER → TRUCK)
# ================================================================

class S3_YardToTruck(TutorialScenario):
    """Container in yard, parked truck wants it.

    Teaches: MOVE_CONTAINER with DestinationType.TRUCK.
    Success: container removed from yard, truck departed.
    """
    id = 3
    name = "yard_to_truck"
    description = "Agent loads container onto parked truck"
    max_steps = 20
    expected_moves = None

    def setup(self, env) -> None:
        c = _make_container("S3_C1", Direction.IMPORT)
        env.yard.add_container(c, _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))

        tk = _make_truck("S3_TK1", pickup_ids=[c.container_id])
        env.trucks[tk.truck_id] = tk
        # Pre-park the truck so agent only needs MOVE_CONTAINER
        spot = next(env.parking.iter_free_in_bay_range(ANCHOR_BAY, ANCHOR_BAY + 2), None)
        if spot is None:
            spot = next(env.parking.iter_free())
        env.parking.allocate(tk, spot.bay, spot.split)

    def check_success(self, env) -> bool:
        return env.yard.container_count == 0 and "S3_TK1" not in env.trucks


# ================================================================
# S4: Yard Export to Train  (MOVE_CONTAINER → TRAIN)
# ================================================================

class S4_YardToTrain(TutorialScenario):
    """Container in yard, train wants it.

    Teaches: MOVE_CONTAINER with DestinationType.TRAIN.
    Success: container loaded onto train.
    """
    id = 4
    name = "yard_to_train"
    description = "Agent loads container onto train"
    max_steps = 20
    expected_moves = None

    def setup(self, env) -> None:
        c = _make_container("S4_C1", Direction.EXPORT)
        env.yard.add_container(c, _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))

        tr = _make_train("S4_TR1", pickup_ids=[c.container_id])
        env.trains[tr.train_id] = tr
        env.rail.slot_train(tr, RailSlot(track_id=0, anchor_bay=ANCHOR_BAY))

    def check_success(self, env) -> bool:
        train = env.trains.get("S4_TR1")
        if train is None:
            return False
        return train.has_container("S4_C1")


# ================================================================
# S5: Train Import → Truck Export  (2-action chain)
# ================================================================

class S5_ImportThenExport(TutorialScenario):
    """Train has container, parked truck wants it.

    Chain: agent IMPORT_VEHICLE (train→yard) → MOVE_CONTAINER (yard→truck).
    Teaches: sequencing import before export.
    """
    id = 5
    name = "import_then_export"
    description = "Agent imports from train, then loads onto truck"
    max_steps = 30
    expected_moves = None

    def setup(self, env) -> None:
        c = _make_container("S5_C1", Direction.IMPORT)
        tr = _make_train("S5_TR1", containers=[c])
        env.trains[tr.train_id] = tr
        env.rail.slot_train(tr, RailSlot(track_id=0, anchor_bay=ANCHOR_BAY))

        tk = _make_truck("S5_TK1", pickup_ids=[c.container_id])
        env.trucks[tk.truck_id] = tk
        # Pre-park truck so chain is: import → export
        env.parking.allocate(tk, bay=ANCHOR_BAY, split=0)

    def check_success(self, env) -> bool:
        return "S5_TK1" not in env.trucks


# ================================================================
# S6: Park → Import  (2-action chain)
# ================================================================

class S6_ParkThenImport(TutorialScenario):
    """Delivery truck (unparked) carries container for yard.

    Chain: agent SLOT_PARKING → IMPORT_VEHICLE (truck→yard).
    Teaches: must park before unloading.
    """
    id = 6
    name = "park_then_import"
    description = "Agent parks truck, then unloads delivery to yard"
    max_steps = 30
    expected_moves = None

    def setup(self, env) -> None:
        c = _make_container("S6_C1", Direction.EXPORT)
        tk = _make_truck("S6_TK1", containers=[c])
        env.trucks[tk.truck_id] = tk
        # Truck NOT parked — must park first, then import

    def check_success(self, env) -> bool:
        return env.yard.get_container("S6_C1") is not None


# ================================================================
# S7: Park → Import → Export to Truck  (full 3-step chain)
# ================================================================

class S7_FullChainTruck(TutorialScenario):
    """Train has container, unparked truck wants it.

    Chain: SLOT_PARKING → IMPORT_VEHICLE (train→yard) → MOVE_CONTAINER (yard→truck).
    Teaches: full pickup lifecycle.
    """
    id = 7
    name = "full_chain_truck"
    description = "Park truck, import from train, load truck"
    max_steps = 40
    expected_moves = None

    def setup(self, env) -> None:
        c = _make_container("S7_C1", Direction.IMPORT)
        tr = _make_train("S7_TR1", containers=[c])
        env.trains[tr.train_id] = tr
        env.rail.slot_train(tr, RailSlot(track_id=0, anchor_bay=ANCHOR_BAY))

        tk = _make_truck("S7_TK1", pickup_ids=[c.container_id])
        env.trucks[tk.truck_id] = tk
        # Truck NOT parked — full chain required

    def check_success(self, env) -> bool:
        return "S7_TK1" not in env.trucks


# ================================================================
# S8: Park → Import → Export to Train  (delivery→train)
# ================================================================

class S8_DeliveryToTrain(TutorialScenario):
    """Delivery truck brings export container, train needs it.

    Chain: SLOT_PARKING → IMPORT_VEHICLE (truck→yard) → MOVE_CONTAINER (yard→train).
    Teaches: delivery-to-export flow.
    """
    id = 8
    name = "delivery_to_train"
    description = "Park delivery truck, unload to yard, load onto train"
    max_steps = 40
    expected_moves = None

    def setup(self, env) -> None:
        c = _make_container("S8_C1", Direction.EXPORT)
        tr = _make_train("S8_TR1", pickup_ids=[c.container_id])
        env.trains[tr.train_id] = tr
        env.rail.slot_train(tr, RailSlot(track_id=0, anchor_bay=ANCHOR_BAY))

        tk = _make_truck("S8_TK1", containers=[c])
        env.trucks[tk.truck_id] = tk
        # Truck NOT parked — full chain required

    def check_success(self, env) -> bool:
        train = env.trains.get("S8_TR1")
        if train is None:
            return False
        return train.has_container("S8_C1")


# ================================================================
# S9: Unbury + Export  (restack + load with distractors)
# ================================================================

class S9_UnburyForTruck(TutorialScenario):
    """Buried container needed by truck, 5 distractors.

    Agent must: restack blocker → MOVE_CONTAINER (target→truck).
    Tests: selective container pick among distractors, restacking.
    """
    id = 9
    name = "unbury_for_truck"
    description = "Restack blocker, then load buried container to truck"
    max_steps = 30
    expected_moves = None

    def setup(self, env) -> None:
        rng = random.Random(42)
        _place_distractors(env, rng, count=5, prefix="S9",
                           exclude_bays={ANCHOR_BAY})

        target = _make_container("S9_TARGET", Direction.IMPORT)
        blocker = _make_container("S9_BLOCK", Direction.IMPORT,
                                  departure=TUTORIAL_TIME + timedelta(days=10))
        env.yard.add_container(target, _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))
        env.yard.add_container(blocker, _yard_placement(bay=ANCHOR_BAY, row=0, tier=1))

        tk = _make_truck("S9_TK1", pickup_ids=[target.container_id])
        env.trucks[tk.truck_id] = tk
        env.parking.allocate(tk, bay=ANCHOR_BAY, split=0)

    def check_success(self, env) -> bool:
        return (env.yard.get_container("S9_TARGET") is None
                and "S9_TK1" not in env.trucks)


# ================================================================
# S10: Unbury + Export to Train  (restack + train load)
# ================================================================

class S10_UnburyForTrain(TutorialScenario):
    """Buried export container needed by train, 5 distractors.

    Agent must: restack blocker → MOVE_CONTAINER (target→train).
    Tests: selective restack for train loading.
    """
    id = 10
    name = "unbury_for_train"
    description = "Restack blocker, then load buried container to train"
    max_steps = 30
    expected_moves = None

    def setup(self, env) -> None:
        rng = random.Random(99)
        _place_distractors(env, rng, count=5, prefix="S10",
                           exclude_bays={ANCHOR_BAY})

        target = _make_container("S10_TARGET", Direction.EXPORT)
        blocker = _make_container("S10_BLOCK", Direction.IMPORT,
                                  departure=TUTORIAL_TIME + timedelta(days=10))
        env.yard.add_container(target, _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))
        env.yard.add_container(blocker, _yard_placement(bay=ANCHOR_BAY, row=0, tier=1))

        tr = _make_train("S10_TR1", pickup_ids=[target.container_id])
        env.trains[tr.train_id] = tr
        env.rail.slot_train(tr, RailSlot(track_id=0, anchor_bay=ANCHOR_BAY))

    def check_success(self, env) -> bool:
        train = env.trains.get("S10_TR1")
        if train is None:
            return False
        return train.has_container("S10_TARGET")


# ================================================================
# Registry
# ================================================================

ALL_SCENARIOS: List[TutorialScenario] = [
    S1_ParkTruck(),           # Primitive: SLOT_PARKING
    S2_TrainImport(),         # Primitive: IMPORT_VEHICLE (train)
    S3_YardToTruck(),         # Primitive: MOVE_CONTAINER → TRUCK
    S4_YardToTrain(),         # Primitive: MOVE_CONTAINER → TRAIN
    S5_ImportThenExport(),    # Chain: import → export
    S6_ParkThenImport(),      # Chain: park → import
    S7_FullChainTruck(),      # Chain: park → import → export (truck)
    S8_DeliveryToTrain(),     # Chain: park → import → export (train)
    S9_UnburyForTruck(),      # Restack + export to truck
    S10_UnburyForTrain(),     # Restack + export to train
]

SCENARIO_BY_ID: Dict[int, TutorialScenario] = {s.id: s for s in ALL_SCENARIOS}