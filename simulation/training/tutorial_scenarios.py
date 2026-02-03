# simulation/training/tutorial_scenarios.py
"""Tutorial scenarios for curriculum training.

Nine progressive micro-scenarios that teach the agent individual skills
before combining them.  Each scenario injects a minimal set of entities
and defines a clear pass/fail criterion.
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

# Yard geometry for tutorials (must match env factory)
# BAY_SPLIT_FACTOR is loaded at runtime to stay DRY
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


# ================================================================
# Base class
# ================================================================

class TutorialScenario(ABC):
    """Base class for tutorial scenarios."""

    id: int
    name: str
    description: str
    max_steps: int
    expected_moves: Optional[int] = None  # exact agent-move count for pass

    @abstractmethod
    def setup(self, env) -> None:
        """Inject entities into a freshly-reset tutorial env."""

    @abstractmethod
    def check_success(self, env) -> bool:
        """Return True when the tutorial goal has been achieved."""

    def check_pass(self, env, agent_moves: int, auto_moves: int) -> bool:
        """Full pass check: success AND move budget (total = agent + auto)."""
        if not self.check_success(env):
            return False
        if self.expected_moves is not None:
            total = agent_moves + auto_moves
            if total != self.expected_moves:
                return False
        return True


# ================================================================
# Scenario 1: Yard -> Truck  (AGENT does export)
# ================================================================

class S1_YardToTruck(TutorialScenario):
    """One container in yard, one PARKED truck needing that container.

    Agent must select container and load it onto the truck.
    Tests: container selection, destination type selection, vehicle selection.
    """
    id = 1
    name = "yard_to_truck"
    description = "Agent loads container onto parked truck"
    max_steps = 25  # Give agent time to explore
    expected_moves = None

    def setup(self, env) -> None:
        c = _make_container("TUT1_C1", Direction.IMPORT)
        env.yard.add_container(c, _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))
        
        # Create truck and PRE-PARK it
        tk = _make_truck("TUT1_TK1", pickup_ids=[c.container_id])
        env.trucks[tk.truck_id] = tk
        
        # Park the truck near the container
        spot = next(env.parking.iter_free_in_bay_range(ANCHOR_BAY, ANCHOR_BAY + 2), None)
        if spot is None:
            spot = next(env.parking.iter_free())
        env.parking.allocate(tk, spot.bay, spot.split)

    def check_success(self, env) -> bool:
        return env.yard.container_count == 0 and "TUT1_TK1" not in env.trucks


# ================================================================
# Scenario 1b: Parking Test (agent must park truck)
# ================================================================

class S1b_ParkTruck(TutorialScenario):
    """One container in yard, one UNPARKED truck.

    Agent must choose SLOT_PARKING to park the truck.
    Success = truck gets parked OR container gets loaded.
    """
    id = 10
    name = "park_truck"
    description = "Agent parks an unparked truck"
    max_steps = 25
    expected_moves = None

    def setup(self, env) -> None:
        c = _make_container("TUT1B_C1", Direction.IMPORT)
        env.yard.add_container(c, _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))
        
        # Truck is NOT parked
        tk = _make_truck("TUT1B_TK1", pickup_ids=[c.container_id])
        env.trucks[tk.truck_id] = tk

    def check_success(self, env) -> bool:
        truck = env.trucks.get("TUT1B_TK1")
        if truck is None:
            return True  # Truck departed = full success
        return truck.parking_spot is not None  # Parked = success


# ================================================================
# Scenario 2: Train -> Yard  (auto-import)
# ================================================================

class S2_TrainToYard(TutorialScenario):
    """A train arrives with one import container. Auto-transfer unloads it.

    Agent learns not to interfere; verifies import pipeline works.
    """
    id = 2
    name = "train_to_yard"
    description = "Auto-transfer unloads train to yard"
    max_steps = 10
    expected_moves = None

    def setup(self, env) -> None:
        c = _make_container("TUT2_C1", Direction.IMPORT)
        tr = _make_train("TUT2_TR1", containers=[c])
        env.trains[tr.train_id] = tr
        env.rail.slot_train(tr, RailSlot(track_id=0, anchor_bay=ANCHOR_BAY))

    def check_success(self, env) -> bool:
        return env.yard.container_count >= 1 and env.yard.get_container("TUT2_C1") is not None


# ================================================================
# Scenario 3: Train -> Yard -> Truck  (import + parking + agent export)
# ================================================================

class S3_TrainYardTruck(TutorialScenario):
    """Train with one container + truck in queue wanting that container.

    Chain: 
    1. Auto-import: train->yard
    2. Agent parks truck
    3. Agent exports: yard->truck
    
    Tests full chain where agent does the export.
    """
    id = 3
    name = "train_yard_truck"
    description = "Full import-to-pickup chain (agent does export)"
    max_steps = 35  # More steps since agent does export
    expected_moves = None

    def setup(self, env) -> None:
        c = _make_container("TUT3_C1", Direction.IMPORT)
        tr = _make_train("TUT3_TR1", containers=[c])
        env.trains[tr.train_id] = tr
        env.rail.slot_train(tr, RailSlot(track_id=0, anchor_bay=ANCHOR_BAY))
        tk = _make_truck("TUT3_TK1", pickup_ids=[c.container_id])
        env.trucks[tk.truck_id] = tk

    def check_success(self, env) -> bool:
        # Truck served and departed (removed from env.trucks)
        return "TUT3_TK1" not in env.trucks and env.yard.get_container("TUT3_C1") is None


# ================================================================
# Scenario 4: Train -> Truck direct  (both pre-positioned)
# ================================================================

class S4_TrainToTruckDirect(TutorialScenario):
    """Train parked with container, truck parked adjacent.

    Auto-transfer should handle direct train->truck.
    Tests that the auto-transfer chain covers TRAIN_TO_TRUCK.
    """
    id = 4
    name = "train_to_truck_direct"
    description = "Direct train-to-truck transfer"
    max_steps = 10
    expected_moves = None

    def setup(self, env) -> None:
        c = _make_container("TUT4_C1", Direction.IMPORT)
        tr = _make_train("TUT4_TR1", containers=[c])
        env.trains[tr.train_id] = tr
        env.rail.slot_train(tr, RailSlot(track_id=0, anchor_bay=ANCHOR_BAY))

        tk = _make_truck("TUT4_TK1", pickup_ids=[c.container_id])
        env.trucks[tk.truck_id] = tk
        # Pre-park the truck adjacent to train anchor
        env.parking.allocate(tk, bay=ANCHOR_BAY, split=0)

    def check_success(self, env) -> bool:
        return "TUT4_TK1" not in env.trucks


# ================================================================
# Scenario 5: Park-first chain  (parking + agent export)
# ================================================================

class S5_ParkFirstChain(TutorialScenario):
    """Like S3 but emphasizes: agent MUST park truck first.

    Chain:
    1. Agent parks truck (required first)
    2. Auto-import: train->yard
    3. Agent exports: yard->truck
    """
    id = 5
    name = "park_first_chain"
    description = "Agent parks truck, then does export"
    max_steps = 40
    expected_moves = None

    def setup(self, env) -> None:
        c = _make_container("TUT5_C1", Direction.IMPORT)
        tr = _make_train("TUT5_TR1", containers=[c])
        env.trains[tr.train_id] = tr
        env.rail.slot_train(tr, RailSlot(track_id=0, anchor_bay=ANCHOR_BAY))
        tk = _make_truck("TUT5_TK1", pickup_ids=[c.container_id])
        env.trucks[tk.truck_id] = tk

    def check_success(self, env) -> bool:
        return "TUT5_TK1" not in env.trucks


# ================================================================
# Scenario 6: Truck -> Yard -> Train  (agent does train export)
# ================================================================

class S6_TruckToTrain(TutorialScenario):
    """Truck carries an export container, train needs it.

    Chain:
    1. Agent parks truck
    2. Auto-import: truck->yard
    3. Agent exports: yard->train
    
    Tests agent's ability to load train.
    """
    id = 6
    name = "truck_to_train"
    description = "Delivery truck to export train (agent loads train)"
    max_steps = 40
    expected_moves = None

    def setup(self, env) -> None:
        c = _make_container("TUT6_C1", Direction.EXPORT)
        tr = _make_train("TUT6_TR1", pickup_ids=[c.container_id])
        env.trains[tr.train_id] = tr
        env.rail.slot_train(tr, RailSlot(track_id=0, anchor_bay=ANCHOR_BAY))
        tk = _make_truck("TUT6_TK1", containers=[c])
        env.trucks[tk.truck_id] = tk

    def check_success(self, env) -> bool:
        train = env.trains.get("TUT6_TR1")
        if train is None:
            return False
        return train.has_container("TUT6_C1")


# ================================================================
# Scenario 7: Unbury  (restack + load)
# ================================================================

class S7_Unbury(TutorialScenario):
    """Two containers stacked; bottom one needed by truck.

    Agent restacks top container, then auto-transfer loads bottom to truck.
    """
    id = 7
    name = "unbury"
    description = "Restack blocker, then auto-load buried container"
    max_steps = 20
    expected_moves = None

    def setup(self, env) -> None:
        target = _make_container("TUT7_TARGET", Direction.IMPORT)
        blocker = _make_container("TUT7_BLOCK", Direction.IMPORT,
                                  departure=TUTORIAL_TIME + timedelta(days=10))
        env.yard.add_container(target, _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))
        env.yard.add_container(blocker, _yard_placement(bay=ANCHOR_BAY, row=0, tier=1))

        tk = _make_truck("TUT7_TK1", pickup_ids=[target.container_id])
        env.trucks[tk.truck_id] = tk
        env.parking.allocate(tk, bay=ANCHOR_BAY, split=0)

    def check_success(self, env) -> bool:
        return (env.yard.get_container("TUT7_TARGET") is None
                and "TUT7_TK1" not in env.trucks)


# ================================================================
# Scenario 8: Selective restack  (distractors, exact 2 moves)
# ================================================================

class S8_SelectiveRestack(TutorialScenario):
    """5 random containers + 2-stack with buried target for truck.

    Agent must restack in exactly 2 agent moves (restack + done).
    Tests that agent ignores distractor containers.
    """
    id = 8
    name = "selective_restack_truck"
    description = "Restack buried container among distractors (2 moves)"
    max_steps = 20
    expected_moves = 2  # restack blocker + yard-to-yard (auto does load)

    def setup(self, env) -> None:
        # Place 5 distractor containers at scattered positions
        rng = random.Random(42)
        used: Set[Tuple[int, int]] = set()
        for i in range(5):
            cid = f"TUT8_D{i}"
            c = _make_container(cid, Direction.IMPORT,
                                departure=TUTORIAL_TIME + timedelta(days=15))
            while True:
                bay = rng.randint(1, env.yard.n_bays - 2)
                row = rng.randint(0, env.yard.n_rows - 1)
                if (bay, row) not in used and bay != ANCHOR_BAY:
                    break
            used.add((bay, row))
            env.yard.add_container(c, _yard_placement(bay=bay, row=row, tier=0))

        # Target stack at anchor bay
        target = _make_container("TUT8_TARGET", Direction.IMPORT)
        blocker = _make_container("TUT8_BLOCK", Direction.IMPORT,
                                  departure=TUTORIAL_TIME + timedelta(days=10))
        env.yard.add_container(target, _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))
        env.yard.add_container(blocker, _yard_placement(bay=ANCHOR_BAY, row=0, tier=1))

        tk = _make_truck("TUT8_TK1", pickup_ids=[target.container_id])
        env.trucks[tk.truck_id] = tk
        env.parking.allocate(tk, bay=ANCHOR_BAY, split=0)

    def check_success(self, env) -> bool:
        return (env.yard.get_container("TUT8_TARGET") is None
                and "TUT8_TK1" not in env.trucks)


# ================================================================
# Scenario 9: Selective restack for train
# ================================================================

class S9_SelectiveRestackTrain(TutorialScenario):
    """Like S8 but the buried container goes to a train.

    5 distractors + 2-stack, target is export for waiting train.
    Agent must restack in exactly 2 moves.
    """
    id = 9
    name = "selective_restack_train"
    description = "Restack buried container for train loading (2 moves)"
    max_steps = 20
    expected_moves = 2

    def setup(self, env) -> None:
        rng = random.Random(99)
        used: Set[Tuple[int, int]] = set()
        for i in range(5):
            cid = f"TUT9_D{i}"
            c = _make_container(cid, Direction.IMPORT,
                                departure=TUTORIAL_TIME + timedelta(days=15))
            while True:
                bay = rng.randint(1, env.yard.n_bays - 2)
                row = rng.randint(0, env.yard.n_rows - 1)
                if (bay, row) not in used and bay != ANCHOR_BAY:
                    break
            used.add((bay, row))
            env.yard.add_container(c, _yard_placement(bay=bay, row=row, tier=0))

        target = _make_container("TUT9_TARGET", Direction.EXPORT)
        blocker = _make_container("TUT9_BLOCK", Direction.IMPORT,
                                  departure=TUTORIAL_TIME + timedelta(days=10))
        env.yard.add_container(target, _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))
        env.yard.add_container(blocker, _yard_placement(bay=ANCHOR_BAY, row=0, tier=1))

        tr = _make_train("TUT9_TR1", pickup_ids=[target.container_id])
        env.trains[tr.train_id] = tr
        env.rail.slot_train(tr, RailSlot(track_id=0, anchor_bay=ANCHOR_BAY))

    def check_success(self, env) -> bool:
        train = env.trains.get("TUT9_TR1")
        if train is None:
            return False
        return train.has_container("TUT9_TARGET")


# ================================================================
# Registry
# ================================================================

ALL_SCENARIOS: List[TutorialScenario] = [
    S1_YardToTruck(),      # Test auto-transfer (truck pre-parked)
    S1b_ParkTruck(),       # Test agent parking action
    S2_TrainToYard(),
    S3_TrainYardTruck(),
    S4_TrainToTruckDirect(),
    S5_ParkFirstChain(),
    S6_TruckToTrain(),
    S7_Unbury(),
    S8_SelectiveRestack(),
    S9_SelectiveRestackTrain(),
]

SCENARIO_BY_ID: Dict[int, TutorialScenario] = {s.id: s for s in ALL_SCENARIOS}