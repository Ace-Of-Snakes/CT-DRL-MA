# simulation/training/tutorial_scenarios.py
"""Tutorial scenarios for curriculum training.

14 progressive scenarios teaching individual skills before combining them.
Direction semantics enforced throughout:
  Import containers: arrive on TRAIN, leave on TRUCK (pickup)
  Export containers: arrive on TRUCK (delivery), leave on TRAIN
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
ANCHOR_BAY = 5
TRAIN_NUM_WAGONS = 5

# Default container specs
CONTAINER_TYPE_40HC = "40' HC"
DEFAULT_LENGTH_FT = 40
DEFAULT_LENGTH_M = 12.2

# Container sizes for randomized tutorials (length_ft, length_m, label)
CONTAINER_SIZES: List[Tuple[int, float, str]] = [
    (20, 6.1, "20' DC"),
    (40, 12.2, "40' HC"),
]

# Yard geometry loaded once at runtime
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
    total_reward: float
    move_log: List[Dict] = field(default_factory=list)

    def __str__(self) -> str:
        tag = "PASS" if self.passed else "FAIL"
        return (f"[{tag}] T{self.scenario_id}: {self.name} "
                f"| steps={self.steps} moves={self.agent_moves} "
                f"R={self.total_reward:.2f}")


# ================================================================
# Entity helpers
# ================================================================

def _make_container(
    cid: str,
    direction: Direction = Direction.IMPORT,
    departure: Optional[datetime] = None,
    length_ft: int = DEFAULT_LENGTH_FT,
    length_m: float = DEFAULT_LENGTH_M,
    container_type: str = CONTAINER_TYPE_40HC,
) -> Container:
    """Create a container for tutorials."""
    return Container(
        container_id=cid,
        direction=direction,
        container_type=container_type,
        arrival_date=TUTORIAL_TIME,
        departure_date=departure or DEFAULT_DEPARTURE,
        goods_type=GoodsType.REGULAR,
        length_ft=length_ft,
        length_m=length_m,
    )


def _make_train(
    tid: str,
    containers: Optional[List[Container]] = None,
    pickup_ids: Optional[List[str]] = None,
    num_wagons: int = TRAIN_NUM_WAGONS,
) -> Train:
    """Create a train, optionally load containers / set pickup ids."""
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


def _yard_placement(bay: int, row: int, tier: int,
                    start_split: int = 0) -> PlacementResult:
    """Convenience wrapper for PlacementResult."""
    return PlacementResult(row=row, bay=bay, tier=tier, start_split=start_split)


def _park_truck_near(env, truck, bay: int) -> None:
    """Park truck at first free spot near *bay*."""
    spot = next(env.parking.iter_free_in_bay_range(bay, bay + 2), None)
    if spot is None:
        spot = next(env.parking.iter_free())
    env.parking.allocate(truck, spot.bay, spot.split)


def _place_distractors(
    env,
    rng: random.Random,
    count: int,
    prefix: str,
    exclude_bays: Set[int],
    sizes: Optional[List[Tuple[int, float, str]]] = None,
) -> int:
    """Place *count* random distractor containers avoiding *exclude_bays*.

    Returns number actually placed.
    """
    sizes = sizes or CONTAINER_SIZES
    placed = 0
    for i in range(count):
        ft, m, label = rng.choice(sizes)
        c = _make_container(
            f"{prefix}_D{i}",
            direction=Direction.IMPORT,
            departure=TUTORIAL_TIME + timedelta(days=15),
            length_ft=ft, length_m=m, container_type=label,
        )
        for _ in range(30):
            bay = rng.randint(0, env.yard.n_bays - 1)
            if bay in exclude_bays:
                continue
            p = env.yard.find_single_placement(c, target_bay=bay)
            if p is not None:
                env.yard.add_container(c, p)
                placed += 1
                break
    return placed


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

    def check_pass(self, env, agent_moves: int) -> bool:
        """Full pass check: success AND optional move budget."""
        if not self.check_success(env):
            return False
        if self.expected_moves is not None:
            if agent_moves != self.expected_moves:
                return False
        return True


# ================================================================
# Phase 1 — Primitives (single action)
# ================================================================

class S1_ParkTruck(TutorialScenario):
    """Unparked pickup truck, Import container in yard.

    Agent must SLOT_PARKING to park the truck.
    """
    id = 1
    name = "park_truck"
    description = "Park an unparked pickup truck"
    max_steps = 15

    def setup(self, env) -> None:
        c = _make_container("S1_C1", Direction.IMPORT)
        env.yard.add_container(c, _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))
        tk = _make_truck("S1_TK1", pickup_ids=[c.container_id])
        env.trucks[tk.truck_id] = tk

    def check_success(self, env) -> bool:
        tk = env.trucks.get("S1_TK1")
        if tk is None:
            return True  # departed = full success
        return tk.parking_spot is not None


class S2_TrainImport(TutorialScenario):
    """Train with 3 Import containers.

    Agent must IMPORT_VEHICLE to unload train -> yard.
    Three containers keep both cranes busy importing.
    """
    id = 2
    name = "train_import"
    description = "Unload 3 Import containers from train to yard"
    max_steps = 15

    def setup(self, env) -> None:
        containers = [
            _make_container(f"S2_C{i}", Direction.IMPORT)
            for i in range(3)
        ]
        tr = _make_train("S2_TR1", containers=containers)
        env.trains[tr.train_id] = tr
        env.rail.slot_train(tr, RailSlot(track_id=0, anchor_bay=ANCHOR_BAY))

    def check_success(self, env) -> bool:
        return all(
            env.yard.get_container(f"S2_C{i}") is not None
            for i in range(3)
        )


class S3_YardToTruck(TutorialScenario):
    """Import container in yard, pre-parked pickup truck wants it.

    Agent must MOVE_CONTAINER -> TRUCK.
    """
    id = 3
    name = "yard_to_truck"
    description = "Load Import container onto parked pickup truck"
    max_steps = 15

    def setup(self, env) -> None:
        c = _make_container("S3_C1", Direction.IMPORT)
        env.yard.add_container(c, _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))
        tk = _make_truck("S3_TK1", pickup_ids=[c.container_id])
        env.trucks[tk.truck_id] = tk
        _park_truck_near(env, tk, ANCHOR_BAY)

    def check_success(self, env) -> bool:
        return (env.yard.get_container("S3_C1") is None
                and "S3_TK1" not in env.trucks)


class S4_YardToTrain(TutorialScenario):
    """Export container in yard, train wants it (in pickup manifest).

    Agent must MOVE_CONTAINER -> TRAIN.
    """
    id = 4
    name = "yard_to_train"
    description = "Load Export container onto waiting train"
    max_steps = 15

    def setup(self, env) -> None:
        c = _make_container("S4_C1", Direction.EXPORT)
        env.yard.add_container(c, _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))
        tr = _make_train("S4_TR1", pickup_ids=[c.container_id])
        env.trains[tr.train_id] = tr
        env.rail.slot_train(tr, RailSlot(track_id=0, anchor_bay=ANCHOR_BAY))

    def check_success(self, env) -> bool:
        tr = env.trains.get("S4_TR1")
        return tr is not None and tr.has_container("S4_C1")


# ================================================================
# Phase 2 — Two-action chains
# ================================================================

class S5_ImportThenExport(TutorialScenario):
    """Train has Import container, pre-parked truck wants it.

    Agent: IMPORT_VEHICLE -> MOVE_CONTAINER->TRUCK.
    """
    id = 5
    name = "import_then_export"
    description = "Import from train, then load onto pickup truck"
    max_steps = 25

    def setup(self, env) -> None:
        c = _make_container("S5_C1", Direction.IMPORT)
        tr = _make_train("S5_TR1", containers=[c])
        env.trains[tr.train_id] = tr
        env.rail.slot_train(tr, RailSlot(track_id=0, anchor_bay=ANCHOR_BAY))
        tk = _make_truck("S5_TK1", pickup_ids=[c.container_id])
        env.trucks[tk.truck_id] = tk
        _park_truck_near(env, tk, ANCHOR_BAY)

    def check_success(self, env) -> bool:
        return "S5_TK1" not in env.trucks


class S6_ParkThenImport(TutorialScenario):
    """Delivery truck (unparked) carries Export container.

    Agent: SLOT_PARKING -> IMPORT_VEHICLE (truck->yard).
    """
    id = 6
    name = "park_then_import"
    description = "Park delivery truck, then unload Export to yard"
    max_steps = 25

    def setup(self, env) -> None:
        c = _make_container("S6_C1", Direction.EXPORT)
        tk = _make_truck("S6_TK1", containers=[c])
        env.trucks[tk.truck_id] = tk

    def check_success(self, env) -> bool:
        return env.yard.get_container("S6_C1") is not None


# ================================================================
# Phase 3 — Full chains (3 actions)
# ================================================================

class S7_FullChainTruck(TutorialScenario):
    """Train has Import container, unparked pickup truck waiting.

    Agent: SLOT_PARKING -> IMPORT_VEHICLE -> MOVE_CONTAINER->TRUCK.
    """
    id = 7
    name = "full_chain_truck"
    description = "Park truck, import from train, load onto truck"
    max_steps = 35

    def setup(self, env) -> None:
        c = _make_container("S7_C1", Direction.IMPORT)
        tr = _make_train("S7_TR1", containers=[c])
        env.trains[tr.train_id] = tr
        env.rail.slot_train(tr, RailSlot(track_id=0, anchor_bay=ANCHOR_BAY))
        tk = _make_truck("S7_TK1", pickup_ids=[c.container_id])
        env.trucks[tk.truck_id] = tk

    def check_success(self, env) -> bool:
        return "S7_TK1" not in env.trucks


class S8_DeliveryToTrain(TutorialScenario):
    """Delivery truck (unparked) with Export container, train wants it.

    Agent: SLOT_PARKING -> IMPORT_VEHICLE -> MOVE_CONTAINER->TRAIN.
    """
    id = 8
    name = "delivery_to_train"
    description = "Park delivery truck, unload to yard, load onto train"
    max_steps = 35

    def setup(self, env) -> None:
        c = _make_container("S8_C1", Direction.EXPORT)
        tr = _make_train("S8_TR1", pickup_ids=[c.container_id])
        env.trains[tr.train_id] = tr
        env.rail.slot_train(tr, RailSlot(track_id=0, anchor_bay=ANCHOR_BAY))
        tk = _make_truck("S8_TK1", containers=[c])
        env.trucks[tk.truck_id] = tk

    def check_success(self, env) -> bool:
        tr = env.trains.get("S8_TR1")
        return tr is not None and tr.has_container("S8_C1")


# ================================================================
# Phase 4 — Restack + load (with distractors)
# ================================================================

class S9_UnburyForTruck(TutorialScenario):
    """Import target buried under blocker, 5 distractors.

    Agent: RESTACK blocker -> MOVE_CONTAINER->TRUCK.
    """
    id = 9
    name = "unbury_for_truck"
    description = "Restack blocker, load buried Import onto truck"
    max_steps = 25

    def setup(self, env) -> None:
        rng = random.Random(42)
        target = _make_container("S9_TARGET", Direction.IMPORT)
        blocker = _make_container(
            "S9_BLOCK", Direction.IMPORT,
            departure=TUTORIAL_TIME + timedelta(days=10),
        )
        env.yard.add_container(target, _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))
        env.yard.add_container(blocker, _yard_placement(bay=ANCHOR_BAY, row=0, tier=1))

        _place_distractors(env, rng, 5, "S9", exclude_bays={ANCHOR_BAY})

        tk = _make_truck("S9_TK1", pickup_ids=[target.container_id])
        env.trucks[tk.truck_id] = tk
        _park_truck_near(env, tk, ANCHOR_BAY)

    def check_success(self, env) -> bool:
        return (env.yard.get_container("S9_TARGET") is None
                and "S9_TK1" not in env.trucks)


class S10_UnburyForTrain(TutorialScenario):
    """Export target buried under blocker, 5 distractors.

    Agent: RESTACK blocker -> MOVE_CONTAINER->TRAIN.
    """
    id = 10
    name = "unbury_for_train"
    description = "Restack blocker, load buried Export onto train"
    max_steps = 25

    def setup(self, env) -> None:
        rng = random.Random(99)
        target = _make_container("S10_TARGET", Direction.EXPORT)
        blocker = _make_container(
            "S10_BLOCK", Direction.IMPORT,
            departure=TUTORIAL_TIME + timedelta(days=10),
        )
        env.yard.add_container(target, _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))
        env.yard.add_container(blocker, _yard_placement(bay=ANCHOR_BAY, row=0, tier=1))

        _place_distractors(env, rng, 5, "S10", exclude_bays={ANCHOR_BAY})

        tr = _make_train("S10_TR1", pickup_ids=[target.container_id])
        env.trains[tr.train_id] = tr
        env.rail.slot_train(tr, RailSlot(track_id=0, anchor_bay=ANCHOR_BAY))

    def check_success(self, env) -> bool:
        tr = env.trains.get("S10_TR1")
        return tr is not None and tr.has_container("S10_TARGET")


# ================================================================
# Phase 5 — Randomised generalization
# ================================================================

class S11_RandomImportChain(TutorialScenario):
    """Random-size Import on train, random distractors, unparked pickup truck.

    Different every run to force generalization over container sizes
    and yard layouts. No fixed RNG seed.
    """
    id = 11
    name = "random_import_chain"
    description = "Full import chain with random sizes and distractors"
    max_steps = 40

    def setup(self, env) -> None:
        rng = random.Random()  # unseeded -> different every run
        ft, m, label = rng.choice(CONTAINER_SIZES)
        c = _make_container(
            "S11_C1", Direction.IMPORT,
            length_ft=ft, length_m=m, container_type=label,
        )
        tr = _make_train("S11_TR1", containers=[c])
        env.trains[tr.train_id] = tr
        env.rail.slot_train(tr, RailSlot(track_id=0, anchor_bay=ANCHOR_BAY))

        n_distractors = rng.randint(3, 8)
        _place_distractors(env, rng, n_distractors, "S11",
                           exclude_bays={ANCHOR_BAY})

        tk = _make_truck("S11_TK1", pickup_ids=[c.container_id])
        env.trucks[tk.truck_id] = tk

    def check_success(self, env) -> bool:
        return "S11_TK1" not in env.trucks


class S12_RandomExportChain(TutorialScenario):
    """Random-size Export on delivery truck, random distractors, train waiting.

    Mirror of S11 for the export path.
    """
    id = 12
    name = "random_export_chain"
    description = "Full export chain with random sizes and distractors"
    max_steps = 40

    def setup(self, env) -> None:
        rng = random.Random()
        ft, m, label = rng.choice(CONTAINER_SIZES)
        c = _make_container(
            "S12_C1", Direction.EXPORT,
            length_ft=ft, length_m=m, container_type=label,
        )
        tr = _make_train("S12_TR1", pickup_ids=[c.container_id])
        env.trains[tr.train_id] = tr
        env.rail.slot_train(tr, RailSlot(track_id=0, anchor_bay=ANCHOR_BAY))

        n_distractors = rng.randint(3, 8)
        _place_distractors(env, rng, n_distractors, "S12",
                           exclude_bays={ANCHOR_BAY})

        tk = _make_truck("S12_TK1", containers=[c])
        env.trucks[tk.truck_id] = tk

    def check_success(self, env) -> bool:
        tr = env.trains.get("S12_TR1")
        return tr is not None and tr.has_container("S12_C1")


# ================================================================
# Phase 6 — Multi-step hard
# ================================================================

class S13_MultiTruckServing(TutorialScenario):
    """3 Import containers in yard, 3 pre-parked trucks each wanting one.

    Tests vehicle selection precision -- agent must match the right
    container to the right truck.
    """
    id = 13
    name = "multi_truck_serving"
    description = "Match 3 containers to 3 pickup trucks"
    max_steps = 30

    def setup(self, env) -> None:
        bays = [ANCHOR_BAY, ANCHOR_BAY + 5, ANCHOR_BAY + 10]
        for i, bay in enumerate(bays):
            c = _make_container(f"S13_C{i}", Direction.IMPORT)
            env.yard.add_container(c, _yard_placement(bay=bay, row=0, tier=0))
            tk = _make_truck(f"S13_TK{i}", pickup_ids=[c.container_id])
            env.trucks[tk.truck_id] = tk
            _park_truck_near(env, tk, bay)

    def check_success(self, env) -> bool:
        return all(f"S13_TK{i}" not in env.trucks for i in range(3))


class S14_DeepUnbury(TutorialScenario):
    """Import target at tier 0, 2 blockers above (tier 1, tier 2).

    Agent must restack twice before loading target onto truck.
    5 random distractors for noise.
    """
    id = 14
    name = "deep_unbury"
    description = "Dig through 2 blockers to reach buried Import container"
    max_steps = 30

    def setup(self, env) -> None:
        rng = random.Random(7)
        target = _make_container("S14_TARGET", Direction.IMPORT)
        blocker1 = _make_container(
            "S14_BLK1", Direction.IMPORT,
            departure=TUTORIAL_TIME + timedelta(days=10),
        )
        blocker2 = _make_container(
            "S14_BLK2", Direction.IMPORT,
            departure=TUTORIAL_TIME + timedelta(days=10),
        )
        env.yard.add_container(target,   _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))
        env.yard.add_container(blocker1, _yard_placement(bay=ANCHOR_BAY, row=0, tier=1))
        env.yard.add_container(blocker2, _yard_placement(bay=ANCHOR_BAY, row=0, tier=2))

        _place_distractors(env, rng, 5, "S14", exclude_bays={ANCHOR_BAY})

        tk = _make_truck("S14_TK1", pickup_ids=[target.container_id])
        env.trucks[tk.truck_id] = tk
        _park_truck_near(env, tk, ANCHOR_BAY)

    def check_success(self, env) -> bool:
        return (env.yard.get_container("S14_TARGET") is None
                and "S14_TK1" not in env.trucks)


# ================================================================
# Registry
# ================================================================

ALL_SCENARIOS: List[TutorialScenario] = [
    # Phase 1: Primitives
    S1_ParkTruck(),
    S2_TrainImport(),
    S3_YardToTruck(),
    S4_YardToTrain(),
    # Phase 2: Two-action chains
    S5_ImportThenExport(),
    S6_ParkThenImport(),
    # Phase 3: Full chains
    S7_FullChainTruck(),
    S8_DeliveryToTrain(),
    # Phase 4: Restack + load
    S9_UnburyForTruck(),
    S10_UnburyForTrain(),
    # Phase 5: Randomised generalization
    S11_RandomImportChain(),
    S12_RandomExportChain(),
    # Phase 6: Multi-step hard
    S13_MultiTruckServing(),
    S14_DeepUnbury(),
]

SCENARIO_BY_ID: Dict[int, TutorialScenario] = {s.id: s for s in ALL_SCENARIOS}