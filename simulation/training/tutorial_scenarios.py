# simulation/training/tutorial_scenarios.py
"""Tutorial scenarios for curriculum training.

25 progressive scenarios teaching individual skills before combining them.
Direction semantics enforced throughout:
  Import containers: arrive on TRAIN, leave on TRUCK (pickup)
  Export containers: arrive on TRUCK (delivery), leave on TRAIN
  Terminal trucks: internal-only, carry swap bodies / trailers
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
from simulation.core.vehicles.terminal_truck import TerminalTruck


# ================================================================
# Constants
# ================================================================

TUTORIAL_TIME = datetime(2026, 1, 15, 8, 0, 0)
DEFAULT_DEPARTURE = TUTORIAL_TIME + timedelta(days=5)
ANCHOR_BAY = 5
TRAIN_NUM_WAGONS = 5

# Default container specs (kept for backward compat / explicit-size callers)
CONTAINER_TYPE_40HC = "40' HC"
DEFAULT_LENGTH_FT = 40
DEFAULT_LENGTH_M = 12.2

# ── Full container type specs from container_data.csv ────────────────────
# Regular containers only (no swap bodies, no trailers).
# Format: (length_ft, length_m, container_type, width_m, height_m, is_high_cube)

CONTAINER_SPECS: List[Tuple[int, float, str, float, float, bool]] = [
    # 20ft
    (20, 6.1,  "20DC", 2.44, 2.44, False),
    (20, 6.1,  "20HC", 2.44, 2.89, True),
    (20, 6.1,  "20G",  2.44, 2.44, False),
    (20, 6.1,  "20T",  2.44, 2.59, False),
    # 22ft
    (22, 6.71, "22T",  2.44, 2.59, False),
    # 24ft
    (24, 7.3,  "24T",  2.44, 2.59, False),
    (24, 7.3,  "24OT", 2.44, 2.59, False),
    # 30ft
    (30, 9.14, "30G",  2.44, 2.44, False),
    (30, 9.14, "30O",  2.44, 2.44, False),
    (30, 9.14, "30T",  2.44, 2.59, False),
    # 40ft
    (40, 12.2, "40HC", 2.44, 2.89, True),
    (40, 12.2, "40DC", 2.44, 2.44, False),
    (40, 12.2, "40G",  2.44, 2.44, False),
    (40, 12.2, "40RH", 2.44, 2.59, False),
    # 45ft
    (45, 13.7, "45G",  2.44, 2.44, False),
    (45, 13.7, "45HC", 2.44, 2.89, True),
    (45, 13.7, "45GH", 2.44, 2.44, False),
]

# Per-entry weights approximating real-world distribution (excludes swap bodies).
# Real: 40ft ~56%, 20ft ~17%, 30ft ~7%, 24ft ~4%, 22ft ~1%, 45ft ~1%.
# 23ft excluded (swap-body only). Renormalised to sum=1.
_LENGTH_WEIGHTS = {20: 0.20, 22: 0.01, 24: 0.05, 30: 0.08, 40: 0.65, 45: 0.01}
_SPEC_WEIGHTS: List[float] = [
    _LENGTH_WEIGHTS[spec[0]] / sum(1 for s in CONTAINER_SPECS if s[0] == spec[0])
    for spec in CONTAINER_SPECS
]
# Normalise
_SPEC_WEIGHT_SUM = sum(_SPEC_WEIGHTS)
_SPEC_WEIGHTS = [w / _SPEC_WEIGHT_SUM for w in _SPEC_WEIGHTS]


def _random_container_spec(
    rng: random.Random,
) -> Tuple[int, float, str, float, float, bool]:
    """Sample a random container spec with real-world frequency weights."""
    return rng.choices(CONTAINER_SPECS, weights=_SPEC_WEIGHTS, k=1)[0]


# Swap body types from container_data.csv
# Format: (length_ft, length_m, container_type, width_m)
SWAP_BODY_SPECS: List[Tuple[int, float, str, float]] = [
    (22, 6.71, "22WB", 2.55),
    (23, 7.01, "23WB", 2.55),
    (24, 7.3,  "24WB", 2.55),
    (30, 9.14, "30WB", 2.55),
    (45, 13.7, "45WB", 2.55),
]

# Legacy alias — used by _place_distractors() and S11/S12
CONTAINER_SIZES: List[Tuple[int, float, str]] = [
    (ft, m, ctype) for ft, m, ctype, _, _, _ in CONTAINER_SPECS
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
    length_ft: int = None,
    length_m: float = None,
    container_type: str = None,
    rng: random.Random = None,
) -> Container:
    """Create a container for tutorials.

    When *length_ft* is ``None`` (the default), a random container type is
    sampled from the real-world distribution so the agent sees diverse sizes
    from the very first tutorial tier.
    """
    if length_ft is None:
        _rng = rng or random.Random()
        ft, m, ctype, _w, _h, _hc = _random_container_spec(_rng)
        length_ft, length_m, container_type = ft, m, ctype
    else:
        # Explicit size provided (e.g. S11/S12 or stacking scenarios)
        if length_m is None:
            length_m = DEFAULT_LENGTH_M
        if container_type is None:
            container_type = CONTAINER_TYPE_40HC
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
) -> int:
    """Place *count* random distractor containers avoiding *exclude_bays*.

    Each distractor gets a randomly sampled type from the full real-world
    container distribution.  Returns number actually placed.
    """
    placed = 0
    for i in range(count):
        ft, m, ctype, _w, _h, _hc = _random_container_spec(rng)
        c = _make_container(
            f"{prefix}_D{i}",
            direction=Direction.IMPORT,
            departure=TUTORIAL_TIME + timedelta(days=15),
            length_ft=ft, length_m=m, container_type=ctype,
            rng=rng,
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


def _slot_train(env, train: Train, track_id: int = 0,
                anchor_bay: int = ANCHOR_BAY) -> None:
    """Register train in env.trains and rail."""
    env.trains[train.train_id] = train
    env.rail.slot_train(train, RailSlot(track_id=track_id, anchor_bay=anchor_bay))


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

    Agent must SLOT_PARKING.
    """
    id = 1
    name = "park_truck"
    description = "Park an unparked pickup truck"
    max_steps = 10

    def setup(self, env) -> None:
        c = _make_container("S1_C1", Direction.IMPORT)
        env.yard.add_container(c, _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))
        tk = _make_truck("S1_TK1", pickup_ids=[c.container_id])
        env.trucks[tk.truck_id] = tk

    def check_success(self, env) -> bool:
        truck = env.trucks.get("S1_TK1")
        if truck is None:
            # Truck departed — must have been parked + loaded first
            return True
        return truck.parking_spot is not None


class S2_TrainImport(TutorialScenario):
    """Train with Import container, no trucks.

    Agent must IMPORT_VEHICLE (train→yard).
    """
    id = 2
    name = "train_import"
    description = "Import container from train to yard"
    max_steps = 15

    def setup(self, env) -> None:
        c = _make_container("S2_C1", Direction.IMPORT)
        tr = _make_train("S2_TR1", containers=[c])
        _slot_train(env, tr)

    def check_success(self, env) -> bool:
        return env.yard.get_container("S2_C1") is not None


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
        _slot_train(env, tr)

    def check_success(self, env) -> bool:
        tr = env.trains.get("S4_TR1")
        return tr is not None and tr.has_container("S4_C1")


# ================================================================
# Phase 2 — Two-action chains
# ================================================================

class S5_ImportAndExport(TutorialScenario):
    """Two independent actions on one train.

    Train carries 1 Import container + wants 1 Export from yard.
    Agent: IMPORT_VEHICLE(train→yard) + MOVE_CONTAINER->TRAIN.
    """
    id = 5
    name = "import_and_export"
    description = "Import one container from train, load a different export onto it"
    max_steps = 25

    def setup(self, env) -> None:
        imp = _make_container("S5_IMP", Direction.IMPORT)
        exp = _make_container("S5_EXP", Direction.EXPORT)
        env.yard.add_container(exp, _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))
        tr = _make_train("S5_TR1", containers=[imp], pickup_ids=[exp.container_id])
        _slot_train(env, tr)

    def check_success(self, env) -> bool:
        tr = env.trains.get("S5_TR1")
        if tr is None:
            return False
        export_loaded = tr.has_container("S5_EXP")
        import_in_yard = env.yard.get_container("S5_IMP") is not None
        return export_loaded and import_in_yard

class S6_ParkThenImport(TutorialScenario):
    """Delivery truck with Export container (unparked), plus train.

    Agent: SLOT_PARKING -> IMPORT_VEHICLE(truck→yard).
    """
    id = 6
    name = "park_then_import"
    description = "Park delivery truck, then import its container to yard"
    max_steps = 25

    def setup(self, env) -> None:
        c = _make_container("S6_C1", Direction.EXPORT)
        tk = _make_truck("S6_TK1", containers=[c])
        env.trucks[tk.truck_id] = tk

    def check_success(self, env) -> bool:
        return (env.yard.get_container("S6_C1") is not None
                and "S6_TK1" not in env.trucks)


# ================================================================
# Phase 3 — Full chains
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
        _slot_train(env, tr)
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
        _slot_train(env, tr)
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
    description = "Restack blocker to reach buried Import, load onto truck"
    max_steps = 25

    def setup(self, env) -> None:
        rng = random.Random(42)
        # All stacked containers must share the same size
        ft, m, ctype, _w, _h, _hc = _random_container_spec(rng)
        target = _make_container("S9_TARGET", Direction.IMPORT,
                                  length_ft=ft, length_m=m, container_type=ctype)
        blocker = _make_container("S9_BLK1", Direction.IMPORT,
                                   departure=TUTORIAL_TIME + timedelta(days=10),
                                   length_ft=ft, length_m=m, container_type=ctype)
        env.yard.add_container(target,  _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))
        env.yard.add_container(blocker, _yard_placement(bay=ANCHOR_BAY, row=0, tier=1))

        _place_distractors(env, rng, 5, "S9", exclude_bays={ANCHOR_BAY})

        tk = _make_truck("S9_TK1", pickup_ids=[target.container_id])
        env.trucks[tk.truck_id] = tk
        _park_truck_near(env, tk, ANCHOR_BAY)

    def check_success(self, env) -> bool:
        return (env.yard.get_container("S9_TARGET") is None
                and "S9_TK1" not in env.trucks)


class S10_UnburyForTrain(TutorialScenario):
    """Export target buried under blocker, 5 distractors, train wants it.

    Agent: RESTACK blocker -> MOVE_CONTAINER->TRAIN.
    """
    id = 10
    name = "unbury_for_train"
    description = "Restack blocker to reach buried Export, load onto train"
    max_steps = 25

    def setup(self, env) -> None:
        rng = random.Random(99)
        # All stacked containers must share the same size
        ft, m, ctype, _w, _h, _hc = _random_container_spec(rng)
        target = _make_container("S10_TARGET", Direction.EXPORT,
                                  length_ft=ft, length_m=m, container_type=ctype)
        blocker = _make_container("S10_BLK1", Direction.EXPORT,
                                   departure=TUTORIAL_TIME + timedelta(days=10),
                                   length_ft=ft, length_m=m, container_type=ctype)
        env.yard.add_container(target,  _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))
        env.yard.add_container(blocker, _yard_placement(bay=ANCHOR_BAY, row=0, tier=1))

        _place_distractors(env, rng, 5, "S10", exclude_bays={ANCHOR_BAY})

        tr = _make_train("S10_TR1", pickup_ids=[target.container_id])
        _slot_train(env, tr)

    def check_success(self, env) -> bool:
        tr = env.trains.get("S10_TR1")
        return (tr is not None
                and tr.has_container("S10_TARGET")
                and env.yard.get_container("S10_TARGET") is None)


# ================================================================
# Phase 5 — Randomised generalisation
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
        _slot_train(env, tr)

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
        _slot_train(env, tr)

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
        # All stacked containers must share the same size
        ft, m, ctype, _w, _h, _hc = _random_container_spec(rng)
        target = _make_container("S14_TARGET", Direction.IMPORT,
                                  length_ft=ft, length_m=m, container_type=ctype)
        blocker1 = _make_container(
            "S14_BLK1", Direction.IMPORT,
            departure=TUTORIAL_TIME + timedelta(days=10),
            length_ft=ft, length_m=m, container_type=ctype,
        )
        blocker2 = _make_container(
            "S14_BLK2", Direction.IMPORT,
            departure=TUTORIAL_TIME + timedelta(days=10),
            length_ft=ft, length_m=m, container_type=ctype,
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
# Phase 7 — Integration: batch operations
# ================================================================

_N_DELIVERY = 3
_DELIVERY_BAYS = [ANCHOR_BAY, ANCHOR_BAY + 8, ANCHOR_BAY + 16]


class S15_MultiDeliveryPark(TutorialScenario):
    """3 unparked delivery trucks with Export containers.

    Agent must park all 3, then import all containers to yard.
    Tests batch parking + batch truck-to-yard import pipeline.
    """
    id = 15
    name = "multi_delivery_park"
    description = "Park 3 delivery trucks and import all containers"
    max_steps = 50

    def setup(self, env) -> None:
        for i in range(_N_DELIVERY):
            c = _make_container(f"S15_C{i}", Direction.EXPORT)
            tk = _make_truck(f"S15_TK{i}", containers=[c])
            env.trucks[tk.truck_id] = tk

    def check_success(self, env) -> bool:
        # All 3 containers in yard, all 3 trucks departed (empty delivery)
        in_yard = all(
            env.yard.get_container(f"S15_C{i}") is not None
            for i in range(_N_DELIVERY)
        )
        trucks_gone = all(
            f"S15_TK{i}" not in env.trucks
            for i in range(_N_DELIVERY)
        )
        return in_yard and trucks_gone


class S16_DeliveryToTrainPipeline(TutorialScenario):
    """2 unparked delivery trucks (Export), 1 train wanting both.

    Full export pipeline: park → truck-to-yard → yard-to-train.
    """
    id = 16
    name = "delivery_to_train_pipeline"
    description = "Park 2 delivery trucks, import to yard, load train"
    max_steps = 55

    def setup(self, env) -> None:
        cids = []
        for i in range(2):
            c = _make_container(f"S16_C{i}", Direction.EXPORT)
            cids.append(c.container_id)
            tk = _make_truck(f"S16_TK{i}", containers=[c])
            env.trucks[tk.truck_id] = tk
        tr = _make_train("S16_TR1", pickup_ids=cids)
        _slot_train(env, tr)

    def check_success(self, env) -> bool:
        tr = env.trains.get("S16_TR1")
        if tr is None:
            return False
        return all(tr.has_container(f"S16_C{i}") for i in range(2))


# ================================================================
# Phase 8 — Integration: bidirectional train (bridge)
# ================================================================

class S17_BidirectionalTrainSimple(TutorialScenario):
    """1 train with 1 Import + wants 1 Export (already in yard).
    1 pre-parked pickup truck for the import.

    Bridge scenario: teaches concurrent import + export on the SAME
    train at minimal scale, with no parking required.
      - Load 1 Export from yard onto train
      - Import 1 Import from train to yard
      - Load import onto pre-parked pickup truck
    """
    id = 17
    name = "bidirectional_train_simple"
    description = "Export to train + import from train + deliver to truck (pre-parked)"
    max_steps = 35

    def setup(self, env) -> None:
        # 1 Import container on the train
        imp = _make_container("S17_IMP0", Direction.IMPORT)

        # 1 Export container in the yard, train wants it
        exp = _make_container("S17_EXP0", Direction.EXPORT)
        env.yard.add_container(exp, _yard_placement(bay=ANCHOR_BAY + 5, row=0, tier=0))

        tr = _make_train("S17_TR1", containers=[imp], pickup_ids=[exp.container_id])
        _slot_train(env, tr)

        # 1 pre-parked pickup truck (no parking step needed)
        tk = _make_truck("S17_TK0", pickup_ids=[imp.container_id])
        env.trucks[tk.truck_id] = tk
        _park_truck_near(env, tk, ANCHOR_BAY)

    def check_success(self, env) -> bool:
        tr = env.trains.get("S17_TR1")
        if tr is None:
            return False
        export_loaded = tr.has_container("S17_EXP0")
        truck_served = "S17_TK0" not in env.trucks
        return export_loaded and truck_served


# ================================================================
# Phase 8b — Integration: concurrent import + export (full)
# ================================================================

class S18_ConcurrentImportExport(TutorialScenario):
    """1 train with 2 Imports. 2 Exports in yard that train wants.
    2 unparked pickup trucks for the imports.

    Tests bidirectional flow on a shared train:
      - Load 2 Exports from yard onto train
      - Import 2 Imports from train to yard
      - Park 2 pickup trucks, load them
    """
    id = 18
    name = "concurrent_import_export"
    description = "Bidirectional train: export to train + import to trucks"
    max_steps = 65

    def setup(self, env) -> None:
        # 2 Import containers on the train
        imports = []
        for i in range(2):
            c = _make_container(f"S18_IMP{i}", Direction.IMPORT)
            imports.append(c)

        # 2 Export containers already in yard, train wants them
        export_ids = []
        for i in range(2):
            c = _make_container(f"S18_EXP{i}", Direction.EXPORT)
            bay = ANCHOR_BAY + 5 + i * 4
            env.yard.add_container(c, _yard_placement(bay=bay, row=0, tier=0))
            export_ids.append(c.container_id)

        tr = _make_train("S18_TR1", containers=imports, pickup_ids=export_ids)
        _slot_train(env, tr)

        # 2 unparked pickup trucks for the imports
        for i, imp in enumerate(imports):
            tk = _make_truck(f"S18_TK{i}", pickup_ids=[imp.container_id])
            env.trucks[tk.truck_id] = tk

    def check_success(self, env) -> bool:
        tr = env.trains.get("S18_TR1")
        if tr is None:
            return False
        exports_loaded = all(tr.has_container(f"S18_EXP{i}") for i in range(2))
        trucks_served = all(f"S18_TK{i}" not in env.trucks for i in range(2))
        return exports_loaded and trucks_served


class S19_MixedFleetDispatch(TutorialScenario):
    """2 delivery trucks (Export, unparked) + 2 pickup trucks (unparked).
    1 train with 2 Imports + wants 2 Exports.

    All four action types needed:
      SLOT_PARKING (4 trucks)
      IMPORT_VEHICLE (train→yard for imports, truck→yard for exports)
      MOVE_CONTAINER→TRAIN (exports)
      MOVE_CONTAINER→TRUCK (imports)
    """
    id = 19
    name = "mixed_fleet_dispatch"
    description = "Park 4 trucks, import/export through yard and train"
    max_steps = 75

    def setup(self, env) -> None:
        # 2 Import containers on train → pickup trucks will want them
        imports = []
        for i in range(2):
            c = _make_container(f"S19_IMP{i}", Direction.IMPORT)
            imports.append(c)

        # 2 Export containers on delivery trucks → train wants them
        export_cids = []
        for i in range(2):
            c = _make_container(f"S19_EXP{i}", Direction.EXPORT)
            export_cids.append(c.container_id)
            tk = _make_truck(f"S19_DTK{i}", containers=[c])
            env.trucks[tk.truck_id] = tk

        tr = _make_train("S19_TR1", containers=imports, pickup_ids=export_cids)
        _slot_train(env, tr)

        # 2 pickup trucks wanting the imports
        for i, imp in enumerate(imports):
            tk = _make_truck(f"S19_PTK{i}", pickup_ids=[imp.container_id])
            env.trucks[tk.truck_id] = tk

    def check_success(self, env) -> bool:
        tr = env.trains.get("S19_TR1")
        if tr is None:
            return False
        exports_loaded = all(tr.has_container(f"S19_EXP{i}") for i in range(2))
        pickups_served = all(f"S19_PTK{i}" not in env.trucks for i in range(2))
        deliveries_done = all(f"S19_DTK{i}" not in env.trucks for i in range(2))
        return exports_loaded and pickups_served and deliveries_done


# ================================================================
# Phase 9 — Stress tests (clutter + scale)
# ================================================================

_STRESS_DISTRACTORS = 12


class S20_CrowdedYardPickup(TutorialScenario):
    """12 distractor containers + 3 Import targets scattered across yard.
    3 unparked pickup trucks, each wanting a different target.

    Tests finding specific containers in a cluttered yard and
    correct vehicle-to-container matching.
    """
    id = 20
    name = "crowded_yard_pickup"
    description = "Serve 3 pickup trucks from a cluttered yard"
    max_steps = 60

    def setup(self, env) -> None:
        rng = random.Random(2026)
        target_bays = [3, 20, 35]

        # Place 3 target Import containers
        for i, bay in enumerate(target_bays):
            c = _make_container(f"S20_T{i}", Direction.IMPORT)
            env.yard.add_container(c, _yard_placement(bay=bay, row=0, tier=0))
            tk = _make_truck(f"S20_TK{i}", pickup_ids=[c.container_id])
            env.trucks[tk.truck_id] = tk

        # 12 distractor containers to create clutter
        _place_distractors(
            env, rng, _STRESS_DISTRACTORS, "S20",
            exclude_bays=set(target_bays),
        )

    def check_success(self, env) -> bool:
        return all(f"S20_TK{i}" not in env.trucks for i in range(3))


class S21_FullMiniDay(TutorialScenario):
    """Maximum integration complexity.

    1 train with 3 Imports + wants 3 Exports.
    3 Export containers in yard.
    2 unparked delivery trucks (bringing 2 more Exports — only 3 needed by train).
    3 unparked pickup trucks (wanting the Imports).
    8 distractor containers.

    Agent must orchestrate: park 5 trucks, import from train,
    import from delivery trucks, load exports onto train, load imports
    onto pickup trucks. Not all delivery containers go to train (surplus).
    """
    id = 21
    name = "full_mini_day"
    description = "Complete terminal day: 5 trucks, 1 train, all action types"
    max_steps = 100

    def setup(self, env) -> None:
        rng = random.Random(42)

        # 3 Import containers on train
        imports = []
        for i in range(3):
            c = _make_container(f"S21_IMP{i}", Direction.IMPORT)
            imports.append(c)

        # 3 Export containers already in yard → train wants these
        export_ids = []
        for i in range(3):
            c = _make_container(f"S21_EXP{i}", Direction.EXPORT)
            bay = ANCHOR_BAY + i * 6
            env.yard.add_container(c, _yard_placement(bay=bay, row=0, tier=0))
            export_ids.append(c.container_id)

        tr = _make_train("S21_TR1", containers=imports, pickup_ids=export_ids,
                         num_wagons=10)
        _slot_train(env, tr)

        # 2 delivery trucks with surplus Export containers (not in train manifest)
        for i in range(2):
            c = _make_container(f"S21_DEXP{i}", Direction.EXPORT,
                                departure=TUTORIAL_TIME + timedelta(days=10))
            tk = _make_truck(f"S21_DTK{i}", containers=[c])
            env.trucks[tk.truck_id] = tk

        # 3 pickup trucks wanting imports
        for i, imp in enumerate(imports):
            tk = _make_truck(f"S21_PTK{i}", pickup_ids=[imp.container_id])
            env.trucks[tk.truck_id] = tk

        # Distractors for clutter
        _place_distractors(env, rng, 8, "S21",
                           exclude_bays={ANCHOR_BAY + i * 6 for i in range(3)})

    def check_success(self, env) -> bool:
        tr = env.trains.get("S21_TR1")
        if tr is None:
            return False
        # All 3 exports loaded on train
        exports_loaded = all(tr.has_container(f"S21_EXP{i}") for i in range(3))
        # All 3 pickup trucks served and departed
        pickups_served = all(f"S21_PTK{i}" not in env.trucks for i in range(3))
        # Delivery truck containers imported to yard
        deliveries_in_yard = all(
            env.yard.get_container(f"S21_DEXP{i}") is not None
            for i in range(2)
        )
        # Delivery trucks departed
        deliveries_gone = all(f"S21_DTK{i}" not in env.trucks for i in range(2))
        return exports_loaded and pickups_served and deliveries_in_yard and deliveries_gone


# ================================================================
# Phase 10 — Terminal truck scenarios
# ================================================================

def _make_swap_body(prefix: str, idx: int,
                    rng: random.Random = None) -> Container:
    """Create a swap body container for terminal truck tutorials.

    Randomly samples from the 5 real swap body types (22WB–45WB) so the
    agent sees diverse swap body sizes from the start.
    """
    _rng = rng or random.Random()
    ft, m, ctype, w = _rng.choice(SWAP_BODY_SPECS)
    return Container(
        container_id=f"{prefix}_SB{idx}",
        direction=Direction.IMPORT,
        container_type=ctype,
        arrival_date=TUTORIAL_TIME,
        departure_date=DEFAULT_DEPARTURE,
        goods_type=GoodsType.REGULAR,
        length_ft=ft,
        length_m=m,
        width_m=w,
        is_swap_body=True,
    )


def _add_terminal_truck(env, prefix: str, idx: int) -> TerminalTruck:
    """Create a terminal truck and register it in env.terminal_trucks + env.trucks.

    Sets TruckStatus.WAITING so _is_queued() detects it correctly.
    """
    tt = TerminalTruck(arrival_time=TUTORIAL_TIME)
    tt.truck_id = f"{prefix}_TT{idx}"
    tt.status = TruckStatus.WAITING
    env.terminal_trucks[tt.truck_id] = tt
    env.trucks[tt.truck_id] = tt
    return tt


class S22_TTCantCarryRegular(TutorialScenario):
    """Terminal truck + regular pickup truck in queue.  Regular Import in yard.

    The agent must learn that TTs cannot carry regular containers and
    use the regular pickup truck instead.

    Success: regular truck departs with the container.
    """
    id = 22
    name = "terminal_truck_dispatch"
    description = "Park TT, dispatch swap body via terminal truck"
    max_steps = 15

    def setup(self, env) -> None:
        # A regular Import container in the yard
        c = _make_container("S22_C1", Direction.IMPORT)
        env.yard.add_container(c, _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))

        # Regular pickup truck wanting that container
        tk = _make_truck("S22_TK1", pickup_ids=[c.container_id])
        env.trucks[tk.truck_id] = tk

        # Terminal truck (cannot carry regular containers)
        _add_terminal_truck(env, "S22", 0)

    def check_success(self, env) -> bool:
        # Regular truck departed with the container
        return (env.yard.get_container("S22_C1") is None
                and "S22_TK1" not in env.trucks)


class S23_RegularTruckPreferred(TutorialScenario):
    """Swap body in yard.  Both a regular pickup truck (assigned to it)
    and a terminal truck are in the queue.

    Regular truck gives +3.0 reward vs TT's +2.0.  Agent should learn
    to prefer the regular truck when it is an option.

    Success: regular truck departs with the swap body.
    """
    id = 23
    name = "regular_truck_preferred"
    description = "Prefer regular truck over TT for swap body (higher reward)"
    max_steps = 20

    def setup(self, env) -> None:
        sb = _make_swap_body("S23", 0)
        env.yard.add_container(sb, _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))

        # Regular pickup truck that wants the swap body
        tk = _make_truck("S23_TK1", pickup_ids=[sb.container_id])
        env.trucks[tk.truck_id] = tk

        # Terminal truck (also could carry it, but less reward)
        _add_terminal_truck(env, "S23", 0)

    def check_success(self, env) -> bool:
        # Regular truck departed with the swap body
        return (env.yard.get_container("S23_SB0") is None
                and "S23_TK1" not in env.trucks)


class S24_TTOnlyOption(TutorialScenario):
    """Swap body in yard + regular Import in yard.
    Regular pickup truck is assigned to the regular container (not the swap body).
    Terminal truck is the only vehicle that can remove the swap body.

    Agent must use TT for swap body AND regular truck for its container.

    Success: swap body removed from yard AND regular truck departs.
    """
    id = 24
    name = "tt_only_option"
    description = "TT is only option for swap body; regular truck takes its own container"
    max_steps = 30

    def setup(self, env) -> None:
        # Swap body — only TT can carry it (no regular truck assigned to it)
        sb = _make_swap_body("S24", 0)
        env.yard.add_container(sb, _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))

        # Regular Import — regular truck is assigned to it
        c = _make_container("S24_C1", Direction.IMPORT)
        env.yard.add_container(c, _yard_placement(bay=ANCHOR_BAY + 5, row=0, tier=0))

        # Regular pickup truck wants the regular container (NOT the swap body)
        tk = _make_truck("S24_TK1", pickup_ids=[c.container_id])
        env.trucks[tk.truck_id] = tk

        # Terminal truck — only vehicle that can remove the swap body
        _add_terminal_truck(env, "S24", 0)

    def check_success(self, env) -> bool:
        sb_gone = env.yard.get_container("S24_SB0") is None
        truck_served = "S24_TK1" not in env.trucks
        return sb_gone and truck_served


class S25_BasicTTDispatch(TutorialScenario):
    """Simplest terminal truck scenario: swap body in yard, only a TT available.

    No competing trucks, no regular containers.  Agent just needs to
    park the TT and dispatch the swap body to it.

    Success: swap body removed from yard.
    """
    id = 25
    name = "basic_tt_dispatch"
    description = "Park TT and dispatch swap body (simplest TT case)"
    max_steps = 15

    def setup(self, env) -> None:
        sb = _make_swap_body("S25", 0)
        env.yard.add_container(sb, _yard_placement(bay=ANCHOR_BAY, row=0, tier=0))

        # Only a terminal truck — no regular trucks
        _add_terminal_truck(env, "S25", 0)

    def check_success(self, env) -> bool:
        return env.yard.get_container("S25_SB0") is None


# ================================================================
# Registry
# ================================================================

ALL_SCENARIOS: List[TutorialScenario] = [
    # Tier 0: Primitives
    S1_ParkTruck(),
    S2_TrainImport(),
    S3_YardToTruck(),
    S4_YardToTrain(),
    # Tier 1: Two-action chains
    S5_ImportAndExport(),
    S6_ParkThenImport(),
    # Tier 2: Full chains
    S7_FullChainTruck(),
    S8_DeliveryToTrain(),
    # Tier 3: Restack + load
    S9_UnburyForTruck(),
    S10_UnburyForTrain(),
    # Tier 4: Randomised generalisation
    S11_RandomImportChain(),
    S12_RandomExportChain(),
    # Tier 5: Multi-step hard
    S13_MultiTruckServing(),
    S14_DeepUnbury(),
    # Tier 6: Terminal truck (park TT, dispatch swap bodies)
    S25_BasicTTDispatch(),
    S22_TTCantCarryRegular(),
    S23_RegularTruckPreferred(),
    S24_TTOnlyOption(),
    # Tier 7: Integration — batch operations
    S15_MultiDeliveryPark(),
    S16_DeliveryToTrainPipeline(),
    # Tier 8: Integration — bidirectional train flow
    S17_BidirectionalTrainSimple(),
    S18_ConcurrentImportExport(),
    S19_MixedFleetDispatch(),
    # Tier 9: Stress tests
    S20_CrowdedYardPickup(),
    S21_FullMiniDay(),
]

SCENARIO_BY_ID: Dict[int, TutorialScenario] = {s.id: s for s in ALL_SCENARIOS}