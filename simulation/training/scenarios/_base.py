# simulation/training/scenarios/_base.py
"""Shared infrastructure for tutorial scenarios.

Contains constants, helper functions, the TutorialResult dataclass,
and the TutorialScenario ABC.

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

# Legacy alias — used by _place_distractors() and randomised scenarios
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
# Structured logging dataclasses
# ================================================================

@dataclass
class MoveRecord:
    """Rich record for a single agent move in a tutorial scenario."""
    step: int
    move_num: int
    move_type: str
    container_id: str
    distance_m: float
    time_s: float
    reward: float
    proximity_bonus: Optional[float] = None
    # Spatial fields for visualization (populated from info["executed"])
    src_row: Optional[int] = None
    src_bay: Optional[int] = None
    dst_row: Optional[int] = None
    dst_bay: Optional[int] = None


@dataclass
class StepEvent:
    """Non-move event that contributes reward (e.g. success/timeout bonus)."""
    step: int
    event_type: str    # "SUCCESS_BONUS" | "TIMEOUT_PENALTY" | "non_move"
    reward: float
    detail: str


# ================================================================
# Result dataclass
# ================================================================

@dataclass
class TutorialResult:
    """Outcome of one tutorial attempt with structured logging."""
    scenario_id: int
    name: str
    passed: bool
    steps: int
    agent_moves: int
    total_reward: float
    move_log: List[MoveRecord] = field(default_factory=list)
    step_events: List[StepEvent] = field(default_factory=list)
    move_log_raw: List[Dict] = field(default_factory=list)  # backward compat

    @property
    def reward_breakdown(self) -> Dict[str, float]:
        """Sum rewards by move type and event type."""
        breakdown: Dict[str, float] = {}
        for m in self.move_log:
            breakdown[m.move_type] = breakdown.get(m.move_type, 0.0) + m.reward
        for e in self.step_events:
            breakdown[e.event_type] = breakdown.get(e.event_type, 0.0) + e.reward
        return breakdown

    @property
    def move_type_counts(self) -> Dict[str, int]:
        """Count moves by type."""
        counts: Dict[str, int] = {}
        for m in self.move_log:
            counts[m.move_type] = counts.get(m.move_type, 0) + 1
        return counts

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
        # Explicit size provided (e.g. stacking scenarios)
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


def _make_swap_body(prefix: str, idx: int,
                    rng: random.Random = None) -> Container:
    """Create a swap body container for terminal truck tutorials.

    Randomly samples from the 5 real swap body types (22WB-45WB) so the
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
    repeatable: bool = False

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
