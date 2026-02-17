# simulation/env/env.py
"""
Base container terminal environment — simulation infrastructure.

Provides the core simulation loop (arrivals, departures, cranes, RMGC,
carryover) used by UnifiedContainerTerminalEnv.  All agent-specific
stepping logic lives in the subclass (unified_env.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
from numpy.typing import NDArray

# -- Facilities (optimised) ----------------------------------------------------
from simulation.core.facilities.yard import OptimizedStorageYard, PlacementResult, EMPTY_SLOT
from simulation.core.facilities.parking import OptimizedParkingArea
from simulation.core.facilities.railyard import OptimizedRailYard, RailSlot

# -- Vehicles -----------------------------------------------
from simulation.core.vehicles.train import Train
from simulation.core.vehicles.truck import Truck
from simulation.core.vehicles.terminal_truck import TerminalTruck

# -- Env components -----------------------------------------
from simulation.env.reward_engine import RewardEngine

# -- Operations (optimised) ----------------------------------------------------
from simulation.operations.terminal_manager import TerminalLogisticsManager, Move
from simulation.operations.crane_movements import TerminalRMGC

# -- Planning -----------------------------------------------
from simulation.planning.logistics_manager import LogisticsManager, DayPlan

# -- Config & enums -----------------------------------------
from simulation.core.enums import MoveType, TruckStatus
from simulation.config.crane_config import CraneDefaults
from simulation.config.operations_config import OperationsDefaults
from simulation.utils.id_generator import IDGenerator

# -- Constants ----------------------------------------------
DEFAULT_STEP_MINUTES: int = 5
IDLE_TICK_SECONDS: float = 30.0  # Time advance when crane has nothing to do
TERMINAL_TRUCK_COUNT: int = 2
DAY_END_HOUR: int = 23
DAY_END_MINUTE: int = 59


# ============================================================
# Data classes
# ============================================================

@dataclass(slots=True)
class CraneState:
    """State tracking for a single RMGC crane."""
    id: int
    busy_until: Optional[datetime] = None


# ============================================================
# Environment
# ============================================================

class ContainerTerminalEnv:
    """
    Base environment: simulation infrastructure for container terminals.

    Public API used by subclasses and trainers:
        reset(day_start, ...)  -> NDArray
        step_all_cranes(agent) -> (state, reward, done, info)
        get_carryover()        -> (trains_dict, trucks_dict)
    """

    def __init__(
        self,
        yard: OptimizedStorageYard,
        rail: OptimizedRailYard,
        parking: OptimizedParkingArea,
        tlm: TerminalLogisticsManager,
        lm: LogisticsManager,
        num_tracks: int,
        step_minutes: int = DEFAULT_STEP_MINUTES,
        overflow_penalty: float = -500.0,
        num_cranes: int = CraneDefaults.NUM_CRANES,
        auto_park: bool = False,
        max_retries: int = 1,
        no_destination_penalty: float = -1.0,
    ):
        # -- Core references ----------------------------------------
        self.yard = yard
        self.rail = rail
        self.parking = parking
        self.tlm = tlm
        self.lm = lm
        self.num_tracks = num_tracks
        self.step_minutes = step_minutes
        self.overflow_penalty = overflow_penalty
        self.auto_park = bool(auto_park)

        # -- Stepping config ----------------------------------------
        self.max_retries = max_retries
        self.no_destination_penalty = no_destination_penalty

        # -- Components ---------------------------------------------
        self.reward_engine = RewardEngine(yard)
        self.rmgc = TerminalRMGC(
            yard=yard, rail=rail, num_tracks=num_tracks,
        )

        # -- Simulation state ---------------------------------------
        self.current_time: Optional[datetime] = None
        self.day_index: int = 0
        self.day_plan: Optional[DayPlan] = None
        self._scheduled_trains: List = []
        self._departed_cache: Dict[str, Train] = {}

        # Active vehicles
        self.trains: Dict[str, Train] = {}
        self.trucks: Dict[str, Truck] = {}
        self.terminal_trucks: Dict[str, TerminalTruck] = {}
        self._tt_busy_until: Dict[str, datetime] = {}
        self._admitted_truck_ids: set = set()

        # Cranes
        self.num_cranes = max(1, int(num_cranes))
        self.cranes: List[CraneState] = []
        self.crane_zones: List[Tuple[int, int]] = []

        # Carryover tracking
        self._carryover_trains: Dict[str, Train] = {}
        self._carryover_trucks: Dict[str, Truck] = {}

        # Cached heat bays
        self._train_heat_bays: Set[int] = set()

        # Optional step-level trace log (set to [] to enable, None to disable)
        self._trace_log: Optional[List[Dict[str, Any]]] = None

    # ================================================================
    # Public API
    # ================================================================

    def reset(
        self,
        day_start: datetime,
        day_index: int = 0,
        trains_override: Optional[List[Train]] = None,
        carryover_trains: Optional[Dict[str, Train]] = None,
        carryover_trucks: Optional[Dict[str, Truck]] = None,
    ) -> NDArray[np.float32]:
        """Reset environment to start of a new day. Returns initial state."""
        self.day_index = day_index
        self.current_time = day_start.replace(hour=0, minute=0, second=0, microsecond=0)

        # Day plan
        self.day_plan = self.lm.plan_day(self.current_time, trains_override=trains_override)
        self._scheduled_trains = list(self.day_plan.todays_trains)

        # Clear vehicles
        self.trains.clear()
        self.trucks.clear()
        self.terminal_trucks.clear()
        self._tt_busy_until.clear()
        self._admitted_truck_ids.clear()
        self._departed_cache.clear()
        self.reward_engine.reset_train_tracking()

        # Clear parking (prevents ghost spots from previous day's terminal trucks)
        if self.parking is not None:
            self.parking.occupied[:] = False
            self.parking.truck_ids.fill(None)
            self.parking._truck_spots.clear()

        # Sync RMGC layout
        self.rmgc.set_layout(yard=self.yard, rail=self.rail, num_tracks=self.num_tracks)

        # Terminal trucks
        self._init_terminal_trucks()

        # Cranes
        self.cranes = [CraneState(i, None) for i in range(self.num_cranes)]
        self.crane_zones = self._make_crane_zones(overlap_bays=CraneDefaults.ZONE_OVERLAP_BAYS)

        # Carryover from previous day
        self._apply_carryover(carryover_trains, carryover_trucks)

        # Admit arrivals
        self._admit_arrivals()

        if self.auto_park:
            self._auto_slot_parking()

        # Update heat bays
        self._update_train_heat()

        return self._encode_state()

    def step_all_cranes(self, agent) -> Tuple[NDArray[np.float32], float, bool, Dict[str, Any]]:
        """Execute steps for all idle cranes. Subclasses must override."""
        raise NotImplementedError("Subclass must implement step_all_cranes")

    def get_carryover(self) -> Tuple[Dict[str, Train], Dict[str, Truck]]:
        """Get vehicles that should carry over to next day."""
        return dict(self._carryover_trains), dict(self._carryover_trucks)

    # ================================================================
    # Container lookup
    # ================================================================

    def _container_at_position(
        self, row: int, split: int, tier: int,
    ) -> Optional[Any]:
        """Look up container record at (row, split, tier) in yard."""
        if not (0 <= tier < self.yard.n_tiers
                and 0 <= row < self.yard.n_rows
                and 0 <= split < self.yard.total_splits):
            return None
        idx = self.yard.position_grid[tier, row, split]
        if idx == EMPTY_SLOT:
            return None
        return self.yard._records[idx]

    # ================================================================
    # Arrivals / departures / time
    # ================================================================

    def _admit_arrivals(self) -> None:
        """Admit scheduled trains and trucks whose arrival time has passed."""
        still = []
        for st in self._scheduled_trains:
            _, h_arr, m_arr = self.lm.time.decode(st.arrival_angle)
            _, h_dep, m_dep = self.lm.time.decode(st.departure_angle)

            arr_dt = self.day_plan.date.replace(hour=h_arr, minute=m_arr, second=0, microsecond=0)
            dep_dt = self.day_plan.date.replace(hour=h_dep, minute=m_dep, second=0, microsecond=0)
            train_id = st.train.train_id

            if train_id not in self.trains and self.current_time >= arr_dt and self.current_time < dep_dt:
                anchor = int(round((st.track_id + 1) * (self.yard.n_bays / (self.num_tracks + 1))))
                anchor = max(0, min(self.yard.n_bays - 1, anchor))
                self.rail.slot_train(st.train, RailSlot(track_id=st.track_id, anchor_bay=anchor))
                st.train.arrival_time = arr_dt
                st.train.departure_time = dep_dt
                self.trains[train_id] = st.train

            still.append(st)
        self._scheduled_trains = still

        # Trucks
        for truck in self.day_plan.trucks_today:
            if not truck or not truck.arrival_time:
                continue
            if truck.truck_id in self._admitted_truck_ids:
                continue
            if truck.arrival_time <= self.current_time:
                truck.status = TruckStatus.WAITING
                self.trucks[truck.truck_id] = truck
                self._admitted_truck_ids.add(truck.truck_id)

    def _admit_arrivals_and_departures(self) -> List[Tuple[str, int]]:
        """Handle arrivals and train departures. Returns [(train_id, leftover_count)]."""
        self._admit_arrivals()
        departed: List[Tuple[str, int]] = []
        to_remove = []

        for train_id, train in self.trains.items():
            if train.departure_time and self.current_time >= train.departure_time:
                leftover = len(train.get_all_pickup_container_ids())
                departed.append((train_id, leftover))
                self._departed_cache[train_id] = train
                self.rail.release_train(train_id)
                to_remove.append(train_id)

        for tid in to_remove:
            self.trains.pop(tid, None)

        return departed

    def _handle_arrivals_departures(self) -> float:
        """Admit arrivals, process train departures, return reward."""
        departed = self._admit_arrivals_and_departures()
        reward = 0.0
        for train_id, _leftover in departed:
            train = self._departed_cache.pop(train_id, None)
            if train:
                reward += self.reward_engine.on_train_departure(train)
        self._update_train_heat()
        return reward

    def _collect_truck_departures(self) -> List[Dict[str, Any]]:
        """Remove departed trucks, return event dicts.

        Terminal trucks are excluded — they never depart.
        """
        events = []
        to_remove = []
        for truck_id, truck in self.trucks.items():
            if getattr(truck, "is_terminal_truck", False):
                continue  # terminal trucks never depart
            if truck.is_ready_to_depart():
                if truck.departure_time is None:
                    truck.departure_time = self.current_time
                wait_min = 0.0
                if truck.arrival_time:
                    wait_min = (truck.departure_time - truck.arrival_time).total_seconds() / 60.0
                events.append({"truck_id": truck_id, "wait_min": wait_min})
                if self.parking:
                    self.parking.release(truck)
                to_remove.append(truck_id)
        for tid in to_remove:
            self.trucks.pop(tid, None)
        return events

    def _advance_time(self, seconds: float = 0.0) -> None:
        """Advance simulation time by actual operation duration.

        Args:
            seconds: Duration of crane operation (from RMGC cost).
                     Falls back to IDLE_TICK_SECONDS if 0.
        """
        dt = seconds if seconds > 0 else IDLE_TICK_SECONDS
        self.current_time += timedelta(seconds=dt)
        self._complete_terminal_truck_jobs()

    def _check_day_end(self) -> bool:
        """Check if simulation day has ended."""
        if self.day_plan is None:
            return True
        day_end = self.day_plan.date.replace(hour=DAY_END_HOUR, minute=DAY_END_MINUTE, second=0)
        return self.current_time >= day_end

    # ================================================================
    # State encoding (overridden by subclass)
    # ================================================================

    def _encode_state(self) -> NDArray[np.float32]:
        """Encode current state. Subclasses must override."""
        raise NotImplementedError("Subclass must implement _encode_state")

    def _update_train_heat(self) -> None:
        """Update cached train heat bays from current trains."""
        self._train_heat_bays.clear()
        for train_id in self.trains:
            anchor = self.rail.get_anchor_bay(train_id)
            if anchor is not None:
                self._train_heat_bays.add(anchor)

    # ================================================================
    # Crane management
    # ================================================================

    def _make_crane_zones(self, overlap_bays: int = 2) -> List[Tuple[int, int]]:
        """Create crane zone boundaries with overlap."""
        n = self.num_cranes
        total = self.yard.n_bays
        if n <= 1:
            return [(0, total)]
        zone_size = total / n
        zones = []
        for i in range(n):
            lo = max(0, int(i * zone_size) - overlap_bays)
            hi = min(total, int((i + 1) * zone_size) + overlap_bays)
            zones.append((lo, hi))
        return zones

    # ================================================================
    # Terminal trucks
    # ================================================================

    def _init_terminal_trucks(self) -> None:
        """Initialise terminal trucks for the day.

        Terminal trucks are added to both ``self.terminal_trucks`` (for TLM
        dispatch) and ``self.trucks`` (so the unified state encoder sees them
        in the queue / parking rows).  Their status is set to
        ``TruckStatus.WAITING`` so that ``_is_queued()`` treats them as
        unparked trucks waiting to be placed by the agent.
        """
        self.terminal_trucks.clear()
        for i in range(TERMINAL_TRUCK_COUNT):
            tt = TerminalTruck(arrival_time=self.current_time)
            if not getattr(tt, "truck_id", None):
                setattr(tt, "truck_id", IDGenerator.generate_terminal_truck_id(i))
            # Use regular TruckStatus so the encoder's _is_queued() picks it up
            tt.status = TruckStatus.WAITING
            self.terminal_trucks[tt.truck_id] = tt
            # Also register in the main trucks dict for encoder visibility
            self.trucks[tt.truck_id] = tt

    def _complete_terminal_truck_jobs(self) -> None:
        """Complete terminal truck jobs that have finished."""
        for tt_id, busy_until in list(self._tt_busy_until.items()):
            if busy_until and busy_until <= self.current_time:
                tt = self.terminal_trucks.get(tt_id)
                if tt:
                    tt.complete_task(self.current_time)
                self._tt_busy_until[tt_id] = None

    # ================================================================
    # Carryover
    # ================================================================

    def _apply_carryover(
        self,
        carryover_trains: Optional[Dict[str, Train]],
        carryover_trucks: Optional[Dict[str, Truck]],
    ) -> None:
        """Apply carryover vehicles from previous day."""
        if carryover_trains:
            for train_id, train in carryover_trains.items():
                if train and (train.departure_time is None or train.departure_time > self.current_time):
                    if self.rail.get_slot(train.train_id) is None:
                        anchor = self.rail.get_anchor_bay(train.train_id)
                        if anchor is None:
                            anchor = max(0, min(self.yard.n_bays - 1, self.yard.n_bays // 2))
                        track_id = int(getattr(train, "rail_track", 0) or 0)
                        self.rail.slot_train(train, RailSlot(track_id=track_id, anchor_bay=anchor))
                    self.trains[train_id] = train

        if carryover_trucks:
            for truck_id, truck in carryover_trucks.items():
                if truck and truck.departure_time is None:
                    self.trucks[truck_id] = truck

    def _rollover_missed_deadlines(self) -> None:
        """Collect vehicles for next-day carryover."""
        self._carryover_trains = {
            tid: t for tid, t in self.trains.items()
            if t and (t.departure_time is None or t.departure_time > self.current_time)
        }
        self._carryover_trucks = {
            tid: t for tid, t in self.trucks.items()
            if t and t.departure_time is None
            and not getattr(t, "is_terminal_truck", False)
        }

    # ================================================================
    # Auto-parking
    # ================================================================

    def _auto_slot_parking(self) -> None:
        """Automatically park trucks (if auto_park enabled)."""
        if not self.auto_park:
            return
        parking_moves = self.tlm.list_parking_moves_active(self.trucks)
        for move in parking_moves:
            try:
                self.tlm.execute(move, self.trains, self.trucks, self.terminal_trucks)
            except Exception:
                pass
