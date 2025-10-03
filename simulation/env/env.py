# simulation/env/env.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable, Any

import numpy as np
from numpy.typing import NDArray

# Facilities imports
from simulation.core.facilities.yard import BooleanStorageYard
from simulation.core.facilities.parking import ParkingArea
from simulation.core.facilities.railyard import BooleanRailYard, RailSlot

# Vehicles imports
from simulation.core.vehicles.train import Train
from simulation.core.vehicles.truck import Truck
from simulation.core.vehicles.terminal_truck import TerminalTruck

# Env imports
from simulation.env.reward_engine import RewardEngine
from simulation.env.state_encoder import TerminalStateEncoder

# Logistics
from simulation.operations.terminal_manager import TerminalLogisticsManager, Move
from simulation.planning.logistics_manager import LogisticsManager, DayPlan
from simulation.analytics.stats_tracker import StatsTracker

# RMGC crane model
from simulation.operations.crane_movements import TerminalRMGC

# Constants and enums
from simulation.core.constants import TERMINAL_TRUCK_TASK_DURATION_S, DEFAULT_STEP_MINUTES
from simulation.core.enums import MoveType
from simulation.config.crane_config import CraneDefaults
from simulation.utils.serialization import serialize_move_record


@dataclass
class CraneState:
    """State tracking for a single crane."""
    id: int
    busy_until: Optional[datetime] = None  # None = idle


class ContainerTerminalEnv:
    """
    Event-driven container terminal with n RMGC cranes.
    
    Features:
    - Multiple cranes with zone-based operation
    - Event-driven simulation (no fixed time steps)
    - Realistic crane timing via TerminalRMGC
    - Automatic arrivals/departures
    - Optional auto-parking or agent-controlled parking
    
    Step loop:
    1. Build broad move list (TerminalLogisticsManager)
    2. Agent selects moves for available cranes (zone overlap constraints)
    3. Execute moves immediately; RMGC provides travel time
    4. Advance time by shortest move duration
    5. Handle arrivals/departures, automatic parking, terminal-truck jobs
    """
    
    def __init__(
        self,
        yard: BooleanStorageYard,
        rail: BooleanRailYard,
        parking: ParkingArea,
        tlm: TerminalLogisticsManager,
        lm: LogisticsManager,
        num_tracks: int,
        step_minutes: int = DEFAULT_STEP_MINUTES,
        overflow_penalty: float = -500.0,
        num_cranes: int = CraneDefaults.NUM_CRANES,
        stats: Optional[StatsTracker] = None,
        auto_park: bool = False
    ):
        """
        Initialize container terminal environment.
        
        Args:
            yard: Yard storage facility
            rail: Rail yard for train positioning
            parking: Truck parking area
            tlm: Terminal logistics manager
            lm: Logistics manager for day planning
            num_tracks: Number of rail tracks
            step_minutes: Minimum step size when no cranes busy
            overflow_penalty: Penalty for constraint violations
            num_cranes: Number of RMGC cranes
            stats: Statistics tracker (optional)
            auto_park: If True, automatically park trucks (no agent control)
        """
        self.yard = yard
        self.rail = rail
        self.parking = parking
        self.tlm = tlm
        self.lm = lm
        self.num_tracks = num_tracks
        self.step_minutes = step_minutes
        self.overflow_penalty = overflow_penalty
        
        self.encoder = TerminalStateEncoder(yard, rail)
        self.reward_engine = RewardEngine(yard)
        self.stats = stats
        
        self.auto_park = bool(auto_park)
        
        # Simulation state
        self.current_time: Optional[datetime] = None
        self.day_index: int = 0
        self.day_plan: Optional[DayPlan] = None
        self._scheduled_trains: List = []
        self._departed_cache: Dict[str, Train] = {}
        
        # Active vehicles
        self.trains: Dict[str, Train] = {}
        self.trucks: Dict[str, Truck] = {}
        self.terminal_trucks: Dict[str, TerminalTruck] = {}
        
        # Terminal truck busy times (env-managed)
        self._tt_busy_until: Dict[str, datetime] = {}
        
        # Track admitted trucks (avoid duplicate admissions)
        self._admitted_truck_ids: set[str] = set()
        
        # Cranes and zones
        self.num_cranes = max(1, int(num_cranes))
        self.cranes: List[CraneState] = []
        self.crane_zones: List[Tuple[int, int]] = []
        
        # RMGC crane model
        self.rmgc = TerminalRMGC(
            yard=self.yard,
            rail=self.rail,
            num_tracks=self.num_tracks
        )
    
    # ==================== Public API ====================
    
    def reset(
        self,
        day_start: datetime,
        day_index: int = 0,
        trains_override: Optional[List[Train]] = None,
        carryover_trains: Optional[Dict[str, Train]] = None,
        carryover_trucks: Optional[Dict[str, Truck]] = None
    ) -> Tuple[NDArray[np.float32], List[Move]]:
        """
        Reset environment to start of a new day.
        
        Args:
            day_start: Start datetime for the day
            day_index: Sequential day index
            trains_override: Optional train list (overrides driving plan)
            carryover_trains: Trains carried over from previous day
            carryover_trucks: Trucks carried over from previous day
            
        Returns:
            Tuple of (state, available_moves)
        """
        # Initialize carryover tracking if needed
        if not hasattr(self, "_carryover_trains"):
            self._carryover_trains = {}
        if not hasattr(self, "_carryover_trucks"):
            self._carryover_trucks = {}
        
        self.day_index = day_index
        self.current_time = day_start.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Create day plan
        self.day_plan = self.lm.plan_day(self.current_time, trains_override=trains_override)
        self._scheduled_trains = list(self.day_plan.todays_trains)
        
        # Clear state
        self.trains.clear()
        self.trucks.clear()
        self.terminal_trucks.clear()
        self._tt_busy_until.clear()
        self._admitted_truck_ids.clear()
        
        # Sync RMGC layout
        self.rmgc.set_layout(
            yard=self.yard,
            rail=self.rail,
            num_tracks=self.num_tracks
        )
        
        # Initialize terminal trucks
        for i in range(2):
            tt = TerminalTruck()
            if not getattr(tt, "truck_id", None):
                from simulation.utils.id_generator import IDGenerator
                setattr(tt, "truck_id", IDGenerator.generate_terminal_truck_id(i))
            self.terminal_trucks[tt.truck_id] = tt
        
        # Initialize cranes
        self.cranes = [CraneState(i, None) for i in range(self.num_cranes)]
        self.crane_zones = self._make_crane_zones(
            overlap_bays=CraneDefaults.ZONE_OVERLAP_BAYS
        )
        
        # Handle carryover (vehicles still present from previous day)
        self._apply_carryover(carryover_trains, carryover_trucks)
        
        # Admit today's scheduled arrivals
        self._admit_arrivals()
        
        # Auto-park if enabled
        if self.auto_park:
            self._auto_slot_parking()
        
        # Generate initial state and moves
        state = self.encoder.encode_with_forecast(
            self.trains,
            self.trucks,
            self.terminal_trucks,
            self.day_plan,
            self.current_time
        )
        moves = self._list_moves()
        
        return state, moves
    
    def step_dual_agent(
        self,
        agent,
        log_cb: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Tuple[NDArray[np.float32], List[Move], float, bool, Dict[str, Any]]:
        """
        Execute one environment step with agent interaction.
        
        Day ends strictly at 23:59 of the same day.
        Carryover is collected and passed to next day's reset().
        
        Args:
            agent: Agent with act() method and optional record_outcome()
            log_cb: Optional callback for logging executed moves
            
        Returns:
            Tuple of (next_state, next_moves, reward, done, info)
        """
        reward = 0.0
        info: Dict[str, Any] = {
            "executed": [],
            "train_departures": [],
            "truck_departures": []
        }
        now = self.current_time
        
        # Handle arrivals/departures
        departed = self._admit_arrivals_and_departures()
        for train_id, leftover in departed:
            train = self._departed_cache.get(train_id)
            if train:
                reward += self.reward_engine.on_train_departure(train)
                info["train_departures"].append(train_id)
                if self.stats:
                    self.stats.on_train_departure(train_id, leftover)
        
        # Auto-park if enabled
        self._auto_slot_parking()
        
        # Check if all cranes are busy
        idle_cranes = [c for c in self.cranes if (c.busy_until is None or c.busy_until <= now)]
        if not idle_cranes:
            # Fast-forward to next crane availability
            next_time = min(c.busy_until for c in self.cranes if c.busy_until is not None)
            advance_min = (next_time - now).total_seconds() / 60.0
            self.current_time = next_time
            self._complete_terminal_truck_jobs()
            reward += self.reward_engine.waiting_penalty(len(self.trucks), advance_min)
            
            state = self.encoder.encode_with_forecast(
                self.trains,
                self.trucks,
                self.terminal_trucks,
                self.day_plan,
                self.current_time
            )
            return state, self._list_moves(), reward, False, info
        
        # Get all available moves
        all_moves = self._list_moves()
        if not all_moves:
            # No moves available - advance time
            self.current_time += timedelta(minutes=self.step_minutes)
            self._complete_terminal_truck_jobs()
            state = self.encoder.encode_with_forecast(
                self.trains,
                self.trucks,
                self.terminal_trucks,
                self.day_plan,
                self.current_time
            )
            return state, self._list_moves(), reward, False, info
        
        # Set up logging
        if log_cb is None and self.stats:
            def log_cb(rec: Dict[str, Any]):
                self.stats.log_move(rec)
        
        # Process each idle crane
        picks: List[Tuple[int, Move, float, float]] = []
        used_bays, used_cids = set(), set()
        zones = self.crane_zones
        
        for crane in idle_cranes:
            # Filter moves for this crane's zone
            candidates = [
                m for m in all_moves
                if self._eligible_for_crane(m, crane.id, zones)
            ]
            if not candidates:
                continue
            
            # Filter out moves with conflicting resources
            filtered = []
            for mv in candidates:
                cids = self._move_container_ids(mv)
                bays = self._move_bays(mv)
                if cids and (used_cids & cids):
                    continue
                if bays and (used_bays & bays):
                    continue
                filtered.append(mv)
            
            if not filtered:
                continue
            
            # Agent selects move
            state = self.encoder.encode_with_forecast(
                self.trains,
                self.trucks,
                self.terminal_trucks,
                self.day_plan,
                now
            )
            action_idx = agent.act(state, filtered)
            
            # Validate action
            if action_idx < 0 or action_idx >= len(filtered):
                if hasattr(agent, 'record_outcome'):
                    agent.record_outcome(False, -1.0)
                reward -= 1.0
                continue
            
            move = filtered[action_idx]
            
            # Handle non-crane moves (parking, terminal trucks)
            if move.type in (MoveType.SLOT_TRUCK_PARKING, MoveType.YARD_TO_TERMINAL_TRUCK):
                success = self._execute_non_crane_move(
                    move,
                    crane,
                    agent,
                    now,
                    log_cb,
                    info
                )
                if success is not None:
                    reward += success
                continue
            
            # Handle crane moves
            crane_reward = self._execute_crane_move(
                move,
                crane,
                agent,
                now,
                log_cb,
                info,
                picks,
                used_bays,
                used_cids
            )
            if crane_reward is not None:
                reward += crane_reward
        
        # Advance time if any moves were executed
        if not picks:
            self.current_time += timedelta(minutes=self.step_minutes)
        else:
            min_time_s = min(t for (_, _, _, t) in picks)
            self.current_time = now + timedelta(seconds=min_time_s)
        
        self._complete_terminal_truck_jobs()
        
        # Handle additional departures after time advance
        departed2 = self._admit_arrivals_and_departures()
        for train_id, leftover in departed2:
            train = self._departed_cache.get(train_id)
            if train:
                reward += self.reward_engine.on_train_departure(train)
                info["train_departures"].append(train_id)
                if self.stats:
                    self.stats.on_train_departure(train_id, leftover)
        
        # Handle truck departures
        events = self._collect_truck_departures()
        info["truck_departures"].extend(events)
        
        # Check for end of day (strict 23:59 cutoff)
        done = False
        day_end = self.day_plan.date.replace(hour=23, minute=59, second=0, microsecond=0)
        if self.current_time >= day_end:
            reward += self.reward_engine.end_of_day_penalty(self.current_time)
            self._rollover_missed_deadlines()
            done = True
        
        # Generate next state and moves
        state = self.encoder.encode_with_forecast(
            self.trains,
            self.trucks,
            self.terminal_trucks,
            self.day_plan,
            self.current_time
        )
        next_moves = self._list_moves()
        
        return state, next_moves, reward, done, info
    
    def get_carryover(self) -> Tuple[Dict[str, Train], Dict[str, Truck]]:
        """
        Get vehicles that should carry over to next day.
        
        Returns:
            Tuple of (carryover_trains, carryover_trucks)
        """
        return (
            dict(getattr(self, "_carryover_trains", {}) or {}),
            dict(getattr(self, "_carryover_trucks", {}) or {})
        )
    
    # ==================== Internal Helpers ====================
    
    def _apply_carryover(
        self,
        carryover_trains: Optional[Dict[str, Train]],
        carryover_trucks: Optional[Dict[str, Truck]]
    ) -> None:
        """Apply carryover trains and trucks from previous day."""
        if carryover_trains:
            for train_id, train in carryover_trains.items():
                if train and (train.departure_time is None or 
                            train.departure_time > self.current_time):
                    # Ensure train has rail slot
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
    
    def _tt_is_available(self, terminal_truck: TerminalTruck) -> bool:
        """Check if terminal truck is available for new task."""
        if not terminal_truck:
            return False
        
        busy_until = self._tt_busy_until.get(terminal_truck.truck_id)
        if busy_until and busy_until > self.current_time:
            return False
        
        if hasattr(terminal_truck, "is_available"):
            return bool(terminal_truck.is_available())
        
        return len(getattr(terminal_truck, "containers", [])) == 0
    
    def _complete_terminal_truck_jobs(self) -> None:
        """Complete terminal truck jobs that have finished."""
        done: List[str] = []
        
        for tt_id, until in self._tt_busy_until.items():
            if until <= self.current_time:
                tt = self.terminal_trucks.get(tt_id)
                if tt:
                    if hasattr(tt, "containers"):
                        tt.containers.clear()
                    if hasattr(tt, "status"):
                        try:
                            from simulation.core.enums import TerminalTruckStatus
                            tt.status = TerminalTruckStatus.IDLE
                        except Exception:
                            pass
                done.append(tt_id)
        
        for tt_id in done:
            self._tt_busy_until.pop(tt_id, None)
    
    def _make_crane_zones(self, overlap_bays: int) -> List[Tuple[int, int]]:
        """
        Create overlapping zones for cranes.
        
        Args:
            overlap_bays: Number of bays to overlap between zones
            
        Returns:
            List of (lo_bay, hi_bay) tuples for each crane
        """
        n = max(1, self.num_cranes)
        B = self.yard.n_bays
        
        if n == 1:
            return [(0, B)]
        
        base = B // n
        zones = []
        
        for i in range(n):
            lo = max(0, i * base - overlap_bays)
            hi = B if i == n - 1 else min(B, (i + 1) * base + overlap_bays)
            zones.append((lo, hi))
        
        return zones
    
    def _list_moves(self) -> List[Move]:
        """Generate all available moves."""
        out: List[Move] = []
        
        # Train <-> Yard
        for train in self.trains.values():
            out.extend(self.tlm.list_yard_to_train(train))
            out.extend(self.tlm.list_train_to_yard(train))
        
        # Truck <-> Yard
        for truck in self.trucks.values():
            out.extend(self.tlm.list_yard_to_truck(truck))
            out.extend(self.tlm.list_truck_to_yard(truck))
        
        # Train <-> Truck
        if self.trains and self.trucks:
            for train in self.trains.values():
                for truck in self.trucks.values():
                    out.extend(self.tlm.list_train_to_truck(train, truck))
                    out.extend(self.tlm.list_truck_to_train(truck, train))
        
        # Terminal trucks
        for tt in self.terminal_trucks.values():
            if self._tt_is_available(tt):
                out.extend(self.tlm.list_yard_to_terminal_truck(tt))
        
        # Parking (active trucks only)
        out.extend(self.tlm.list_parking_moves_active(self.trucks))
        
        # Yard <-> Yard
        out.extend(self.tlm.list_yard_to_yard())
        
        return out
    
    def _auto_slot_parking(self) -> None:
        """Automatically park trucks if auto_park is enabled."""
        if not self.auto_park:
            return
        
        parking_moves = self.tlm.list_parking_moves_active(self.trucks)
        for move in parking_moves:
            try:
                self.tlm.execute(move, self.trains, self.trucks, self.terminal_trucks)
            except Exception:
                pass
    
    def _execute_non_crane_move(
        self,
        move: Move,
        crane: CraneState,
        agent,
        now: datetime,
        log_cb: Optional[Callable],
        info: Dict[str, Any]
    ) -> Optional[float]:
        """
        Execute non-crane move (parking or terminal truck).
        
        Returns:
            Reward value or None if failed
        """
        ok = self.tlm.execute(move, self.trains, self.trucks, self.terminal_trucks)
        if not ok:
            if hasattr(agent, 'record_outcome'):
                agent.record_outcome(False, -0.5)
            return -0.5
        
        if move.type == MoveType.YARD_TO_TERMINAL_TRUCK:
            r = self.reward_engine.immediate_reward(
                move.type.value,
                0.0,
                TERMINAL_TRUCK_TASK_DURATION_S
            )
            if hasattr(agent, 'record_outcome'):
                agent.record_outcome(True, r)
            
            # Set terminal truck busy
            tt_id = move.args.get("terminal_truck_id")
            if tt_id:
                self._tt_busy_until[tt_id] = now + timedelta(
                    seconds=TERMINAL_TRUCK_TASK_DURATION_S
                )
            
            rec = serialize_move_record(
                timestamp=now,
                crane_id=crane.id,
                move_type=move.type.value,
                args=move.args,
                distance_m=0.0,
                time_s=TERMINAL_TRUCK_TASK_DURATION_S,
                reward=r
            )
        else:  # SLOT_TRUCK_PARKING
            r = 0.5
            if hasattr(agent, 'record_outcome'):
                agent.record_outcome(True, r)
            
            rec = serialize_move_record(
                timestamp=now,
                crane_id=crane.id,
                move_type=move.type.value,
                args=move.args,
                distance_m=0.0,
                time_s=0.0,
                reward=r,
                delta_bay=move.args.get("delta_bay", 0)
            )
        
        info["executed"].append(rec)
        if log_cb:
            try:
                log_cb(rec)
            except Exception:
                pass
        
        return r
    
    def _execute_crane_move(
        self,
        move: Move,
        crane: CraneState,
        agent,
        now: datetime,
        log_cb: Optional[Callable],
        info: Dict[str, Any],
        picks: List,
        used_bays: set,
        used_cids: set
    ) -> Optional[float]:
        """
        Execute crane move.
        
        Returns:
            Reward value or None if failed
        """
        # Get endpoints
        move_dict = {"type": move.type.value, "args": move.args}
        endpoints = self.rmgc.endpoints_for_move(
            move_dict,
            self.trains,
            self.trucks,
            self.yard
        )
        if endpoints is None:
            if hasattr(agent, 'record_outcome'):
                agent.record_outcome(False, -0.1)
            return -0.1
        
        # Calculate cost
        cost = self.rmgc.estimate_move_cost(move.type.value, endpoints[0], endpoints[1])
        
        # Execute
        ok = self.tlm.execute(move, self.trains, self.trucks, self.terminal_trucks)
        if not ok:
            if hasattr(agent, 'record_outcome'):
                agent.record_outcome(False, -0.05)
            return -0.05
        
        # Calculate reward
        r = self.reward_engine.immediate_reward(
            move.type.value,
            cost.distance_m,
            cost.time_s
        )
        
        # Check for first service reward
        if move.type in (MoveType.YARD_TO_TRUCK, MoveType.TRAIN_TO_TRUCK):
            truck_id = move.args.get("truck_id")
            if truck_id:
                truck = self.trucks.get(truck_id)
                if truck and truck.loading_start_time is None:
                    truck.loading_start_time = now
                    wait_min = 0.0
                    if truck.arrival_time:
                        wait_min = (now - truck.arrival_time).total_seconds() / 60.0
                    first_service_reward = self.reward_engine.truck_first_service_reward(wait_min)
                    r += first_service_reward
                    if self.stats:
                        self.stats.on_truck_first_service(truck=truck, wait_minutes=wait_min)
        
        # Record outcome
        if hasattr(agent, 'record_outcome'):
            agent.record_outcome(True, r)
        
        # Update crane state
        self.cranes[crane.id].busy_until = now + timedelta(seconds=cost.time_s)
        picks.append((crane.id, move, cost.distance_m, cost.time_s))
        used_bays |= self._move_bays(move)
        used_cids |= self._move_container_ids(move)
        
        # Log
        rec = serialize_move_record(
            timestamp=now,
            crane_id=crane.id,
            move_type=move.type.value,
            args=move.args,
            distance_m=cost.distance_m,
            time_s=cost.time_s,
            reward=r
        )
        info["executed"].append(rec)
        if log_cb:
            try:
                log_cb(rec)
            except Exception:
                pass
        
        return r
    
    def _admit_arrivals(self) -> None:
        """Admit scheduled arrivals without checking departures."""
        still = []
        self._departed_cache = {}
        
        for st in self._scheduled_trains:
            _, h_arr, m_arr = self.lm.time.decode(st.arrival_angle)
            _, h_dep, m_dep = self.lm.time.decode(st.departure_angle)
            
            arr_dt = self.day_plan.date.replace(hour=h_arr, minute=m_arr, second=0, microsecond=0)
            dep_dt = self.day_plan.date.replace(hour=h_dep, minute=m_dep, second=0, microsecond=0)
            
            train_id = st.train.train_id
            
            if train_id not in self.trains and self.current_time >= arr_dt and self.current_time < dep_dt:
                # Calculate anchor bay from track
                anchor = int(round((st.track_id + 1) * (self.yard.n_bays / (self.num_tracks + 1))))
                anchor = max(0, min(self.yard.n_bays - 1, anchor))
                
                self.rail.slot_train(st.train, RailSlot(track_id=st.track_id, anchor_bay=anchor))
                st.train.arrival_time = arr_dt
                st.train.departure_time = dep_dt
                self.trains[train_id] = st.train
                
                if self.stats:
                    self.stats.on_train_arrival(st.train)
            
            still.append(st)
        
        self._scheduled_trains = still
        
        # Admit trucks (only once per truck)
        for truck in self.day_plan.trucks_today:
            if not truck or not truck.arrival_time:
                continue
            if truck.truck_id in self._admitted_truck_ids:
                continue
            if truck.arrival_time <= self.current_time:
                from simulation.core.enums import TruckStatus
                truck.status = TruckStatus.WAITING
                self.trucks[truck.truck_id] = truck
                self._admitted_truck_ids.add(truck.truck_id)
                if self.stats:
                    self.stats.on_truck_arrival(truck)
    
    def _admit_arrivals_and_departures(self) -> List[Tuple[str, int]]:
        """
        Handle both arrivals and departures.
        
        Returns:
            List of (train_id, leftover_count) for departed trains
        """
        departed: List[Tuple[str, int]] = []
        still = []
        self._departed_cache = {}
        
        for st in self._scheduled_trains:
            _, h_arr, m_arr = self.lm.time.decode(st.arrival_angle)
            _, h_dep, m_dep = self.lm.time.decode(st.departure_angle)
            
            arr_dt = self.day_plan.date.replace(hour=h_arr, minute=m_arr, second=0, microsecond=0)
            dep_dt = self.day_plan.date.replace(hour=h_dep, minute=m_dep, second=0, microsecond=0)
            
            train_id = st.train.train_id
            
            # Handle arrival
            if train_id not in self.trains and self.current_time >= arr_dt and self.current_time < dep_dt:
                anchor = int(round((st.track_id + 1) * (self.yard.n_bays / (self.num_tracks + 1))))
                anchor = max(0, min(self.yard.n_bays - 1, anchor))
                
                self.rail.slot_train(st.train, RailSlot(track_id=st.track_id, anchor_bay=anchor))
                st.train.arrival_time = arr_dt
                st.train.departure_time = dep_dt
                self.trains[train_id] = st.train
                
                if self.stats:
                    self.stats.on_train_arrival(st.train)
            
            # Handle departure
            if train_id in self.trains and self.current_time >= dep_dt:
                train = self.trains.pop(train_id)
                leftover = len(train.get_all_pickup_container_ids())
                self.rail.release_train(train_id)
                self._departed_cache[train_id] = train
                departed.append((train_id, leftover))
                continue
            
            still.append(st)
        
        self._scheduled_trains = still
        
        # Admit trucks (only once per truck)
        for truck in self.day_plan.trucks_today:
            if not truck or not truck.arrival_time:
                continue
            if truck.truck_id in self._admitted_truck_ids:
                continue
            if truck.arrival_time <= self.current_time:
                from simulation.core.enums import TruckStatus
                truck.status = TruckStatus.WAITING
                self.trucks[truck.truck_id] = truck
                self._admitted_truck_ids.add(truck.truck_id)
                if self.stats:
                    self.stats.on_truck_arrival(truck)
        
        return departed
    
    def _collect_truck_departures(self) -> List[Dict[str, Any]]:
        """Collect trucks ready to depart."""
        events = []
        to_remove = []
        
        for truck_id, truck in self.trucks.items():
            if truck.is_ready_to_depart():
                if truck.departure_time is None:
                    truck.departure_time = self.current_time
                
                wait_min = 0.0
                if truck.arrival_time:
                    wait_min = (truck.departure_time - truck.arrival_time).total_seconds() / 60.0
                
                events.append({"truck_id": truck.truck_id, "wait_min": wait_min})
                
                if self.stats:
                    self.stats.on_truck_departure(truck=truck, wait_minutes=wait_min)
                
                if self.parking:
                    self.parking.release(truck)
                
                to_remove.append(truck_id)
        
        for truck_id in to_remove:
            self.trucks.pop(truck_id, None)
        
        return events
    
    def _rollover_missed_deadlines(self) -> None:
        """Collect carryover vehicles for next day."""
        self._carryover_trains = {
            tid: train for tid, train in self.trains.items()
            if train and (train.departure_time is None or train.departure_time > self.current_time)
        }
        self._carryover_trucks = {
            tid: truck for tid, truck in self.trucks.items()
            if truck and truck.departure_time is None
        }
    
    def _eligible_for_crane(self, move: Move, crane_id: int, zones: List[Tuple[int, int]]) -> bool:
        """Check if move is eligible for a specific crane."""
        bays = self._move_bays(move)
        if not bays:
            return True
        
        lo, hi = zones[crane_id]
        return all(lo <= b < hi for b in bays)
    
    def _move_bays(self, move: Move) -> set:
        """Extract bay numbers involved in a move."""
        bays = set()
        args = move.args
        
        if move.type in (MoveType.TRAIN_TO_YARD, MoveType.TRUCK_TO_YARD, MoveType.YARD_TO_YARD):
            dst = args.get("placement")
            if dst:
                bays.add(dst.bay)
        
        if move.type in (MoveType.YARD_TO_TRAIN, MoveType.YARD_TO_TRUCK, MoveType.YARD_TO_TERMINAL_TRUCK):
            cid = args.get("container_id")
            pl = self.yard.get_container_placement(cid)
            if pl:
                bays.add(pl.bay)
        
        if move.type in (MoveType.TRAIN_TO_TRUCK, MoveType.TRUCK_TO_TRAIN):
            if "train_id" in args:
                anchor = self.rail.get_anchor_bay(args["train_id"])
                if anchor is not None:
                    bays.add(anchor)
        
        return bays
    
    def _move_container_ids(self, move: Move) -> set:
        """Extract container IDs involved in a move."""
        ids = set()
        args = move.args
        
        if "container_id" in args and args["container_id"]:
            ids.add(args["container_id"])
        
        return ids