# simulation/env/env.py
"""
Unified container terminal environment for hierarchical DQN training.

Merges base simulation infrastructure (arrivals, departures, cranes, RMGC)
with hierarchical stepping (action pool → stage1/stage2 → execute).

Optimised for training throughput:
- State encoded once per step_all_cranes, passed between crane steps
- Time advances after every crane action (simulation correctness)
- No legacy step_dual_agent path
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
from numpy.typing import NDArray

# ── Facilities (optimised) ─────────────────────────────────────
from simulation.core.facilities.yard import OptimizedStorageYard, PlacementResult
from simulation.core.facilities.parking import OptimizedParkingArea
from simulation.core.facilities.railyard import OptimizedRailYard, RailSlot

# ── Vehicles ───────────────────────────────────────────────────
from simulation.core.vehicles.train import Train
from simulation.core.vehicles.truck import Truck
from simulation.core.vehicles.terminal_truck import TerminalTruck

# ── Env components ─────────────────────────────────────────────
from simulation.env.reward_engine import RewardEngine
from simulation.env.state_encoder import SplitLevelStateEncoder

# ── Operations (optimised) ─────────────────────────────────────
from simulation.operations.terminal_manager import TerminalLogisticsManager, Move
from simulation.operations.hierarchical_moves import HierarchicalMoveGenerator
from simulation.operations.crane_movements import TerminalRMGC

# ── Planning & analytics ──────────────────────────────────────
from simulation.planning.logistics_manager import LogisticsManager, DayPlan
from simulation.analytics.stats_tracker import StatsTracker

# ── RL agent types ─────────────────────────────────────────────
from simulation.rl.features.featurizers import (
    MoveableContainer, Destination, ParkingAction,
    SourceType, DestinationType,
)
from simulation.rl.multihead_dqn.agent import MultiHeadDQNAgent, ActionResult
from simulation.rl.multihead_dqn.config import (
    ActionType as AgentActionType, DestinationType as AgentDestType,
)
from simulation.rl.multihead_dqn.replay_buffer import Transition

# ── Config & enums ─────────────────────────────────────────────
from simulation.core.enums import MoveType, TruckStatus
from simulation.config.crane_config import CraneDefaults
from simulation.config.operations_config import OperationsDefaults

# ── Constants ──────────────────────────────────────────────────
DEFAULT_STEP_MINUTES: int = 5
IDLE_TICK_SECONDS: float = 30.0  # Time advance when crane has nothing to do
TERMINAL_TRUCK_COUNT: int = 2
HIERARCHICAL_PROXIMITY_BAYS: int = OperationsDefaults.PROXIMITY_SEARCH_BAYS
DAY_END_HOUR: int = 23
DAY_END_MINUTE: int = 59

# MultiHead agent integration
DEFAULT_N_SPLITS: int = 10          # 20ft container — smallest common size
VEHICLE_FEAT_DIM: int = 8           # must match HeadConfig.vehicle_feat_dim
SECONDS_PER_DAY: float = 86_400.0
MAX_DEPARTURE_HOURS: float = 24.0


# ═══════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════

@dataclass(slots=True)
class CraneState:
    """State tracking for a single RMGC crane."""
    id: int
    busy_until: Optional[datetime] = None


@dataclass
class HierarchicalStepResult:
    """Result of one hierarchical step for a single crane."""
    next_state: NDArray[np.float32]
    reward: float
    done: bool
    info: Dict[str, Any]
    transition: Optional[Transition] = None
    was_valid_move: bool = False


# ═══════════════════════════════════════════════════════════════
# Environment
# ═══════════════════════════════════════════════════════════════

class ContainerTerminalEnv:
    """
    Unified environment: simulation infrastructure + hierarchical stepping.

    Public API used by CurriculumTrainer:
        reset(day_start, ...)  → NDArray
        step_all_cranes(agent) → (state, reward, done, info)
        get_carryover()        → (trains_dict, trucks_dict)
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
        stats: Optional[StatsTracker] = None,
        auto_park: bool = False,
        max_retries: int = 1,
        no_destination_penalty: float = -1.0,
    ):
        # ── Core references ───────────────────────────────────
        self.yard = yard
        self.rail = rail
        self.parking = parking
        self.tlm = tlm
        self.lm = lm
        self.num_tracks = num_tracks
        self.step_minutes = step_minutes
        self.overflow_penalty = overflow_penalty
        self.auto_park = bool(auto_park)
        self.stats = stats

        # ── Hierarchical config ───────────────────────────────
        self.max_retries = max_retries
        self.no_destination_penalty = no_destination_penalty

        # ── Components ────────────────────────────────────────
        self.encoder = SplitLevelStateEncoder(yard, rail)
        self.reward_engine = RewardEngine(yard)
        self.move_gen = HierarchicalMoveGenerator(
            yard=yard, rail=rail, parking=parking,
            proximity=HIERARCHICAL_PROXIMITY_BAYS,
        )
        self.rmgc = TerminalRMGC(
            yard=yard, rail=rail, num_tracks=num_tracks,
        )

        # ── Simulation state ──────────────────────────────────
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

        # Cached heat bays for hierarchical move gen
        self._train_heat_bays: Set[int] = set()

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

    def step_all_cranes(
        self, agent: MultiHeadDQNAgent,
    ) -> Tuple[NDArray[np.float32], float, bool, Dict[str, Any]]:
        """
        Execute hierarchical steps for all idle cranes.

        Encodes state ONCE, passes between crane steps.
        Returns (next_state, total_reward, done, info).
        """
        total_reward = 0.0
        all_info: Dict[str, Any] = {
            "crane_results": [],
            "executed": [],
            "transitions": [],
        }
        done = False

        # Handle arrivals/departures (reward from train departures)
        departure_reward = self._handle_arrivals_departures()
        total_reward += departure_reward

        if self.auto_park:
            self._auto_slot_parking()

        now = self.current_time
        idle_cranes = [c for c in self.cranes if c.busy_until is None or c.busy_until <= now]

        # Encode state ONCE
        state = self._encode_state()

        if not idle_cranes:
            # All cranes busy → fast-forward to next available
            next_time = min(c.busy_until for c in self.cranes if c.busy_until is not None)
            advance_min = (next_time - now).total_seconds() / 60.0
            self.current_time = next_time
            self._complete_terminal_truck_jobs()
            total_reward += self.reward_engine.waiting_penalty(len(self.trucks), advance_min)
            state = self._encode_state()
            done = self._check_day_end()
            return state, total_reward, done, all_info

        # Process each idle crane
        for crane in idle_cranes:
            result = self._step_multihead(agent, crane.id, state)
            state = result.next_state

            total_reward += result.reward
            all_info["crane_results"].append({
                "crane_id": crane.id,
                "reward": result.reward,
                "was_valid": result.was_valid_move,
            })
            all_info["executed"].extend(result.info.get("executed", []))

            if result.transition is not None:
                all_info["transitions"].append(result.transition)
                agent.remember(result.transition)

            if result.done:
                done = True
                break

        # Handle truck departures
        truck_events = self._collect_truck_departures()
        all_info["truck_departures"] = truck_events

        if not done:
            done = self._check_day_end()
            if done:
                total_reward += self.reward_engine.end_of_day_penalty(self.current_time)
                self._rollover_missed_deadlines()

        return state, total_reward, done, all_info

    def get_carryover(self) -> Tuple[Dict[str, Train], Dict[str, Truck]]:
        """Get vehicles that should carry over to next day."""
        return dict(self._carryover_trains), dict(self._carryover_trucks)

    # ================================================================
    # MultiHead DQN stepping
    # ================================================================

    def _step_multihead(
        self,
        agent: MultiHeadDQNAgent,
        crane_id: int,
        state_np: NDArray[np.float32],
    ) -> HierarchicalStepResult:
        """Execute one step with retry loop for invalid placements.

        Design:
        1. Auto-transfer: exports (yard→train/truck) then imports (train/truck→yard)
        2. If no transfer pending, agent picks yard restack or parking action
        3. Placement is validated before execution
        4. Invalid -> penalty, time stands still, agent retries
        5. After max_retries -> advance time, return accumulated penalty
        """
        info: Dict[str, Any] = {"crane_id": crane_id, "executed": [], "retries": 0}

        # ── Phase 1: auto-transfer (exports + imports) ──────────
        transfer_result = self._try_auto_transfer(crane_id, info)
        if transfer_result is not None:
            return transfer_result

        # ── Phase 2: agent decision with retry on invalid moves ─
        total_penalty = 0.0

        for retry in range(self.max_retries):
            info["retries"] = retry

            occ_mask = self.encoder.get_occupancy_mask()
            val_mask = self.encoder.get_validity_mask(DEFAULT_N_SPLITS)
            veh_feats, veh_mask, veh_list = self._build_vehicle_features()
            park_feats, park_mask, park_list = self._build_parking_features()

            has_containers = occ_mask.any()
            has_parking = len(park_mask) > 0 and park_mask.any()

            if not has_containers and not has_parking:
                break  # nothing actionable

            action = agent.act(
                state=state_np,
                occupancy_mask=occ_mask,
                validity_mask=val_mask,
                vehicle_feats=veh_feats,
                vehicle_mask=veh_mask,
                parking_feats=park_feats,
                parking_mask=park_mask,
            )

            # Parking — no spatial validation needed
            if action.action_type == AgentActionType.SLOT_PARKING:
                result = self._execute_parking_action(
                    state_np, action, park_list, crane_id, info,
                )
                result.reward += total_penalty
                return result

            # Container move — validate placement, then execute
            result = self._try_execute_move(
                state_np, action, veh_list, crane_id, info,
            )
            if result is not None:
                result.reward += total_penalty
                return result

            # Invalid move: accumulate penalty, time stands still, retry
            total_penalty += self.no_destination_penalty
            info.setdefault("retry_reasons", []).append("invalid_placement")

        # All retries exhausted -> advance time
        self._advance_time()
        return HierarchicalStepResult(
            next_state=self._encode_state(),
            reward=total_penalty,
            done=self._check_day_end(),
            info=info, was_valid_move=False,
        )

    # ================================================================
    # Move validation + execution
    # ================================================================

    def _try_execute_move(
        self,
        state_np: NDArray,
        action: ActionResult,
        vehicle_list: List,
        crane_id: int,
        info: Dict,
    ) -> Optional[HierarchicalStepResult]:
        """Validate and execute a spatial move.

        Returns HierarchicalStepResult on success, None on invalid move.
        Time only advances on success.
        """
        # ── Validate container selection ──────────────────────
        if action.container_pos is None:
            return None
        row, split, tier = action.container_pos
        record = self._container_at_position(row, split, tier)
        if record is None:
            return None

        container = record.container
        container_id = container.container_id
        dest_type = action.dest_type

        # ── Build move based on destination type ──────────────
        if dest_type is None or dest_type == AgentDestType.YARD:
            move = self._validate_yard_placement(container, record, action)
        elif dest_type == AgentDestType.TRAIN:
            move = self._validate_train_load(container, action, vehicle_list)
        elif dest_type == AgentDestType.TRUCK:
            move = self._validate_truck_load(container, action, vehicle_list)
        else:
            return None

        if move is None:
            return None  # validation failed — caller handles retry

        # ── Execute via TLM ───────────────────────────────────
        try:
            self.tlm.execute(move, self.trains, self.trucks, self.terminal_trucks)
        except Exception as e:
            info.setdefault("errors", []).append(str(e))
            return None

        # ── Crane timing & reward ─────────────────────────────
        epc = self.rmgc.endpoints_and_cost_for_move(
            move, self.trains, self.trucks, self.yard,
        )
        cost = epc[2] if epc else None
        if crane_id < len(self.cranes) and cost:
            self.cranes[crane_id].busy_until = (
                self.current_time + timedelta(seconds=cost.time_s)
            )

        reward = self.reward_engine.immediate_reward(
            move.type.value,
            cost.distance_m if cost else 0.0,
            cost.time_s if cost else 0.0,
        )

        info["executed"].append({
            "move_type": move.type.value,
            "container_id": container_id,
            "distance_m": cost.distance_m if cost else 0.0,
            "time_s": cost.time_s if cost else 0.0,
        })

        self._advance_time(cost.time_s if cost else 0.0)
        next_state = self._encode_state()
        done = self._check_day_end()

        transition = Transition(
            state=state_np,
            action_type=int(AgentActionType.MOVE_CONTAINER),
            container_pos=action.container_pos,
            dest_type=int(dest_type) if dest_type is not None else -1,
            placement_pos=action.placement_pos,
            vehicle_idx=action.vehicle_idx,
            reward=reward,
            next_state=next_state,
            done=done,
        )

        return HierarchicalStepResult(
            next_state=next_state, reward=reward, done=done,
            info=info, transition=transition, was_valid_move=True,
        )

    # ── Placement validators ──────────────────────────────────

    def _validate_yard_placement(
        self,
        container: Any,
        record: Any,
        action: ActionResult,
    ) -> Optional[Move]:
        """Validate a yard-to-yard restack. Returns Move or None."""
        if action.placement_pos is None:
            return None
        pr, ps, pt = action.placement_pos

        # Bounds check
        if not (0 <= pr < self.yard.n_rows
                and 0 <= ps < self.yard.total_splits
                and 0 <= pt < self.yard.n_tiers):
            return None

        # Container width in splits
        n_splits = self.yard.container_length_map.get(
            getattr(container, "length_ft", 0), 0
        )
        if n_splits <= 0:
            return None

        end_split = min(ps + n_splits, self.yard.total_splits)
        if end_split - ps < n_splits:
            return None  # overflows yard

        # All target splits must be free
        if self.yard.occupancy_mask[pt, pr, ps:end_split].any():
            return None

        # Support: tier > 0 requires occupied splits below
        if pt > 0 and not self.yard.occupancy_mask[pt - 1, pr, ps:end_split].all():
            return None

        # Reject same-position move
        pl = record.placement
        src_abs = pl.bay * self.yard.split_factor + pl.start_split
        if pr == pl.row and ps == src_abs and pt == pl.tier:
            return None

        p_bay = ps // self.yard.split_factor
        p_start = ps % self.yard.split_factor
        return Move(
            type=MoveType.YARD_TO_YARD,
            args={
                "container_id": container.container_id,
                "placement": PlacementResult(
                    row=pr, bay=p_bay, tier=pt, start_split=p_start,
                ),
            },
        )

    def _validate_train_load(
        self,
        container: Any,
        action: ActionResult,
        vehicle_list: List,
    ) -> Optional[Move]:
        """Validate a yard-to-train load. Returns Move or None."""
        idx = action.vehicle_idx
        if idx < 0 or idx >= len(vehicle_list):
            return None
        veh = vehicle_list[idx]
        if not hasattr(veh, "train_id"):
            return None
        if not veh.has_space_for_container(container):
            return None
        return Move(
            type=MoveType.YARD_TO_TRAIN,
            args={"container_id": container.container_id, "train_id": veh.train_id},
        )

    def _validate_truck_load(
        self,
        container: Any,
        action: ActionResult,
        vehicle_list: List,
    ) -> Optional[Move]:
        """Validate a yard-to-truck load. Returns Move or None."""
        idx = action.vehicle_idx
        if idx < 0 or idx >= len(vehicle_list):
            return None
        veh = vehicle_list[idx]
        if not hasattr(veh, "truck_id"):
            return None
        if not veh.can_accommodate_container(container):
            return None
        return Move(
            type=MoveType.YARD_TO_TRUCK,
            args={"container_id": container.container_id, "truck_id": veh.truck_id},
        )

    # ================================================================
    # Parking execution
    # ================================================================

    def _execute_parking_action(
        self,
        state_np: NDArray,
        action: ActionResult,
        parking_list: List,
        crane_id: int,
        info: Dict,
    ) -> HierarchicalStepResult:
        """Execute a parking action chosen by the agent."""
        idx = action.parking_idx
        if idx < 0 or idx >= len(parking_list):
            return self._idle_step(info)

        park_action: ParkingAction = parking_list[idx]
        truck = self.trucks.get(park_action.truck_id)

        reward = 0.0
        if truck and self.parking:
            success = self.parking.allocate(
                truck, park_action.spot.bay, park_action.spot.split,
            )
            if success:
                # Reward scales with proximity to pickup containers
                base_reward = 0.1
                if park_action.preferred_bay is not None:
                    dist = abs(park_action.delta_bay)
                    proximity_bonus = max(0.0, 0.5 * (1.0 - dist / 5.0))
                else:
                    proximity_bonus = 0.0
                reward = base_reward + proximity_bonus
                info["executed"].append({
                    "move_type": "SLOT_PARKING",
                    "truck_id": park_action.truck_id,
                    "bay": park_action.spot.bay,
                    "delta_bay": park_action.delta_bay,
                })

        self._advance_time()
        next_state = self._encode_state()
        done = self._check_day_end()

        transition = Transition(
            state=state_np,
            action_type=int(AgentActionType.SLOT_PARKING),
            parking_idx=idx,
            reward=reward,
            next_state=next_state,
            done=done,
        )

        return HierarchicalStepResult(
            next_state=next_state, reward=reward, done=done,
            info=info, transition=transition, was_valid_move=True,
        )

    # ── Convenience wrappers ──────────────────────────────────

    def _idle_step(self, info: Dict) -> HierarchicalStepResult:
        """Advance time with zero reward (nothing to do)."""
        self._advance_time()
        return HierarchicalStepResult(
            next_state=self._encode_state(), reward=0.0,
            done=self._check_day_end(), info=info, was_valid_move=False,
        )

    # ================================================================
    # Auto-import (train->yard, truck->yard)
    # ================================================================
    # Uses yard.find_single_placement directly — avoids TLM list methods
    # that call search_placements (missing on OptimizedStorageYard).

    def _try_auto_transfer(
        self, crane_id: int, info: Dict,
    ) -> Optional[HierarchicalStepResult]:
        """Execute one pending import or export automatically.

        Priority:
        1. Export yard→train  (trains have departure deadlines)
        2. Export yard→truck  (reduce truck waiting time)
        3. Import train→yard
        4. Import truck→yard
        Returns None when nothing to transfer (agent takes over).
        """
        # ── Exports ───────────────────────────────────────────
        for train_id, train in self.trains.items():
            moves = self.tlm.list_yard_to_train(train)
            if moves:
                return self._exec_auto_move(
                    moves[0], moves[0].args["container_id"], crane_id, info,
                )

        for truck_id, truck in self.trucks.items():
            moves = self.tlm.list_yard_to_truck(truck)
            if moves:
                return self._exec_auto_move(
                    moves[0], moves[0].args["container_id"], crane_id, info,
                )

        # ── Imports ───────────────────────────────────────────
        for train_id, train in self.trains.items():
            for container in train.get_all_containers():
                if getattr(container, "direction", "") != "Import":
                    continue
                anchor = self.tlm._goods_anchor(container)
                placement = self.yard.find_single_placement(
                    container, target_bay=anchor,
                )
                if placement is None:
                    continue
                move = Move(
                    type=MoveType.TRAIN_TO_YARD,
                    args={
                        "train_id": train_id,
                        "container_id": container.container_id,
                        "placement": placement,
                    },
                )
                return self._exec_auto_move(
                    move, container.container_id, crane_id, info,
                )

        for truck_id, truck in self.trucks.items():
            if not truck.parking_spot or not truck.containers:
                continue
            for container in list(truck.containers):
                anchor = self.tlm._goods_anchor(container)
                placement = self.yard.find_single_placement(
                    container, target_bay=anchor,
                )
                if placement is None:
                    continue
                move = Move(
                    type=MoveType.TRUCK_TO_YARD,
                    args={
                        "truck_id": truck_id,
                        "container_id": container.container_id,
                        "placement": placement,
                    },
                )
                return self._exec_auto_move(
                    move, container.container_id, crane_id, info,
                )

        return None  # nothing to transfer

    def _exec_auto_move(
        self, move: Move, container_id: str, crane_id: int, info: Dict,
    ) -> HierarchicalStepResult:
        """Execute an auto-transfer move (import or export).

        Cost is computed BEFORE execution because exports remove the
        container from the yard, making endpoint resolution impossible
        after the fact.
        """
        # Compute cost before execution (yard placement still exists)
        epc = self.rmgc.endpoints_and_cost_for_move(
            move, self.trains, self.trucks, self.yard,
        )
        cost = epc[2] if epc else None

        try:
            self.tlm.execute(move, self.trains, self.trucks, self.terminal_trucks)
        except Exception as e:
            info.setdefault("errors", []).append(str(e))
            return self._idle_step(info)

        if crane_id < len(self.cranes) and cost:
            self.cranes[crane_id].busy_until = (
                self.current_time + timedelta(seconds=cost.time_s)
            )

        reward = self.reward_engine.immediate_reward(
            move.type.value,
            cost.distance_m if cost else 0.0,
            cost.time_s if cost else 0.0,
        )
        info["executed"].append({
            "move_type": move.type.value,
            "container_id": container_id,
            "auto_transfer": True,
        })

        self._advance_time(cost.time_s if cost else 0.0)
        next_state = self._encode_state()
        done = self._check_day_end()

        return HierarchicalStepResult(
            next_state=next_state, reward=reward, done=done,
            info=info, transition=None, was_valid_move=True,
        )

    # ================================================================
    # Feature builders for MultiHead agent
    # ================================================================

    def _build_vehicle_features(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List]:
        """Build vehicle feature arrays for trains + trucks.

        Returns (feats (V, Fv), mask (V,), ordered_vehicle_list).
        """
        vehicles: List = []
        feats_list: List[np.ndarray] = []
        now = self.current_time

        # Trains — use proper Train API methods
        for train_id, train in self.trains.items():
            anchor = self.rail.get_anchor_bay(train_id)
            dep = train.departure_time
            hours_left = (
                (dep - now).total_seconds() / 3600.0 if dep and dep > now else 0.0
            )
            total = train.get_container_count()
            total_wagons = len(train.wagons)
            pickup_pending = len(train.get_all_pickup_container_ids())
            feats_list.append(np.array([
                1.0,                                                     # is_train
                (anchor or 0) / max(self.yard.n_bays - 1, 1),           # bay_norm
                total / max(total_wagons * 2, 1),                        # frac_loaded
                min(hours_left / MAX_DEPARTURE_HOURS, 1.0),             # time_left
                1.0 if pickup_pending > 0 else 0.0,                     # has_export_demand
                1.0 if not train.wagons_with_space else 0.0,            # full flag (inverted)
                min(total / 50.0, 1.0),                                  # count_norm
                1.0 - min(hours_left / MAX_DEPARTURE_HOURS, 1.0),       # urgency
            ], dtype=np.float32))
            vehicles.append(train)

        # Trucks — use proper Truck attributes
        for truck_id, truck in self.trucks.items():
            has_delivery = bool(truck.containers)
            has_pickup = bool(truck.pickup_container_ids)
            feats_list.append(np.array([
                0.0,                                                     # is_train=0
                0.5,                                                     # bay_norm (unknown)
                len(truck.containers) / 2.0,                             # frac_loaded
                0.5,                                                     # time_left
                1.0 if has_delivery else 0.0,                            # has_delivery
                1.0 if has_pickup else 0.0,                              # has_pickup_demand
                0.0,
                0.5,
            ], dtype=np.float32))
            vehicles.append(truck)

        if not vehicles:
            empty_feats = np.zeros((0, VEHICLE_FEAT_DIM), dtype=np.float32)
            empty_mask = np.zeros(0, dtype=bool)
            return empty_feats, empty_mask, []

        feats = np.stack(feats_list)                             # (V, 8)
        mask = np.ones(len(vehicles), dtype=bool)                # all valid
        return feats, mask, vehicles

    # Maximum candidate spots per truck for ParkingHead
    _PARKING_CANDIDATES_PER_TRUCK: int = 5
    _PARKING_SEARCH_RADIUS: int = 5  # bays each side of preferred

    def _build_parking_features(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List]:
        """Build parking feature arrays with MULTIPLE spots per truck.

        Each pickup truck collects exactly ONE container.  We generate
        up to _PARKING_CANDIDATES_PER_TRUCK spots at varying distances
        from that container's bay so the ParkingHead learns to minimise
        crane travel.

        Returns (feats (P, 8), mask (P,), action_list).
        """
        actions: List = []
        feats_list: List[np.ndarray] = []
        n_bays = max(self.yard.n_bays, 1)

        for truck_id, truck in self.trucks.items():
            if truck.parking_spot:
                continue

            # Single pickup container
            target_bay, target_tier, accessible = self._pickup_info(truck)
            center = target_bay if target_bay is not None else n_bays // 2
            has_target = target_bay is not None

            radius = self._PARKING_SEARCH_RADIUS
            candidates_found = 0

            for offset in range(radius + 1):
                for sign in ([0] if offset == 0 else [-1, 1]):
                    bay = center + offset * (sign if sign else 1)
                    if bay < 0 or bay >= n_bays:
                        continue
                    spot = next(
                        self.parking.iter_free_in_bay_range(bay, bay), None,
                    )
                    if spot is None:
                        continue

                    delta = spot.bay - center
                    from simulation.rl.features.featurizers import ParkingAction
                    pa = ParkingAction(
                        truck_id=truck_id,
                        spot=spot,
                        preferred_bay=target_bay,
                        delta_bay=delta,
                    )
                    actions.append(pa)
                    feats_list.append(np.array([
                        spot.bay / max(n_bays - 1, 1),            # bay_norm
                        center / max(n_bays - 1, 1),              # target_bay_norm
                        abs(delta) / max(n_bays, 1),              # dist_from_target
                        1.0 if delta == 0 else 0.0,               # is_optimal
                        1.0 if accessible else 0.0,               # pickup_accessible
                        (target_tier or 0) / max(self.yard.n_tiers - 1, 1),  # tier_norm (0=ground=easy)
                        1.0 if truck.containers else 0.0,         # has_delivery
                        1.0 if has_target else 0.0,               # has_target
                    ], dtype=np.float32))

                    candidates_found += 1
                    if candidates_found >= self._PARKING_CANDIDATES_PER_TRUCK:
                        break
                if candidates_found >= self._PARKING_CANDIDATES_PER_TRUCK:
                    break

        if not actions:
            empty_feats = np.zeros((0, VEHICLE_FEAT_DIM), dtype=np.float32)
            empty_mask = np.zeros(0, dtype=bool)
            return empty_feats, empty_mask, []

        feats = np.stack(feats_list)
        mask = np.ones(len(actions), dtype=bool)
        return feats, mask, actions

    def _pickup_info(self, truck) -> Tuple[Optional[int], Optional[int], bool]:
        """Get (bay, tier, is_accessible) for truck's single pickup container."""
        pickup_ids = getattr(truck, "pickup_container_ids", set())
        if not pickup_ids:
            return None, None, False
        cid = next(iter(pickup_ids))
        pl = self.yard.get_placement(cid)
        if pl is None:
            return None, None, False
        accessible = self.yard.is_accessible(cid)
        return pl.bay, pl.tier, accessible

    def _container_at_position(
        self, row: int, split: int, tier: int,
    ) -> Optional[Any]:
        """Look up container record at (row, split, tier) in yard."""
        if not (0 <= tier < self.yard.n_tiers
                and 0 <= row < self.yard.n_rows
                and 0 <= split < self.yard.total_splits):
            return None
        from simulation.core.facilities.yard import EMPTY_SLOT
        idx = self.yard.position_grid[tier, row, split]
        if idx == EMPTY_SLOT:
            return None
        return self.yard._records[idx]

    # ================================================================
    # Move helpers
    # ================================================================

    _MOVE_TYPE_MAP = {
        (SourceType.YARD, DestinationType.YARD): MoveType.YARD_TO_YARD,
        (SourceType.YARD, DestinationType.TRAIN): MoveType.YARD_TO_TRAIN,
        (SourceType.YARD, DestinationType.TRUCK): MoveType.YARD_TO_TRUCK,
        (SourceType.TRAIN, DestinationType.YARD): MoveType.TRAIN_TO_YARD,
        (SourceType.TRAIN, DestinationType.TRUCK): MoveType.TRAIN_TO_TRUCK,
        (SourceType.TRUCK, DestinationType.YARD): MoveType.TRUCK_TO_YARD,
        (SourceType.TRUCK, DestinationType.TRAIN): MoveType.TRUCK_TO_TRAIN,
    }


    @classmethod
    def _resolve_move_type(cls, source: SourceType, dest: DestinationType) -> MoveType:
        return cls._MOVE_TYPE_MAP.get((source, dest), MoveType.YARD_TO_YARD)

    @staticmethod
    def _build_move_args(cont: MoveableContainer, dest: Destination) -> Dict[str, Any]:
        """Build move args dictionary for TLM."""
        args: Dict[str, Any] = {"container_id": cont.container_id}

        if cont.source_type == SourceType.TRAIN:
            args["train_id"] = cont.source_id
        elif cont.source_type == SourceType.TRUCK:
            args["truck_id"] = cont.source_id

        if dest.dest_type == DestinationType.YARD:
            args["placement"] = PlacementResult(
                row=dest.row, bay=dest.bay, tier=dest.tier,
                start_split=dest.start_split,
            )
        elif dest.dest_type == DestinationType.TRAIN:
            args["train_id"] = dest.dest_id
        elif dest.dest_type == DestinationType.TRUCK:
            args["truck_id"] = dest.dest_id

        return args

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
                if self.stats:
                    self.stats.on_train_arrival(st.train)

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
                if self.stats:
                    self.stats.on_truck_arrival(truck)

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
            train = self._departed_cache.get(train_id)
            if train:
                reward += self.reward_engine.on_train_departure(train)
                if self.stats:
                    self.stats.on_train_departure(train_id, _leftover)
        self._update_train_heat()
        return reward

    def _collect_truck_departures(self) -> List[Dict[str, Any]]:
        """Remove departed trucks, return event dicts."""
        events = []
        to_remove = []
        for truck_id, truck in self.trucks.items():
            if truck.is_ready_to_depart():
                if truck.departure_time is None:
                    truck.departure_time = self.current_time
                wait_min = 0.0
                if truck.arrival_time:
                    wait_min = (truck.departure_time - truck.arrival_time).total_seconds() / 60.0
                events.append({"truck_id": truck_id, "wait_min": wait_min})
                if self.stats:
                    self.stats.on_truck_departure(truck=truck, wait_minutes=wait_min)
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
    # State encoding
    # ================================================================

    def _encode_state(self) -> NDArray[np.float32]:
        """Single point for state encoding — (C, R, S, T) at split resolution."""
        return self.encoder.encode(
            self.trains, self.trucks, self.current_time,
        )

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
        """Initialise terminal trucks for the day."""
        self.terminal_trucks.clear()
        for i in range(TERMINAL_TRUCK_COUNT):
            tt = TerminalTruck()
            if not getattr(tt, "truck_id", None):
                from simulation.utils.id_generator import IDGenerator
                setattr(tt, "truck_id", IDGenerator.generate_terminal_truck_id(i))
            self.terminal_trucks[tt.truck_id] = tt

    def _complete_terminal_truck_jobs(self) -> None:
        """Complete terminal truck jobs that have finished."""
        for tt_id, busy_until in list(self._tt_busy_until.items()):
            if busy_until and busy_until <= self.current_time:
                tt = self.terminal_trucks.get(tt_id)
                if tt and hasattr(tt, "complete_job"):
                    tt.complete_job()
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