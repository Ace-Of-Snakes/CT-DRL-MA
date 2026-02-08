# simulation/env/unified_env.py
"""Unified spatial environment — replaces multihead stepping.

Changes from env.py:
  - State encoded via UnifiedStateEncoder (10ch, R=15)
  - Agent stepping is 2-stage: source → dest (no hierarchical heads)
  - Move type inferred from spatial regions
  - Truck parking is an explicit agent decision (QUEUE → PARKING)
  - No vehicle feature arrays or parking feature arrays
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from simulation.core.facilities.yard import OptimizedStorageYard, PlacementResult
from simulation.core.facilities.parking import OptimizedParkingArea
from simulation.core.facilities.railyard import OptimizedRailYard
from simulation.core.vehicles.train import Train
from simulation.core.vehicles.truck import Truck
from simulation.core.enums import MoveType, TruckStatus
from simulation.operations.terminal_manager import TerminalLogisticsManager, Move

# Parent env — inherits all simulation infrastructure
from simulation.env.env import ContainerTerminalEnv

# Unified components (Phase 1-3 deliverables)
from simulation.env.unified_state_encoder import UnifiedStateEncoder, CH
from simulation.rl.multihead_dqn.config import MultiHeadDQNConfig, UnifiedDims
from simulation.rl.multihead_dqn.unified_agent import (
    UnifiedDQNAgent, UnifiedActionResult, resolve_move_type,
)
from simulation.rl.multihead_dqn.unified_replay_buffer import UnifiedTransition


# ── Constants ────────────────────────────────────────────────────────────

# Smallest common container — conservative for yard validity masks
MIN_CONTAINER_SPLITS: int = 10

# Parking reward shaping (plan §12.4)
PARKING_REWARD_BASE: float = 0.1
PARKING_PROXIMITY_BONUS_MAX: float = 0.5
PARKING_PROXIMITY_RADIUS_BAYS: int = 5

# Channel indices (from CH enum)
_CH_DIRECTION: int = CH.DIRECTION
_DIRECTION_EXPORT_THRESHOLD: float = 0.5


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass
class UnifiedStepResult:
    """Result of one unified step for a single crane."""
    next_state: NDArray[np.float32]
    reward: float
    done: bool
    info: Dict[str, Any]
    transition: Optional[UnifiedTransition] = None
    was_valid_move: bool = False


# ══════════════════════════════════════════════════════════════════════════
# Unified Environment
# ══════════════════════════════════════════════════════════════════════════

class UnifiedContainerTerminalEnv(ContainerTerminalEnv):
    """Extended env with unified spatial stepping.

    Inherits all infrastructure (arrivals, departures, cranes, RMGC, etc.)
    from ContainerTerminalEnv. Overrides state encoding and stepping to use
    the 2-head unified agent.
    """

    def __init__(self, *args, dims: Optional[UnifiedDims] = None, **kwargs):
        # Force auto_park off — agent handles parking explicitly
        kwargs["auto_park"] = False
        super().__init__(*args, **kwargs)

        self.dims = dims or UnifiedDims(
            n_yard_rows=self.yard.n_rows,
            n_bays=self.yard.n_bays,
            split_factor=self.yard.split_factor,
            n_tiers=self.yard.n_tiers,
        )
        self.unified_encoder = UnifiedStateEncoder(
            self.yard, self.rail, self.parking, self.dims,
        )
        self._s_stride: int = 4  # must match CNNConfig.s_stride
        self._s_down: int = self.dims.n_splits // self._s_stride

    # ── State encoding override ───────────────────────────────────────

    def _encode_state(self) -> NDArray[np.float32]:
        """Unified state: (C, R_uni, S, T)."""
        return self.unified_encoder.encode(
            self.trains, self.trucks, self.current_time,
        )

    # ══════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════

    def step_all_cranes(
        self, agent: UnifiedDQNAgent,
    ) -> Tuple[NDArray[np.float32], float, bool, Dict[str, Any]]:
        """Execute unified steps for all idle cranes.

        Returns (state, total_reward, done, info). Transitions are in
        info["transitions"] — caller (TutorialRunner) handles remember().
        """
        total_reward = 0.0
        all_info: Dict[str, Any] = {
            "crane_results": [], "executed": [], "transitions": [],
        }
        done = False

        departure_reward = self._handle_arrivals_departures()
        total_reward += departure_reward

        now = self.current_time
        idle_cranes = [
            c for c in self.cranes
            if c.busy_until is None or c.busy_until <= now
        ]

        state = self._encode_state()

        if not idle_cranes:
            # All cranes busy — fast-forward to next available
            busy_times = [
                c.busy_until for c in self.cranes if c.busy_until is not None
            ]
            if busy_times:
                next_time = min(busy_times)
                advance_min = (next_time - now).total_seconds() / 60.0
                self.current_time = next_time
                self._complete_terminal_truck_jobs()
                total_reward += self.reward_engine.waiting_penalty(
                    len(self.trucks), advance_min,
                )
            else:
                self._advance_time()

            state = self._encode_state()
            done = self._check_day_end()
            return state, total_reward, done, all_info

        # Track containers moved this step to prevent crane-to-crane reversals
        self._step_blacklist: set = set()

        for crane in idle_cranes:
            result = self._step_unified(agent, crane.id, state)
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

            if result.done:
                done = True
                break

        truck_events = self._collect_truck_departures()
        all_info["truck_departures"] = truck_events

        if not done:
            done = self._check_day_end()
            if done:
                total_reward += self.reward_engine.end_of_day_penalty(
                    self.current_time,
                )
                self._rollover_missed_deadlines()

        return state, total_reward, done, all_info

    # ══════════════════════════════════════════════════════════════════
    # Core unified step
    # ══════════════════════════════════════════════════════════════════

    def _step_unified(
        self,
        agent: UnifiedDQNAgent,
        crane_id: int,
        state_np: NDArray[np.float32],
    ) -> UnifiedStepResult:
        """One unified step: source selection → dest selection → execute."""
        info: Dict[str, Any] = {
            "crane_id": crane_id, "executed": [], "retries": 0,
        }
        total_penalty = 0.0

        for retry in range(self.max_retries + 1):
            info["retries"] = retry

            source_mask = self.unified_encoder.get_source_mask(
                self.trains, self.trucks,
            )
            # Exclude containers moved by earlier cranes this step
            blacklist = getattr(self, "_step_blacklist", set())
            if blacklist:
                self._apply_source_blacklist(source_mask, blacklist)
            if not source_mask.any():
                break  # nothing to do

            dest_mask_fn = self._make_dest_mask_fn(state_np)

            action = agent.act(
                state=state_np,
                source_mask=source_mask,
                dest_mask_fn=dest_mask_fn,
            )

            if action.move_type is None or action.dest_pos is None:
                break  # no valid action — idle

            if action.move_type == "PARK_TRUCK":
                result = self._execute_park_truck(
                    state_np, action, crane_id, info,
                )
                result.reward += total_penalty
                return result

            # Container move
            result = self._execute_unified_move(
                state_np, action, crane_id, info,
            )
            if result is not None:
                result.reward += total_penalty
                return result

            # Invalid → penalty, retry
            total_penalty += self.no_destination_penalty
            info.setdefault("retry_reasons", []).append("invalid_placement")

        # All retries exhausted or nothing to do — idle advance
        self._advance_time()
        return UnifiedStepResult(
            next_state=self._encode_state(),
            reward=total_penalty,
            done=self._check_day_end(),
            info=info, was_valid_move=False,
        )

    # ══════════════════════════════════════════════════════════════════
    # Dest mask construction
    # ══════════════════════════════════════════════════════════════════

    def _make_dest_mask_fn(
        self, state: np.ndarray,
    ) -> Callable[[int, int, int], np.ndarray]:
        """Build closure returning (R_uni, S_down, T) dest mask per source.

        Pre-computes yard validity and parking free masks (expensive)
        once, then filters per-source via region logic.
        """
        d = self.dims
        stride = self._s_stride
        S_down = self._s_down

        # Pre-compute shared masks
        yard_valid_full = self.unified_encoder.get_yard_validity_mask(
            MIN_CONTAINER_SPLITS,
        )
        yard_valid_down = _downsample_mask(yard_valid_full, stride)
        parking_free_down = self._parking_free_mask_down()

        def dest_mask_fn(
            src_row: int, src_split: int, src_tier: int,
        ) -> np.ndarray:
            region = d.region_of(src_row)
            mask = np.zeros((d.R_unified, S_down, d.n_tiers), dtype=bool)

            if region == "QUEUE":
                # Queue → Parking: free parking spots
                mask[d.parking_row_start, :, 0] = parking_free_down

            elif region in ("RAIL", "PARKING"):
                # Rail/Parking → Yard: valid yard placements
                mask[d.yard_row_start:d.yard_row_end] = (
                    yard_valid_down[d.yard_row_start:d.yard_row_end]
                )

            elif region == "YARD":
                # Yard → Yard: restack targets
                mask[d.yard_row_start:d.yard_row_end] = (
                    yard_valid_down[d.yard_row_start:d.yard_row_end]
                )
                # Exclude source position
                src_s_down = src_split // stride
                if d.yard_row_start <= src_row < d.yard_row_end:
                    mask[src_row, src_s_down, src_tier] = False

                # Direction-dependent vehicle targets
                direction = state[_CH_DIRECTION, src_row, src_split, src_tier]
                container_rec = self._yard_container_at_unified(
                    src_row, src_split, src_tier,
                )

                if container_rec is not None:
                    container = container_rec.container
                    if direction > _DIRECTION_EXPORT_THRESHOLD:
                        # Export → matching train wagons
                        self._add_rail_dest_mask(
                            mask, container, d, S_down, stride,
                        )
                    else:
                        # Import → matching pickup truck in parking
                        self._add_truck_dest_mask(
                            mask, container, d, S_down, stride,
                        )

            return mask

        return dest_mask_fn

    def _parking_free_mask_down(self) -> np.ndarray:
        """(S_down,) bool: True where a free parking spot exists in stride window."""
        S_down = self._s_down
        stride = self._s_stride
        S = self.dims.n_splits

        if self.parking is None:
            return np.zeros(S_down, dtype=bool)

        # parking.occupied is (n_bays, split_factor) — flatten to splits
        free = ~self.parking.occupied
        free_flat = free.reshape(-1)

        # Pad/trim to exactly S
        if len(free_flat) < S:
            padded = np.zeros(S, dtype=bool)
            padded[:len(free_flat)] = free_flat
            free_flat = padded
        else:
            free_flat = free_flat[:S]

        # Downsample: any free in stride window → True
        trimmed = free_flat[:S_down * stride]
        return trimmed.reshape(S_down, stride).any(axis=1)

    def _add_rail_dest_mask(
        self,
        mask: np.ndarray,
        container: Any,
        d: UnifiedDims,
        S_down: int,
        stride: int,
    ) -> None:
        """Mark rail positions for trains wanting this export container."""
        cid = container.container_id
        sf = d.split_factor

        for train_id, train in self.trains.items():
            if cid not in train.get_all_pickup_container_ids():
                continue
            if not train.has_space_for_container(container):
                continue

            slot = self.rail.get_slot(train_id)
            if slot is None:
                continue

            track_row = slot.track_id
            if track_row >= d.rail_row_end:
                continue

            anchor = slot.anchor_bay
            for wagon_idx, wagon in enumerate(train.wagons):
                if not _wagon_has_space(wagon):
                    continue
                wagon_bay = anchor + wagon_idx
                if wagon_bay < 0 or wagon_bay >= d.n_bays:
                    continue
                s_start = wagon_bay * sf
                sd_start = s_start // stride
                sd_end = min(sd_start + max(sf // stride, 1), S_down)
                mask[track_row, sd_start:sd_end, 0] = True

    def _add_truck_dest_mask(
        self,
        mask: np.ndarray,
        container: Any,
        d: UnifiedDims,
        S_down: int,
        stride: int,
    ) -> None:
        """Mark parking positions for trucks wanting this import container."""
        cid = container.container_id
        sf = d.split_factor

        for truck_id, truck in self.trucks.items():
            pickup_ids = getattr(truck, "pickup_container_ids", set())
            if cid not in pickup_ids:
                continue
            if not truck.can_accommodate_container(container):
                continue

            spot = getattr(truck, "parking_spot", None)
            if spot is None:
                continue

            s = spot.bay * sf + (spot.split or 0)
            s_down = s // stride
            if 0 <= s_down < S_down:
                mask[d.parking_row_start, s_down, 0] = True

    # ══════════════════════════════════════════════════════════════════
    # Execution: PARK_TRUCK (queue → parking)
    # ══════════════════════════════════════════════════════════════════

    def _execute_park_truck(
        self,
        state_np: NDArray,
        action: UnifiedActionResult,
        crane_id: int,
        info: Dict,
    ) -> UnifiedStepResult:
        """Park a queued truck at the agent-chosen parking spot."""
        d = self.dims
        sf = d.split_factor

        # Resolve source: queued truck
        truck = self._find_queued_truck_at(
            action.source_pos[0], action.source_pos[1],
        )
        if truck is None:
            return self._idle_result(info)

        if self.parking is None:
            return self._idle_result(info)

        # Resolve destination: parking (bay, split_offset)
        dst_split = action.dest_pos[1]
        dst_bay = dst_split // sf
        dst_split_offset = dst_split % sf

        success = self.parking.allocate(truck, dst_bay, dst_split_offset)
        if not success:
            return self._idle_result(info)

        # Proximity reward: bonus for parking near goods anchor
        preferred = self.unified_encoder._preferred_bay_for_truck(truck)
        if preferred is not None:
            delta = abs(dst_bay - preferred)
            proximity_bonus = PARKING_PROXIMITY_BONUS_MAX * max(
                0.0, 1.0 - delta / PARKING_PROXIMITY_RADIUS_BAYS,
            )
        else:
            proximity_bonus = 0.0

        reward = PARKING_REWARD_BASE + proximity_bonus

        info["executed"].append({
            "move_type": "PARK_TRUCK",
            "truck_id": truck.truck_id,
            "bay": dst_bay,
            "proximity_bonus": round(proximity_bonus, 4),
        })

        # No crane cost for parking — just advance time
        self._advance_time()
        next_state = self._encode_state()
        done = self._check_day_end()

        transition = UnifiedTransition(
            state=state_np,
            source_pos_down=action.source_pos_down,
            dest_pos_down=action.dest_pos_down,
            reward=reward,
            next_state=next_state,
            done=done,
        )

        return UnifiedStepResult(
            next_state=next_state, reward=reward, done=done,
            info=info, transition=transition, was_valid_move=True,
        )

    # ══════════════════════════════════════════════════════════════════
    # Execution: container moves
    # ══════════════════════════════════════════════════════════════════

    def _execute_unified_move(
        self,
        state_np: NDArray,
        action: UnifiedActionResult,
        crane_id: int,
        info: Dict,
    ) -> Optional[UnifiedStepResult]:
        """Execute a container move. Returns None if invalid."""
        move = self._resolve_move(action)
        if move is None:
            return None

        # Compute RMGC cost BEFORE execution (container still in yard)
        epc = self.rmgc.endpoints_and_cost_for_move(
            move, self.trains, self.trucks, self.yard,
        )
        cost = epc[2] if epc else None

        try:
            success = self.tlm.execute(
                move, self.trains, self.trucks, self.terminal_trucks,
            )
        except Exception as e:
            info.setdefault("errors", []).append(str(e))
            return None

        if not success:
            return None

        # Crane timing
        time_s = cost.time_s if cost else 0.0
        if crane_id < len(self.cranes) and cost:
            self.cranes[crane_id].busy_until = (
                self.current_time + timedelta(seconds=time_s)
            )

        distance_m = cost.distance_m if cost else 0.0
        reward = self.reward_engine.immediate_reward(
            move.type.value, distance_m, time_s,
        )

        container_id = move.args.get("container_id", "")
        info["executed"].append({
            "move_type": move.type.value,
            "container_id": container_id,
            "distance_m": round(distance_m, 2),
            "time_s": round(time_s, 2),
        })

        # Prevent other cranes from reversing this move in the same step
        blacklist = getattr(self, "_step_blacklist", None)
        if blacklist is not None and container_id:
            blacklist.add(container_id)

        self._advance_time(time_s)
        next_state = self._encode_state()
        done = self._check_day_end()

        transition = UnifiedTransition(
            state=state_np,
            source_pos_down=action.source_pos_down,
            dest_pos_down=action.dest_pos_down,
            reward=reward,
            next_state=next_state,
            done=done,
        )

        return UnifiedStepResult(
            next_state=next_state, reward=reward, done=done,
            info=info, transition=transition, was_valid_move=True,
        )

    # ══════════════════════════════════════════════════════════════════
    # Move resolution: spatial coordinates → Move
    # ══════════════════════════════════════════════════════════════════

    _MOVE_DISPATCH = {
        "YARD_TO_YARD":   "_resolve_yard_to_yard",
        "YARD_TO_TRAIN":  "_resolve_yard_to_train",
        "YARD_TO_TRUCK":  "_resolve_yard_to_truck",
        "TRAIN_TO_YARD":  "_resolve_rail_to_yard",
        "TRUCK_TO_YARD":  "_resolve_parking_to_yard",
    }

    def _resolve_move(self, action: UnifiedActionResult) -> Optional[Move]:
        """Convert unified spatial action to a TLM Move."""
        mt = action.move_type
        handler_name = self._MOVE_DISPATCH.get(mt)
        if handler_name is None:
            return None
        return getattr(self, handler_name)(action)

    def _resolve_yard_to_yard(self, action: UnifiedActionResult) -> Optional[Move]:
        """Restack: yard container → different yard position."""
        record = self._yard_container_at_unified(*action.source_pos)
        if record is None:
            return None

        container = record.container
        dst_row, dst_split, dst_tier = action.dest_pos
        d = self.dims

        yard_row = dst_row - d.yard_row_start
        if not (0 <= yard_row < d.n_yard_rows):
            return None

        # Validate span fits
        n_splits = self.yard.container_length_map.get(
            getattr(container, "length_ft", 0), 0,
        )
        if n_splits <= 0 or dst_split + n_splits > d.n_splits:
            return None

        # Occupancy + support checks
        occ = self.yard.occupancy_mask
        if occ[dst_tier, yard_row, dst_split:dst_split + n_splits].any():
            return None
        if dst_tier > 0:
            if not occ[dst_tier - 1, yard_row, dst_split:dst_split + n_splits].all():
                return None

        # Reject no-op (same position)
        pl = record.placement
        src_abs = pl.bay * self.yard.split_factor + pl.start_split
        if yard_row == pl.row and dst_split == src_abs and dst_tier == pl.tier:
            return None

        dst_bay = dst_split // self.yard.split_factor
        dst_start = dst_split % self.yard.split_factor
        return Move(
            type=MoveType.YARD_TO_YARD,
            args={
                "container_id": container.container_id,
                "placement": PlacementResult(
                    row=yard_row, bay=dst_bay, tier=dst_tier,
                    start_split=dst_start,
                ),
            },
        )

    def _resolve_yard_to_train(self, action: UnifiedActionResult) -> Optional[Move]:
        """Export: yard container → train."""
        record = self._yard_container_at_unified(*action.source_pos)
        if record is None:
            return None

        container = record.container
        if not _is_export(container):
            return None

        # Find train on dest track
        dst_row = action.dest_pos[0]
        if dst_row >= self.dims.rail_row_end:
            return None

        train = self._find_train_on_track(dst_row)
        if train is None:
            return None

        if container.container_id not in train.get_all_pickup_container_ids():
            return None
        if not train.has_space_for_container(container):
            return None

        return Move(
            type=MoveType.YARD_TO_TRAIN,
            args={
                "container_id": container.container_id,
                "train_id": train.train_id,
            },
        )

    def _resolve_yard_to_truck(self, action: UnifiedActionResult) -> Optional[Move]:
        """Import: yard container → pickup truck at parking."""
        record = self._yard_container_at_unified(*action.source_pos)
        if record is None:
            return None

        container = record.container
        if not _is_import(container):
            return None

        # Find truck at dest parking position
        truck = self._find_truck_at_parking_split(action.dest_pos[1])
        if truck is None:
            return None

        if container.container_id not in getattr(truck, "pickup_container_ids", set()):
            return None
        if not truck.can_accommodate_container(container):
            return None

        return Move(
            type=MoveType.YARD_TO_TRUCK,
            args={
                "container_id": container.container_id,
                "truck_id": truck.truck_id,
            },
        )

    def _resolve_rail_to_yard(self, action: UnifiedActionResult) -> Optional[Move]:
        """Import: train container → yard placement."""
        container, train = self._find_rail_container(
            action.source_pos[0], action.source_pos[1],
        )
        if container is None or train is None:
            return None

        dst_row, dst_split, dst_tier = action.dest_pos
        d = self.dims
        yard_row = dst_row - d.yard_row_start
        if not (0 <= yard_row < d.n_yard_rows):
            return None

        anchor_bay = dst_split // self.yard.split_factor
        placement = self.yard.find_single_placement(
            container, target_bay=anchor_bay,
        )
        if placement is None:
            return None

        return Move(
            type=MoveType.TRAIN_TO_YARD,
            args={
                "train_id": train.train_id,
                "container_id": container.container_id,
                "placement": placement,
            },
        )

    def _resolve_parking_to_yard(self, action: UnifiedActionResult) -> Optional[Move]:
        """Delivery: truck container → yard placement."""
        truck = self._find_truck_at_parking_split(action.source_pos[1])
        if truck is None or not getattr(truck, "containers", None):
            return None

        container = truck.containers[0]

        dst_row, dst_split, dst_tier = action.dest_pos
        d = self.dims
        yard_row = dst_row - d.yard_row_start
        if not (0 <= yard_row < d.n_yard_rows):
            return None

        anchor_bay = dst_split // self.yard.split_factor
        placement = self.yard.find_single_placement(
            container, target_bay=anchor_bay,
        )
        if placement is None:
            return None

        return Move(
            type=MoveType.TRUCK_TO_YARD,
            args={
                "truck_id": truck.truck_id,
                "container_id": container.container_id,
                "placement": placement,
            },
        )

    # ══════════════════════════════════════════════════════════════════
    # Entity resolution helpers
    # ══════════════════════════════════════════════════════════════════

    def _yard_container_at_unified(
        self, row: int, split: int, tier: int,
    ) -> Optional[Any]:
        """Look up container at unified (row, split, tier)."""
        d = self.dims
        yard_row = row - d.yard_row_start
        if not (0 <= yard_row < d.n_yard_rows):
            return None
        return self._container_at_position(yard_row, split, tier)

    def _find_queued_truck_at(
        self, queue_row: int, split: int,
    ) -> Optional[Truck]:
        """Find the truck encoded at (queue_row, split).

        Rebuilds queue layout (same algo as encoder) to find truck identity.
        """
        d = self.dims
        qi = queue_row - d.queue_row_start
        if qi < 0 or qi >= d.n_queue_rows:
            return None

        sf = d.split_factor

        # Rebuild queue layout: {queue_idx: {split_pos: Truck}}
        occupied: Dict[int, Dict[int, Truck]] = {
            i: {} for i in range(d.n_queue_rows)
        }
        for tk in self.trucks.values():
            if not self.unified_encoder._is_queued(tk):
                continue
            preferred = self.unified_encoder._preferred_bay_for_truck(tk)
            if preferred is None:
                preferred = d.n_bays // 2
            preferred = max(0, min(preferred, d.n_bays - 1))
            s = preferred * sf

            for q_idx in range(d.n_queue_rows):
                if s not in occupied[q_idx]:
                    occupied[q_idx][s] = tk
                    break

        return occupied.get(qi, {}).get(split)

    def _find_rail_container(
        self, track_row: int, split: int,
    ) -> Tuple[Optional[Any], Optional[Train]]:
        """Find container on train at (track_row, split)."""
        d = self.dims
        sf = d.split_factor

        if track_row >= d.rail_row_end:
            return None, None

        target_bay = split // sf

        for train_id, train in self.trains.items():
            slot = self.rail.get_slot(train_id)
            if slot is None or slot.track_id != track_row:
                continue

            wagon_idx = target_bay - slot.anchor_bay
            if wagon_idx < 0 or wagon_idx >= len(train.wagons):
                continue

            wagon = train.wagons[wagon_idx]
            for container in wagon.containers.values():
                return container, train

        return None, None

    def _find_truck_at_parking_split(self, split: int) -> Optional[Truck]:
        """Find truck parked at a given absolute split position."""
        if self.parking is None:
            return None

        sf = self.dims.split_factor
        bay = split // sf
        split_offset = split % sf

        if bay >= self.parking.n_bays or split_offset >= self.parking.split_factor:
            return None

        truck_id = self.parking.truck_ids[bay, split_offset]
        if truck_id is None:
            return None

        return self.trucks.get(truck_id)

    def _find_train_on_track(self, track_id: int) -> Optional[Train]:
        """Find the train assigned to a specific rail track."""
        for train_id, train in self.trains.items():
            slot = self.rail.get_slot(train_id)
            if slot is not None and slot.track_id == track_id:
                return train
        return None

    # ── Source blacklist (crane anti-reversal) ──────────────────────

    def _apply_source_blacklist(
        self, mask: np.ndarray, blacklist: set,
    ) -> None:
        """Zero out source mask positions for recently-moved containers.

        Prevents crane N+1 from reversing what crane N just did in the
        same step_all_cranes() call.
        """
        d = self.dims

        # Yard: check container IDs at marked positions
        for rec in self.yard.iter_records():
            if rec.container.container_id in blacklist:
                r = d.yard_row_start + rec.placement.row
                mask[r, rec.placement.abs_start, rec.placement.tier] = False

        # Rail: check containers on trains
        for train_id, train in self.trains.items():
            slot = self.rail.get_slot(train_id)
            if slot is None:
                continue
            track_row = self.unified_encoder._track_id_to_row(slot.track_id)
            if track_row is None or track_row >= d.rail_row_end:
                continue
            anchor = slot.anchor_bay
            for wagon_idx, wagon in enumerate(train.wagons):
                wagon_bay = anchor + wagon_idx
                if wagon_bay < 0 or wagon_bay >= d.n_bays:
                    continue
                split_cursor = 0
                for container in wagon.containers.values():
                    sf = self.dims.split_factor
                    n_splits = max(1, int(np.ceil(
                        getattr(container, "length_ft", 20) / (40.0 / sf)
                    )))
                    s0 = wagon_bay * sf + split_cursor
                    if container.container_id in blacklist:
                        mask[track_row, s0, 0] = False
                    split_cursor += n_splits

        # Parking: check trucks whose containers were just moved
        for tk in self.trucks.values():
            spot = getattr(tk, "parking_spot", None)
            if spot is None:
                continue
            containers = getattr(tk, "containers", None)
            if not containers:
                continue
            if any(c.container_id in blacklist for c in containers):
                bay = getattr(spot, "bay", None)
                split_offset = getattr(spot, "split", 0) or 0
                if bay is not None:
                    s = bay * self.dims.split_factor + split_offset
                    if 0 <= s < d.n_splits:
                        mask[d.parking_row_start, s, 0] = False

    # ── Idle / utility ────────────────────────────────────────────────

    def _idle_result(self, info: Dict) -> UnifiedStepResult:
        """Advance time, return zero-reward result."""
        self._advance_time()
        return UnifiedStepResult(
            next_state=self._encode_state(), reward=0.0,
            done=self._check_day_end(), info=info, was_valid_move=False,
        )


# ══════════════════════════════════════════════════════════════════════════
# Pure helper functions
# ══════════════════════════════════════════════════════════════════════════

def _downsample_mask(mask: np.ndarray, stride: int) -> np.ndarray:
    """Downsample (R, S, T) bool mask → (R, S_down, T) via max-pool."""
    R, S, T = mask.shape
    S_down = S // stride
    trimmed = mask[:, :S_down * stride, :]
    return trimmed.reshape(R, S_down, stride, T).any(axis=2)


def _wagon_has_space(wagon) -> bool:
    """Check if wagon has room for at least one more container."""
    capacity = getattr(wagon, "capacity", 2)
    current = len(getattr(wagon, "containers", {}))
    return current < capacity


def _is_export(container) -> bool:
    """True if container direction is Export."""
    d = getattr(container, "direction", None)
    if d is None:
        return False
    return (d.value if hasattr(d, "value") else str(d)) == "Export"


def _is_import(container) -> bool:
    """True if container direction is Import."""
    d = getattr(container, "direction", None)
    if d is None:
        return False
    return (d.value if hasattr(d, "value") else str(d)) == "Import"