# simulation/env/terminal_env.py
"""Spatial environment — replaces multihead stepping.

Changes from env.py:
  - State encoded via StateEncoder (10ch, R=15)
  - Agent stepping is 2-stage: source → dest (no hierarchical heads)
  - Move type inferred from spatial regions
  - Truck parking is an explicit agent decision (QUEUE → PARKING)
  - No vehicle feature arrays or parking feature arrays
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)

from simulation.core.facilities.yard import StorageYard, PlacementResult
from simulation.core.facilities.parking import ParkingArea
from simulation.core.facilities.railyard import RailYard
from simulation.core.vehicles.train import Train
from simulation.core.vehicles.truck import Truck
from simulation.core.enums import MoveType, TruckStatus, TerminalTruckStatus
from simulation.utils.direction_utils import is_import as _is_import, is_export as _is_export
from simulation.operations.terminal_manager import TerminalLogisticsManager, Move
from simulation.analytics.move_csv_logger import MoveCSVLogger
from simulation.env.facility_coordinator import FacilityCoordinator

# Parent env — inherits all simulation infrastructure
from simulation.env.env import ContainerTerminalEnv

# RL components
from simulation.env.state_encoder import StateEncoder, CH
from simulation.rl.multihead_dqn.config import MultiHeadDQNConfig, Dims, DEFAULT_S_STRIDE
from simulation.rl.multihead_dqn.agent import (
    DQNAgent, ActionResult, resolve_move_type,
)
from simulation.rl.multihead_dqn.replay_buffer import Transition


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

# Flat IDLE penalty — every deliberate IDLE costs the same small amount so the
# agent always prefers a productive move over waiting.
IDLE_PENALTY: float = -1.0


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass
class StepResult:
    """Result of one unified step for a single crane."""
    next_state: NDArray[np.float32]
    reward: float
    done: bool
    info: Dict[str, Any]
    transition: Optional[Transition] = None
    was_valid_move: bool = False


# ══════════════════════════════════════════════════════════════════════════
# Terminal Environment
# ══════════════════════════════════════════════════════════════════════════

class TerminalEnv(ContainerTerminalEnv):
    """Extended env with unified spatial stepping.

    Inherits all infrastructure (arrivals, departures, cranes, RMGC, etc.)
    from ContainerTerminalEnv. Overrides state encoding and stepping to use
    the 2-head unified agent.
    """

    def __init__(
        self,
        *args,
        dims: Optional[Dims] = None,
        log_dir: Optional[str] = None,
        **kwargs,
    ):
        # Force auto_park off — agent handles parking explicitly
        kwargs["auto_park"] = False
        super().__init__(*args, **kwargs)

        self.dims = dims or Dims(
            n_yard_rows=self.yard.n_rows,
            n_bays=self.yard.n_bays,
            split_factor=self.yard.split_factor,
            n_tiers=self.yard.n_tiers,
        )
        self.encoder = StateEncoder(
            self.yard, self.rail, self.parking, self.dims,
        )
        self.coordinator = FacilityCoordinator(
            self.yard, self.rail, self.parking, self.dims,
        )
        self._s_stride: int = DEFAULT_S_STRIDE
        self._s_down: int = self.dims.n_splits // self._s_stride

        # Optional CSV move logger (activated when log_dir is provided)
        self._log_dir = log_dir
        self.move_logger: Optional[MoveCSVLogger] = None

    def reset(self, *args, **kwargs):
        """Override reset to (re-)initialise the CSV move logger per day."""
        state = super().reset(*args, **kwargs)
        # Create a fresh logger for this day if logging is enabled
        if self._log_dir:
            if self.move_logger is not None:
                self.move_logger.close()
            label = f"day{self.day_index}"
            self.move_logger = MoveCSVLogger(self._log_dir, label=label)
        return state

    # ── State encoding override ───────────────────────────────────────

    def _encode_state(self) -> NDArray[np.float32]:
        """Encode state: (C, R_uni, S, T)."""
        return self.encoder.encode(
            self.trains, self.trucks, self.current_time,
            crane_states=self.cranes,
        )

    # ══════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════

    def step_all_cranes(
        self, agent: DQNAgent,
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
                n_waiting = sum(
                    1 for t in self.trucks.values()
                    if not getattr(t, "is_terminal_truck", False)
                )
                total_reward += self.reward_engine.waiting_penalty(
                    n_waiting, advance_min,
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
        agent: DQNAgent,
        crane_id: int,
        state_np: NDArray[np.float32],
    ) -> StepResult:
        """One unified step: source selection → dest selection → execute."""
        info: Dict[str, Any] = {
            "crane_id": crane_id, "executed": [], "retries": 0,
        }
        total_penalty = 0.0
        sf = self.dims.split_factor

        for retry in range(self.max_retries + 1):
            info["retries"] = retry

            source_mask = self.encoder.get_source_mask(
                self.trains, self.trucks, self.current_time,
            )
            # Exclude containers moved by earlier cranes this step
            blacklist = getattr(self, "_step_blacklist", set())
            if blacklist:
                self._apply_source_blacklist(source_mask, blacklist)
            if not source_mask.any():
                log.debug(
                    "crane=%d retry=%d | NO_SOURCE (sources=%d blacklist=%d)",
                    crane_id, retry, 0, len(blacklist),
                )
                break  # nothing to do

            dest_mask_fn = self._make_dest_mask_fn(state_np)

            action = agent.act(
                state=state_np,
                source_mask=source_mask,
                dest_mask_fn=dest_mask_fn,
            )

            if action.move_type == "IDLE":
                log.debug(
                    "crane=%d retry=%d | IDLE chosen (penalty=%.1f)",
                    crane_id, retry, IDLE_PENALTY,
                )
                # Deliberate idle — flat penalty every time
                self._advance_time()
                next_state = self._encode_state()
                done = self._check_day_end()
                transition = Transition(
                    state=state_np,
                    source_pos_down=(-1, -1, -1),
                    dest_pos_down=None,
                    reward=IDLE_PENALTY,
                    next_state=next_state,
                    done=done,
                )
                info["idle"] = True
                return StepResult(
                    next_state=next_state,
                    reward=total_penalty + IDLE_PENALTY,
                    done=done,
                    info=info,
                    transition=transition,
                    was_valid_move=True,
                )

            if action.move_type is None or action.dest_pos is None:
                log.debug(
                    "crane=%d retry=%d | NO_DEST src=%s src_region=%s "
                    "move_type=%s dest_pos=%s",
                    crane_id, retry,
                    action.source_pos, action.source_region,
                    action.move_type, action.dest_pos,
                )
                break  # no valid action — idle

            # Log the attempt
            src = action.source_pos
            dst = action.dest_pos
            log.debug(
                "crane=%d retry=%d | ATTEMPT %s  "
                "src=(%d,s%d,t%d)[%s]  dst=(%d,s%d,t%d)[%s]",
                crane_id, retry, action.move_type,
                src[0], src[1], src[2], action.source_region,
                dst[0], dst[1], dst[2], action.dest_region,
            )

            if action.move_type == "PARK_TRUCK":
                result = self._execute_park_truck(
                    state_np, action, crane_id, info,
                )
                log.debug(
                    "crane=%d | PARK_TRUCK → bay=%d valid=%s reward=%.2f",
                    crane_id, dst[1] // sf, result.was_valid_move,
                    result.reward,
                )
                result.reward += total_penalty
                return result

            # Container move
            result = self._execute_move(
                state_np, action, crane_id, info,
            )
            if result is not None:
                log.debug(
                    "crane=%d | SUCCESS %s  "
                    "src=(%d,b%d,t%d) → dst=(%d,b%d,t%d)  reward=%.2f",
                    crane_id, action.move_type,
                    src[0], src[1] // sf, src[2],
                    dst[0], dst[1] // sf, dst[2],
                    result.reward,
                )
                result.reward += total_penalty
                return result

            # Invalid → penalty, retry
            total_penalty += self.no_destination_penalty
            info.setdefault("retry_reasons", []).append("invalid_placement")
            log.debug(
                "crane=%d retry=%d | RESOLVER_REJECTED %s  "
                "src=(%d,s%d,t%d) dst=(%d,s%d,t%d)  penalty=%.1f",
                crane_id, retry, action.move_type,
                src[0], src[1], src[2],
                dst[0], dst[1], dst[2],
                total_penalty,
            )

        # All retries exhausted or nothing to do — idle advance
        log.debug(
            "crane=%d | FALLTHROUGH retries=%d total_penalty=%.1f",
            crane_id, info["retries"], total_penalty,
        )
        self._advance_time()
        return StepResult(
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
    ) -> Callable[[int, int, int], Tuple[np.ndarray, int]]:
        """Build closure returning ``(mask, n_splits)`` per source.

        *mask* is an ``(R_uni, S_down, T)`` boolean array of valid
        destinations.  *n_splits* is the source container's footprint in
        sub-bays so the caller can resolve the downsampled destination back
        to full resolution with the correct width.

        Computes yard validity per source container's actual n_splits so the
        mask accurately reflects whether that container fits.  A lightweight
        cache avoids recomputing the same validity mask for same-size
        containers within a single step.
        """
        d = self.dims
        stride = self._s_stride
        S_down = self._s_down
        length_map = self.yard.container_length_map
        sf = d.split_factor

        parking_free_down = self._parking_free_mask_down()

        # Lazy cache: n_splits → downsampled yard validity (populated on demand)
        _yard_cache: Dict[int, np.ndarray] = {}

        def _yard_valid_for(n_splits: int) -> np.ndarray:
            """Get (or compute) downsampled yard validity mask for *n_splits*."""
            cached = _yard_cache.get(n_splits)
            if cached is not None:
                return cached
            full = self.encoder.get_yard_validity_mask(n_splits)
            down = _downsample_mask(full, stride)
            _yard_cache[n_splits] = down
            return down

        def dest_mask_fn(
            src_row: int, src_split: int, src_tier: int,
        ) -> Tuple[np.ndarray, int]:
            region = d.region_of(src_row)
            mask = np.zeros((d.R_unified, S_down, d.n_tiers), dtype=bool)

            if region == "QUEUE":
                # Queue → Parking: free parking spots
                mask[d.parking_row_start, :, 0] = parking_free_down
                return mask, 1

            # Look up actual container to get correct n_splits
            container = self._source_container_object(
                region, src_row, src_split, src_tier,
            )
            n_splits = _container_n_splits(container, length_map, sf)
            yard_down = _yard_valid_for(n_splits)

            if region in ("RAIL", "PARKING"):
                # Rail/Parking → Yard: valid yard placements for this size
                mask[d.yard_row_start:d.yard_row_end] = (
                    yard_down[d.yard_row_start:d.yard_row_end]
                )
                # Also allow direct transfers:
                #   RAIL → PARKING (TRAIN_TO_TRUCK)
                #   PARKING → RAIL (TRUCK_TO_TRAIN)
                if container is not None:
                    if region == "RAIL":
                        # Train container → matching pickup truck
                        if _is_import(container):
                            self._add_truck_dest_mask(
                                mask, container, d, S_down, stride,
                            )
                    else:
                        # Truck container → matching train
                        if _is_export(container):
                            self._add_rail_dest_mask(
                                mask, container, d, S_down, stride,
                            )

            elif region == "YARD":
                # Yard → Yard: restack targets for this container size
                mask[d.yard_row_start:d.yard_row_end] = (
                    yard_down[d.yard_row_start:d.yard_row_end]
                )
                # Exclude source position
                src_s_down = src_split // stride
                if d.yard_row_start <= src_row < d.yard_row_end:
                    mask[src_row, src_s_down, src_tier] = False

                # Direction-dependent vehicle targets
                direction = state[_CH_DIRECTION, src_row, src_split, src_tier]
                if container is not None:
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

                    # Swap body / trailer → parked terminal truck
                    if (getattr(container, "is_swap_body", False)
                            or getattr(container, "is_trailer", False)):
                        self._add_terminal_truck_dest_mask(
                            mask, d, S_down, stride,
                        )

            return mask, n_splits

        return dest_mask_fn

    def _source_container_object(
        self, region: str, row: int, split: int, tier: int,
    ):
        """Return the container object at the given source position, or None."""
        if region == "RAIL":
            rec = self.coordinator.get_rail_record(row, split)
            return rec.container if rec is not None else None
        elif region == "PARKING":
            truck = self.coordinator.get_truck_at(split)
            if truck is not None:
                containers = getattr(truck, "containers", [])
                return containers[0] if containers else None
        elif region == "YARD":
            rec = self.coordinator.get_yard_record(row, split, tier)
            return rec.container if rec is not None else None
        return None

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
        d: Dims,
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
        d: Dims,
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

    def _add_terminal_truck_dest_mask(
        self,
        mask: np.ndarray,
        d: Dims,
        S_down: int,
        stride: int,
    ) -> None:
        """Mark parking positions of available terminal trucks.

        Called when the source container is a swap body or trailer. Lights up
        the parking-row positions of all parked, idle terminal trucks so the
        agent can dispatch swap bodies to them.
        """
        sf = d.split_factor
        for tt in self.terminal_trucks.values():
            if not tt.is_available():
                continue
            spot = getattr(tt, "parking_spot", None)
            if spot is None:
                continue
            s = spot.bay * sf + (spot.split or 0)
            s_down = s // stride
            if 0 <= s_down < S_down:
                mask[d.parking_row_start, s_down, 0] = True

    # ══════════════════════════════════════════════════════════════════
    # CSV logging helper
    # ══════════════════════════════════════════════════════════════════

    def _log_container_move(
        self,
        move: Move,
        action: ActionResult,
        container_id: str,
        crane_id: int,
        distance_m: float,
        time_s: float,
    ) -> None:
        """Write one row to the moves CSV (if logger active)."""
        d = self.dims

        # Look up container metadata
        container = None
        for rec in self.yard.iter_records():
            if rec.container.container_id == container_id:
                container = rec.container
                break
        # Also check trains if not found in yard (e.g. TRAIN_TO_YARD after move)
        if container is None:
            for train in self.trains.values():
                c = train.find_container(container_id) if hasattr(train, "find_container") else None
                if c:
                    container = c
                    break

        length_ft = getattr(container, "length_ft", 0) if container else 0
        goods_type = getattr(container, "goods_type", "") if container else ""
        is_sb = getattr(container, "is_swap_body", False) if container else False
        is_tr = getattr(container, "is_trailer", False) if container else False

        src = action.source_pos or (-1, -1, -1)
        dst = action.dest_pos or (-1, -1, -1)
        sf = d.split_factor

        self.move_logger.log_move(
            timestamp=self.current_time,
            move_type=move.type.value,
            container_id=container_id,
            length_ft=length_ft,
            goods_type=str(goods_type),
            is_swap_body=is_sb,
            is_trailer=is_tr,
            src_region=action.source_region or "",
            src_row=src[0],
            src_bay=src[1] // sf if src[1] >= 0 else -1,
            src_split=src[1],
            src_tier=src[2],
            dst_region=action.dest_region or "",
            dst_row=dst[0],
            dst_bay=dst[1] // sf if dst[1] >= 0 else -1,
            dst_split=dst[1],
            dst_tier=dst[2],
            crane_id=crane_id,
            crane_time_s=time_s,
            crane_distance_m=distance_m,
        )

    # ══════════════════════════════════════════════════════════════════
    # Execution: PARK_TRUCK (queue → parking)
    # ══════════════════════════════════════════════════════════════════

    def _execute_park_truck(
        self,
        state_np: NDArray,
        action: ActionResult,
        crane_id: int,
        info: Dict,
    ) -> StepResult:
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

        # Terminal trucks become IDLE once parked so they can accept dispatches
        if getattr(truck, "is_terminal_truck", False):
            truck.status = TerminalTruckStatus.IDLE

        # Proximity reward: bonus for parking near goods anchor
        preferred = self.encoder._preferred_bay_for_truck(truck)
        if preferred is not None:
            delta = abs(dst_bay - preferred)
            proximity_bonus = PARKING_PROXIMITY_BONUS_MAX * max(
                0.0, 1.0 - delta / PARKING_PROXIMITY_RADIUS_BAYS,
            )
        else:
            proximity_bonus = 0.0

        reward = PARKING_REWARD_BASE + proximity_bonus

        src_park = action.source_pos or (-1, -1, -1)
        dst_park = action.dest_pos or (-1, -1, -1)
        sf_park = self.dims.split_factor
        info["executed"].append({
            "move_type": "PARK_TRUCK",
            "truck_id": truck.truck_id,
            "bay": dst_bay,
            "proximity_bonus": round(proximity_bonus, 4),
            # Spatial data for visualization
            "src_row": src_park[0],
            "src_bay": src_park[1] // sf_park if src_park[1] >= 0 else -1,
            "dst_row": dst_park[0],
            "dst_bay": dst_park[1] // sf_park if dst_park[1] >= 0 else -1,
        })

        # CSV event logging
        if self.move_logger is not None:
            is_tt = getattr(truck, "is_terminal_truck", False)
            vtype = "terminal_truck" if is_tt else (
                "pickup" if getattr(truck, "is_pickup_truck", False) else "delivery"
            )
            self.move_logger.log_vehicle_event(
                timestamp=self.current_time,
                event_type="PARK_TRUCK",
                vehicle_id=truck.truck_id,
                vehicle_type=vtype,
                track_or_bay=str(dst_bay),
            )

        # No crane cost for parking — just advance time
        self._advance_time()
        next_state = self._encode_state()
        done = self._check_day_end()

        transition = Transition(
            state=state_np,
            source_pos_down=action.source_pos_down,
            dest_pos_down=action.dest_pos_down,
            reward=reward,
            next_state=next_state,
            done=done,
        )

        return StepResult(
            next_state=next_state, reward=reward, done=done,
            info=info, transition=transition, was_valid_move=True,
        )

    # ══════════════════════════════════════════════════════════════════
    # Execution: container moves
    # ══════════════════════════════════════════════════════════════════

    def _execute_move(
        self,
        state_np: NDArray,
        action: ActionResult,
        crane_id: int,
        info: Dict,
    ) -> Optional[StepResult]:
        """Execute a container move. Returns None if invalid."""
        move = self._resolve_move(action)
        if move is None:
            # Resolver already logged the specific rejection reason.
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
            log.debug("  TLM exception: %s", e)
            return None

        if not success:
            log.debug("  TLM execute returned False for %s",
                       move.type.value)
            return None

        # Sync rail grid after any train-mutating move
        if move.type in (
            MoveType.TRAIN_TO_YARD, MoveType.YARD_TO_TRAIN,
            MoveType.TRAIN_TO_TRUCK, MoveType.TRUCK_TO_TRAIN,
        ):
            train_id = move.args.get("train_id")
            if train_id:
                train = self.trains.get(train_id)
                if train:
                    self.rail.sync_train(train_id, train)

        # Terminal truck busy timer — TT carries the container offsite
        if move.type == MoveType.YARD_TO_TERMINAL_TRUCK:
            from simulation.core.constants import TERMINAL_TRUCK_TASK_DURATION_S
            tt_id = move.args.get("terminal_truck_id")
            if tt_id:
                self._tt_busy_until[tt_id] = (
                    self.current_time
                    + timedelta(seconds=TERMINAL_TRUCK_TASK_DURATION_S)
                )

        # Crane timing
        time_s = cost.time_s if cost else 0.0
        if crane_id < len(self.cranes) and cost:
            self.cranes[crane_id].busy_until = (
                self.current_time + timedelta(seconds=time_s)
            )
            # Track destination bay for crane proximity channel
            if action.dest_pos is not None:
                self.cranes[crane_id].last_bay = (
                    action.dest_pos[1] // self.dims.split_factor
                )

        distance_m = cost.distance_m if cost else 0.0
        reward = self.reward_engine.immediate_reward(
            move.type.value, distance_m, time_s,
        )

        container_id = move.args.get("container_id", "")
        src = action.source_pos or (-1, -1, -1)
        dst = action.dest_pos or (-1, -1, -1)
        sf = self.dims.split_factor
        info["executed"].append({
            "move_type": move.type.value,
            "container_id": container_id,
            "distance_m": round(distance_m, 2),
            "time_s": round(time_s, 2),
            # Spatial data for visualization
            "src_row": src[0],
            "src_bay": src[1] // sf if src[1] >= 0 else -1,
            "dst_row": dst[0],
            "dst_bay": dst[1] // sf if dst[1] >= 0 else -1,
        })

        # CSV move logging
        if self.move_logger is not None:
            self._log_container_move(
                move, action, container_id, crane_id, distance_m, time_s,
            )

        # Prevent other cranes from reversing this move in the same step
        blacklist = getattr(self, "_step_blacklist", None)
        if blacklist is not None and container_id:
            blacklist.add(container_id)

        self._advance_time(time_s)
        next_state = self._encode_state()
        done = self._check_day_end()

        transition = Transition(
            state=state_np,
            source_pos_down=action.source_pos_down,
            dest_pos_down=action.dest_pos_down,
            reward=reward,
            next_state=next_state,
            done=done,
        )

        return StepResult(
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
        "TRAIN_TO_TRUCK": "_resolve_rail_to_parking",
        "TRUCK_TO_TRAIN": "_resolve_parking_to_rail",
    }

    def _resolve_move(self, action: ActionResult) -> Optional[Move]:
        """Convert unified spatial action to a TLM Move."""
        mt = action.move_type
        handler_name = self._MOVE_DISPATCH.get(mt)
        if handler_name is None:
            return None
        return getattr(self, handler_name)(action)

    def _resolve_yard_to_yard(self, action: ActionResult) -> Optional[Move]:
        """Restack: yard container → different yard position."""
        record = self.coordinator.get_yard_record(*action.source_pos)
        if record is None:
            log.debug("  YARD_TO_YARD reject: no container at src %s",
                       action.source_pos)
            return None

        container = record.container
        n_splits = self.yard.container_length_map.get(
            getattr(container, "length_ft", 0), 0,
        )

        placement = self.coordinator.validate_yard_dest(
            *action.dest_pos, n_splits=n_splits,
        )
        if placement is None:
            log.debug("  YARD_TO_YARD reject: invalid dest %s", action.dest_pos)
            return None

        # Reject no-op (same position)
        pl = record.placement
        src_abs = pl.bay * self.yard.split_factor + pl.start_split
        if (placement.row == pl.row
                and action.dest_pos[1] == src_abs
                and placement.tier == pl.tier):
            log.debug("  YARD_TO_YARD reject: no-op (same position)")
            return None

        return Move(
            type=MoveType.YARD_TO_YARD,
            args={
                "container_id": container.container_id,
                "placement": placement,
            },
        )

    def _resolve_yard_to_train(self, action: ActionResult) -> Optional[Move]:
        """Export: yard container → train."""
        record = self.coordinator.get_yard_record(*action.source_pos)
        if record is None:
            return None

        container = record.container
        if not _is_export(container):
            return None

        dst_row = action.dest_pos[0]
        train = self.coordinator.get_train_on_track(dst_row, self.trains)
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

    def _resolve_yard_to_truck(self, action: ActionResult) -> Optional[Move]:
        """Import: yard container → pickup truck at parking.

        Also handles terminal truck dispatch: if the destination truck is a
        terminal truck and the source container is a swap body or trailer,
        the move is resolved as YARD_TO_TERMINAL_TRUCK instead.
        """
        record = self.coordinator.get_yard_record(*action.source_pos)
        if record is None:
            return None

        container = record.container
        truck = self.coordinator.get_truck_at(action.dest_pos[1])
        if truck is None:
            return None

        # Terminal truck dispatch: swap body/trailer → parked terminal truck
        if getattr(truck, "is_terminal_truck", False):
            is_sb = getattr(container, "is_swap_body", False)
            is_tr = getattr(container, "is_trailer", False)
            if is_sb or is_tr:
                return Move(
                    type=MoveType.YARD_TO_TERMINAL_TRUCK,
                    args={
                        "terminal_truck_id": truck.truck_id,
                        "container_id": container.container_id,
                    },
                )
            return None

        if not _is_import(container):
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

    def _resolve_rail_to_yard(self, action: ActionResult) -> Optional[Move]:
        """Import: train container → yard placement (exact coordinates)."""
        rec = self.coordinator.get_rail_record(
            action.source_pos[0], action.source_pos[1],
        )
        if rec is None:
            log.debug("  TRAIN_TO_YARD reject: no container at rail src %s",
                       action.source_pos)
            return None

        container = rec.container
        n_splits = self.yard.container_length_map.get(
            getattr(container, "length_ft", 0), 0,
        )

        placement = self.coordinator.validate_yard_dest(
            *action.dest_pos, n_splits=n_splits,
        )
        if placement is None:
            log.debug("  TRAIN_TO_YARD reject: invalid dest %s", action.dest_pos)
            return None

        return Move(
            type=MoveType.TRAIN_TO_YARD,
            args={
                "train_id": rec.train_id,
                "container_id": container.container_id,
                "placement": placement,
            },
        )

    def _resolve_parking_to_yard(self, action: ActionResult) -> Optional[Move]:
        """Delivery: truck container → yard placement (exact coordinates)."""
        truck = self.coordinator.get_truck_at(action.source_pos[1])
        if truck is None or not getattr(truck, "containers", None):
            log.debug("  TRUCK_TO_YARD reject: no truck/container at "
                       "parking split %d", action.source_pos[1])
            return None

        container = truck.containers[0]
        n_splits = self.yard.container_length_map.get(
            getattr(container, "length_ft", 0), 0,
        )

        placement = self.coordinator.validate_yard_dest(
            *action.dest_pos, n_splits=n_splits,
        )
        if placement is None:
            log.debug("  TRUCK_TO_YARD reject: invalid dest %s", action.dest_pos)
            return None

        return Move(
            type=MoveType.TRUCK_TO_YARD,
            args={
                "truck_id": truck.truck_id,
                "container_id": container.container_id,
                "placement": placement,
            },
        )

    def _resolve_rail_to_parking(self, action: ActionResult) -> Optional[Move]:
        """Direct transfer: train container → pickup truck at parking."""
        rec = self.coordinator.get_rail_record(
            action.source_pos[0], action.source_pos[1],
        )
        if rec is None:
            return None

        container = rec.container
        if not _is_import(container):
            return None

        truck = self.coordinator.get_truck_at(action.dest_pos[1])
        if truck is None:
            return None

        if getattr(truck, "is_terminal_truck", False):
            return None

        if container.container_id not in getattr(truck, "pickup_container_ids", set()):
            return None
        if not truck.can_accommodate_container(container):
            return None

        return Move(
            type=MoveType.TRAIN_TO_TRUCK,
            args={
                "train_id": rec.train_id,
                "truck_id": truck.truck_id,
                "container_id": container.container_id,
            },
        )

    def _resolve_parking_to_rail(self, action: ActionResult) -> Optional[Move]:
        """Direct transfer: truck container → train."""
        truck = self.coordinator.get_truck_at(action.source_pos[1])
        if truck is None or not getattr(truck, "containers", None):
            return None

        container = truck.containers[0]
        if not _is_export(container):
            return None

        dst_row = action.dest_pos[0]
        train = self.coordinator.get_train_on_track(dst_row, self.trains)
        if train is None:
            return None

        if container.container_id not in train.get_all_pickup_container_ids():
            return None
        if not train.has_space_for_container(container):
            return None

        return Move(
            type=MoveType.TRUCK_TO_TRAIN,
            args={
                "train_id": train.train_id,
                "truck_id": truck.truck_id,
                "container_id": container.container_id,
            },
        )

    def _find_queued_truck_at(
        self, queue_row: int, split: int,
    ) -> Optional[Truck]:
        """Find the truck encoded at (queue_row, split).

        Delegates to the canonical ``_compute_queue_layout()`` so the
        mapping is identical to the tensor encoding and source mask.
        """
        d = self.dims
        if queue_row < d.queue_row_start or queue_row >= d.queue_row_end:
            return None

        layout = self.encoder._compute_queue_layout(
            self.trucks, self.current_time,
        )
        lookup = {(r, s): tk for tk, r, s in layout}
        return lookup.get((queue_row, split))

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

        # Rail: iterate grid records (replaces wagon split_cursor walk)
        for rec in self.rail.iter_records():
            if rec.container.container_id in blacklist:
                mask[rec.track, rec.abs_split, 0] = False

        # Parking: iterate records (replaces truck iteration + bay math)
        for rec in self.parking.iter_records():
            containers = getattr(rec.truck, "containers", None)
            if not containers:
                continue
            if any(c.container_id in blacklist for c in containers):
                s = rec.abs_split
                if 0 <= s < d.n_splits:
                    mask[d.parking_row_start, s, 0] = False

    # ── Idle / utility ────────────────────────────────────────────────

    def _idle_result(self, info: Dict) -> StepResult:
        """Advance time, return zero-reward result."""
        self._advance_time()
        return StepResult(
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


def _container_n_splits(container, length_map, split_factor: int) -> int:
    """Calculate the number of sub-bay splits a container occupies.

    Derives it directly from the container's ``length_ft`` and the yard's
    ``split_factor`` (sub-bays per bay).  Falls back to the yard's
    ``container_length_map`` when the attribute is missing, and to 20 ft
    (the minimum) as a last resort.
    """
    if container is None:
        return max(1, split_factor // 2)  # conservative: half a bay (20 ft)
    length_ft = getattr(container, "length_ft", None)
    if length_ft is None:
        cid = getattr(container, "container_id", None)
        length_ft = length_map.get(cid, 20) if cid and length_map else 20
    ft_per_split = 40.0 / split_factor
    return max(1, int(np.ceil(length_ft / ft_per_split)))


