# simulation/env/state_encoder.py
"""14-channel state encoder for Spatial DQN.

Output shape: (C, R_uni, S, T) where
  C     = 14 channels
  R_uni = n_tracks + n_parking + n_yard + n_queue (default 15)
  S     = n_bays * split_factor (default 1160)
  T     = n_tiers (default 5)

Channels 0-9:  per-entity features (occupancy, type, urgency, demand, …)
Channel  10:   crane proximity (triangular decay from each crane's last bay)
Channel  11:   train departure urgency (per-train countdown on rail rows)
Channel  12:   time progress (uniform seconds_since_midnight / 86400)
Channel  13:   export pickup demand (yard containers wanted by present trains)

Layout:
  Rows 0..6   Rail tracks    — containers on train wagons
  Row  7      Parking        — parked trucks with containers
  Rows 8..12  Yard           — stacked storage containers
  Rows 13..14 Queue          — waiting-to-park trucks at target bay
"""
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from simulation.core.facilities.yard import StorageYard
from simulation.core.facilities.railyard import RailYard
from simulation.core.vehicles.train import Train
from simulation.core.vehicles.truck import Truck
from simulation.core.enums import TruckStatus
from simulation.utils.direction_utils import is_import, is_export
from simulation.rl.multihead_dqn.config import Dims, DEFAULT_S_STRIDE


# ── Channel specification ────────────────────────────────────────────────

@dataclass(frozen=True)
class ChannelSpec:
    """Named channel indices for the unified state tensor."""
    OCCUPANCY: int = 0
    CONTAINER_START: int = 1
    CONTAINER_TYPE: int = 2
    ACCESSIBLE: int = 3
    DEPARTURE_URGENCY: int = 4
    BLOCKS_URGENT: int = 5
    DIRECTION: int = 6
    CONTAINER_HASH: int = 7
    TRUCK_DEMAND: int = 8
    TRAIN_DEMAND: int = 9
    CRANE_PROXIMITY: int = 10
    TRAIN_DEPARTURE_URGENCY: int = 11
    TIME_PROGRESS: int = 12
    EXPORT_PICKUP_DEMAND: int = 13  # yard containers wanted by present trains

    NUM_CHANNELS: int = 14


CH = ChannelSpec()

# ── Container type categorical values ────────────────────────────────────

TYPE_REGULAR: float = 0.25
TYPE_REEFER: float = 0.50
TYPE_DANGEROUS: float = 0.75
TYPE_SWAP: float = 1.00

# ── Terminal truck direction sentinel ────────────────────────────────────
# Terminal trucks use a DIRECTION value of 0.5 in queue/parking rows,
# distinguishing them from pickup trucks (0.0) and delivery trucks (1.0).
DIRECTION_TERMINAL_TRUCK: float = 0.5

# ── Normalisation constants ──────────────────────────────────────────────

MAX_DEPARTURE_DAYS: float = 30.0
SECONDS_PER_DAY: float = 86_400.0
DEFAULT_URGENCY: float = 1.0
MAX_TRUCK_WAIT_HOURS: float = 4.0
MAX_TRAIN_DEPARTURE_HOURS: float = 12.0
CRANE_DECAY_BAYS: int = 10

# ── Hash constants ───────────────────────────────────────────────────────

_HASH_PRIME: int = 2654435761
_HASH_MOD: int = 2**32


# ══════════════════════════════════════════════════════════════════════════
# Encoder
# ══════════════════════════════════════════════════════════════════════════

class StateEncoder:
    """Encodes terminal state as (C, R_uni, S, T) unified spatial tensor.

    All terminal entities (trains, yard, trucks, queue) share one
    coordinate system so the CNN can learn cross-modal spatial relationships.
    """

    __slots__ = ("yard", "rail", "parking", "dims", "_sf")

    def __init__(
        self,
        yard: StorageYard,
        rail: RailYard,
        parking=None,
        dims: Optional[Dims] = None,
    ):
        self.yard = yard
        self.rail = rail
        self.parking = parking
        self.dims = dims or Dims(
            n_yard_rows=yard.n_rows,
            n_bays=yard.n_bays,
            split_factor=yard.split_factor,
            n_tiers=yard.n_tiers,
        )
        self._sf = yard.split_factor

    # ── Public API ────────────────────────────────────────────────────

    def encode(
        self,
        trains: Dict[str, Train],
        trucks: Dict[str, Truck],
        now: datetime,
        crane_states: Optional[List] = None,
    ) -> np.ndarray:
        """Build unified state tensor (C, R_uni, S, T)."""
        d = self.dims
        S = d.n_splits
        T = d.n_tiers
        tensor = np.zeros((CH.NUM_CHANNELS, d.R_unified, S, T), dtype=np.float32)

        # Pre-compute demand maps
        truck_demand = _collect_truck_demand(trucks)
        train_demand = _collect_train_demand(trains, self.rail, d.n_bays)

        # Fill per-entity channels (0-9)
        self._fill_rail_rows(tensor, trains, now)
        self._fill_parking_row(tensor, trucks, now)
        self._fill_yard_rows(tensor, truck_demand, train_demand, now)
        self._fill_queue_rows(tensor, trucks, now)

        # Fill global / cross-entity channels (10-13)
        if crane_states is not None:
            self._fill_crane_proximity(tensor, crane_states)
        self._fill_train_departure_urgency(tensor, trains, now)
        self._fill_time_progress(tensor, now)
        self._fill_export_pickup_demand(tensor, trains, now)

        return tensor

    # ── Rail rows (0..n_tracks-1) ─────────────────────────────────────

    def _fill_rail_rows(
        self,
        tensor: np.ndarray,
        trains: Dict[str, Train],
        now: datetime,
    ) -> None:
        """Encode containers on trains via rail grid records.

        Uses RailYard.iter_records() for spatial data — same pattern as
        _fill_yard_rows() uses StorageYard.iter_records().
        """
        d = self.dims
        sf = self._sf

        # Container channels from grid records
        for rec in self.rail.iter_records():
            track = rec.track
            s0 = rec.abs_split
            s1 = min(s0 + rec.n_splits, d.n_splits)
            if s0 >= d.n_splits:
                continue
            container = rec.container

            tensor[CH.OCCUPANCY, track, s0:s1, 0] = 1.0
            tensor[CH.CONTAINER_START, track, s0, 0] = 1.0
            tensor[CH.CONTAINER_TYPE, track, s0:s1, 0] = _container_type_value(container)
            tensor[CH.ACCESSIBLE, track, s0:s1, 0] = 1.0  # all rail containers accessible
            tensor[CH.DEPARTURE_URGENCY, track, s0:s1, 0] = _compute_urgency(container, now)
            tensor[CH.DIRECTION, track, s0:s1, 0] = _direction_value(container)
            tensor[CH.CONTAINER_HASH, track, s0:s1, 0] = _container_hash(container.container_id)

        # Pickup demand (containers the train wants but aren't loaded yet)
        for train_id, train in trains.items():
            slot = self.rail.get_slot(train_id)
            if slot is None:
                continue
            track_row = self._track_id_to_row(slot.track_id)
            if track_row is None or track_row >= d.rail_row_end:
                continue
            anchor = slot.anchor_bay
            for cid in train.get_all_pickup_container_ids():
                if train.container_locations.get(cid) is not None:
                    continue  # already on train, encoded above
                s_center = min(anchor * sf + sf // 2, d.n_splits - 1)
                tensor[CH.TRAIN_DEMAND, track_row, s_center, 0] = max(
                    tensor[CH.TRAIN_DEMAND, track_row, s_center, 0], 1.0
                )

    # ── Parking row ───────────────────────────────────────────────────

    def _fill_parking_row(
        self,
        tensor: np.ndarray,
        trucks: Dict[str, Truck],
        now: datetime,
    ) -> None:
        """Encode parked trucks via parking grid records."""
        d = self.dims
        pr = d.parking_row_start

        for rec in self.parking.iter_records():
            tk = rec.truck
            s = rec.abs_split
            if s < 0 or s >= d.n_splits:
                continue

            is_tt = getattr(tk, "is_terminal_truck", False)
            has_pickup = bool(getattr(tk, "pickup_container_ids", None))
            has_delivery = bool(getattr(tk, "containers", None))

            tensor[CH.OCCUPANCY, pr, s, 0] = 1.0
            tensor[CH.CONTAINER_START, pr, s, 0] = 1.0
            tensor[CH.ACCESSIBLE, pr, s, 0] = 1.0

            # Direction: 0.5 for terminal trucks, else container direction or heuristic
            if is_tt:
                tensor[CH.DIRECTION, pr, s, 0] = DIRECTION_TERMINAL_TRUCK
            elif has_delivery:
                tensor[CH.DIRECTION, pr, s, 0] = _direction_value(tk.containers[0])
            else:
                tensor[CH.DIRECTION, pr, s, 0] = 0.0  # pickup truck (Import)

            # Wait urgency (terminal trucks never depart — no urgency)
            if not is_tt:
                arr = getattr(tk, "arrival_time", None)
                if arr and now and arr < now:
                    wait_h = (now - arr).total_seconds() / 3600.0
                    tensor[CH.DEPARTURE_URGENCY, pr, s, 0] = min(1.0, wait_h / MAX_TRUCK_WAIT_HOURS)

            # Demand flags
            if has_pickup:
                tensor[CH.TRUCK_DEMAND, pr, s, 0] = 1.0
            if has_delivery:
                tensor[CH.CONTAINER_TYPE, pr, s, 0] = _container_type_value(tk.containers[0])

            tensor[CH.CONTAINER_HASH, pr, s, 0] = _container_hash(
                getattr(tk, "truck_id", str(id(tk)))
            )

    # ── Yard rows ─────────────────────────────────────────────────────

    def _fill_yard_rows(
        self,
        tensor: np.ndarray,
        truck_demand: Dict[str, float],
        train_demand: Dict[str, float],
        now: datetime,
    ) -> None:
        """Encode storage yard containers (unchanged logic, shifted rows)."""
        d = self.dims
        y0 = d.yard_row_start
        S = d.n_splits
        T = d.n_tiers

        # Occupancy from yard mask (yard rows are 0-indexed internally)
        # yard.occupancy_mask shape: (T, R_yard, S) — transpose to (R_yard, S, T)
        occ = self.yard.occupancy_mask.transpose(1, 2, 0).astype(np.float32)
        n_yard = min(occ.shape[0], d.n_yard_rows)
        tensor[CH.OCCUPANCY, y0:y0 + n_yard, :, :] = occ[:n_yard]

        # Per-container channels
        urgency_grid = np.full((n_yard, S, T), -1.0, dtype=np.float32)

        for rec in self.yard.iter_records():
            c = rec.container
            pl = rec.placement
            r_local = pl.row
            if r_local >= n_yard:
                continue
            r = y0 + r_local
            t = pl.tier
            s0 = pl.abs_start
            s1 = min(s0 + rec.n_splits, S)
            cid = c.container_id

            tensor[CH.CONTAINER_START, r, s0, t] = 1.0
            tensor[CH.CONTAINER_TYPE, r, s0:s1, t] = _container_type_value(c)

            if rec.is_accessible:
                tensor[CH.ACCESSIBLE, r, s0:s1, t] = 1.0

            urg = _compute_urgency(c, now)
            tensor[CH.DEPARTURE_URGENCY, r, s0:s1, t] = urg
            urgency_grid[r_local, s0, t] = urg

            if cid in truck_demand:
                tensor[CH.TRUCK_DEMAND, r, s0:s1, t] = truck_demand[cid]
            if cid in train_demand:
                tensor[CH.TRAIN_DEMAND, r, s0:s1, t] = train_demand[cid]

            tensor[CH.DIRECTION, r, s0:s1, t] = _direction_value(c)
            tensor[CH.CONTAINER_HASH, r, s0:s1, t] = _container_hash(cid)

        # Blocking severity (yard rows only)
        _fill_blocking(tensor, urgency_grid, y0, n_yard, S, T)

    # ── Queue rows ────────────────────────────────────────────────────

    def _fill_queue_rows(
        self,
        tensor: np.ndarray,
        trucks: Dict[str, Truck],
        now: datetime,
    ) -> None:
        """Encode queued (not-yet-parked) trucks at urgency-sorted positions.

        Trucks are placed sequentially sorted by urgency (most urgent first,
        terminal trucks last).  Position is stride-spaced for clean 1-to-1
        mapping with the CNN's downsampled feature map.
        """
        for tk, row, s in self._compute_queue_layout(trucks, now):
            is_tt = getattr(tk, "is_terminal_truck", False)
            has_pickup = bool(getattr(tk, "pickup_container_ids", None))
            has_delivery = bool(getattr(tk, "containers", None))

            tensor[CH.OCCUPANCY, row, s, 0] = 1.0
            tensor[CH.CONTAINER_START, row, s, 0] = 1.0
            tensor[CH.ACCESSIBLE, row, s, 0] = 1.0

            # Direction: 0.5 for terminal trucks, 1.0 for delivery, 0.0 for pickup
            if is_tt:
                tensor[CH.DIRECTION, row, s, 0] = DIRECTION_TERMINAL_TRUCK
            else:
                tensor[CH.DIRECTION, row, s, 0] = 1.0 if (has_delivery and not has_pickup) else 0.0

            # Wait urgency (terminal trucks never depart — no urgency)
            if not is_tt:
                arr = getattr(tk, "arrival_time", None)
                if arr and now and arr < now:
                    wait_h = (now - arr).total_seconds() / 3600.0
                    tensor[CH.DEPARTURE_URGENCY, row, s, 0] = min(1.0, wait_h / MAX_TRUCK_WAIT_HOURS)

            # Demand
            if has_pickup:
                tensor[CH.TRUCK_DEMAND, row, s, 0] = 1.0
            if has_delivery and tk.containers:
                tensor[CH.CONTAINER_TYPE, row, s, 0] = _container_type_value(tk.containers[0])

            tensor[CH.CONTAINER_HASH, row, s, 0] = _container_hash(
                getattr(tk, "truck_id", str(id(tk)))
            )

    # ── Mask helpers ──────────────────────────────────────────────────

    def get_source_mask(
        self,
        trains: Dict[str, Train],
        trucks: Dict[str, Truck],
        now: Optional[datetime] = None,
    ) -> np.ndarray:
        """Source mask (R_uni, S, T): True where an actionable entity starts.

        Marks:
          - Accessible yard containers (CONTAINER_START)
          - Import containers on rail (CONTAINER_START)
          - Delivery trucks in parking (containers to unload)
          - Queued trucks in queue rows (ready to park)
        """
        d = self.dims
        mask = np.zeros((d.R_unified, d.n_splits, d.n_tiers), dtype=bool)

        # Yard: accessible container starts
        for rec in self.yard.iter_records():
            if rec.is_accessible:
                r = d.yard_row_start + rec.placement.row
                mask[r, rec.placement.abs_start, rec.placement.tier] = True

        # Rail: import containers via grid records
        for rec in self.rail.iter_records():
            # Only Import containers are source-selectable from rail.
            # Use the container's actual direction attribute — NOT the
            # pickup list, which gets cleared after loading and would
            # cause loaded Export containers to be misclassified as sources.
            if is_import(rec.container):
                s0 = rec.abs_split
                if 0 <= s0 < d.n_splits:
                    mask[rec.track, s0, 0] = True

        # Parking: delivery trucks with containers to unload via records.
        # Only mark trucks that carry EXPORT containers (genuine deliveries).
        # Pickup trucks that just collected Import containers should NOT be sources.
        for rec in self.parking.iter_records():
            tk = rec.truck
            containers = getattr(tk, "containers", None)
            if not containers:
                continue
            if not any(is_export(c) for c in containers):
                continue
            s = rec.abs_split
            if 0 <= s < d.n_splits:
                mask[d.parking_row_start, s, 0] = True

        # Queue: waiting trucks (ready to park)
        # Uses canonical layout — identical to _fill_queue_rows()
        for _tk, row, s in self._compute_queue_layout(trucks, now):
            mask[row, s, 0] = True

        return mask

    def get_yard_validity_mask(
        self,
        n_splits_needed: int,
        goods_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Validity mask for yard placements, padded to (R_uni, S, T).

        Only yard rows can receive containers; all other rows are False.
        """
        d = self.dims
        R_yard = d.n_yard_rows
        S = d.n_splits
        T = d.n_tiers

        occ = self.yard.occupancy_mask  # (T, R_yard, S)
        valid = np.zeros((R_yard, S, T), dtype=bool)

        # Support mask: tier>0 needs occupied tier below
        support = occ if T > 1 else None

        for t in range(T):
            if t > 0 and not np.any(occ[t - 1]):
                break

            for r in range(R_yard):
                free = (~occ[t, r, :]).astype(np.int32)
                cs = np.empty(S + 1, dtype=np.int32)
                cs[0] = 0
                np.cumsum(free, out=cs[1:])

                end_idx = np.arange(n_splits_needed, S + 1)
                start_idx = end_idx - n_splits_needed
                contiguous = (cs[end_idx] - cs[start_idx]) == n_splits_needed
                valid[r, :S - n_splits_needed + 1, t] = contiguous

                # Support check for upper tiers
                if support is not None and t > 0:
                    sup_row = support[t - 1, r, :]
                    for s_pos in range(S - n_splits_needed + 1):
                        if valid[r, s_pos, t] and not np.all(
                            sup_row[s_pos:s_pos + n_splits_needed]
                        ):
                            valid[r, s_pos, t] = False

        if goods_mask is not None:
            valid &= goods_mask

        # Pad into unified shape (only yard rows are valid)
        padded = np.zeros((d.R_unified, S, T), dtype=bool)
        padded[d.yard_row_start:d.yard_row_end] = valid
        return padded

    # ── Internal helpers ──────────────────────────────────────────────

    def _track_id_to_row(self, track_id) -> Optional[int]:
        """Convert track ID to row index in unified matrix."""
        try:
            tid = int(track_id)
        except (ValueError, TypeError):
            return None
        if 0 <= tid < self.dims.n_tracks:
            return tid
        return None

    @staticmethod
    def _is_queued(truck: Truck) -> bool:
        """Check if truck is in the waiting queue (arrived, not parked)."""
        return (
            getattr(truck, "parking_spot", None) is None
            and getattr(truck, "status", None) == TruckStatus.WAITING
        )

    def _compute_queue_layout(
        self,
        trucks: Dict[str, Truck],
        now: Optional[datetime] = None,
    ) -> List[Tuple[Truck, int, int]]:
        """Canonical queue layout: urgency-sorted sequential placement.

        Returns list of ``(truck, unified_row, split)`` tuples assigning
        each queued truck a deterministic position in the queue rows.

        Trucks are placed at stride-spaced splits (0, 4, 8, …) for a
        clean 1-to-1 mapping with the CNN's downsampled feature map.

        Sort order (index 0 = most urgent):
          1. Terminal trucks LAST  (``tt_flag`` 0 vs 1)
          2. ``DEPARTURE_URGENCY`` descending (longer wait → lower index)
          3. ``truck_id`` ascending for deterministic tiebreak
        """
        d = self.dims
        stride = DEFAULT_S_STRIDE
        trucks_per_row = d.n_splits // stride  # 1160 // 4 = 290

        queued: List[Tuple[int, float, str, Truck]] = []
        for tk in trucks.values():
            if not self._is_queued(tk):
                continue
            is_tt = getattr(tk, "is_terminal_truck", False)
            tt_flag = 1 if is_tt else 0
            if is_tt:
                urgency = 0.0
            else:
                arr = getattr(tk, "arrival_time", None)
                if arr and now and arr < now:
                    wait_h = (now - arr).total_seconds() / 3600.0
                    urgency = min(1.0, wait_h / MAX_TRUCK_WAIT_HOURS)
                else:
                    urgency = 0.0
            truck_id = getattr(tk, "truck_id", str(id(tk)))
            queued.append((tt_flag, -urgency, truck_id, tk))

        # tt_flag ASC → -urgency ASC (= urgency DESC) → truck_id ASC
        queued.sort(key=lambda x: (x[0], x[1], x[2]))

        result: List[Tuple[Truck, int, int]] = []
        for i, (_, _, _, tk) in enumerate(queued):
            row_idx = i // trucks_per_row
            if row_idx >= d.n_queue_rows:
                break  # queue rows exhausted (580+ trucks)
            unified_row = d.queue_row_start + row_idx
            split = (i % trucks_per_row) * stride
            result.append((tk, unified_row, split))
        return result

    # ── Global / cross-entity channels (10-12) ─────────────────────────

    def _fill_crane_proximity(
        self,
        tensor: np.ndarray,
        crane_states: List,
    ) -> None:
        """Paint triangular proximity signal for each crane along S.

        1.0 at the crane's last bay, linearly decaying to 0.0 at
        ±CRANE_DECAY_BAYS distance. Broadcast across all rows and tiers.
        """
        d = self.dims
        sf = self._sf
        S = d.n_splits
        half_width = CRANE_DECAY_BAYS * sf

        splits = np.arange(S, dtype=np.float32)
        for crane in crane_states:
            bay = getattr(crane, "last_bay", None)
            if bay is None:
                continue
            center = float(bay * sf + sf // 2)
            dist = np.abs(splits - center)
            signal = np.maximum(0.0, 1.0 - dist / half_width)  # (S,)
            # Broadcast to (R_uni, S, T) — take element-wise max
            np.maximum(
                tensor[CH.CRANE_PROXIMITY],
                signal[np.newaxis, :, np.newaxis],
                out=tensor[CH.CRANE_PROXIMITY],
            )

    def _fill_train_departure_urgency(
        self,
        tensor: np.ndarray,
        trains: Dict[str, Train],
        now: datetime,
    ) -> None:
        """Fill per-train departure urgency on rail rows.

        urgency = 1.0 when train departs NOW, 0.0 when departure is
        MAX_TRAIN_DEPARTURE_HOURS away or more.
        """
        d = self.dims
        sf = self._sf

        for train_id, train in trains.items():
            dep = getattr(train, "departure_time", None)
            if dep is None:
                continue

            slot = self.rail.get_slot(train_id)
            if slot is None:
                continue
            track_row = self._track_id_to_row(slot.track_id)
            if track_row is None or track_row >= d.rail_row_end:
                continue

            hours_until = max(0.0, (dep - now).total_seconds() / 3600.0)
            urgency = 1.0 - min(1.0, hours_until / MAX_TRAIN_DEPARTURE_HOURS)

            # Fill the train's footprint on its track row
            anchor = slot.anchor_bay
            n_wagons = len(train.wagons)
            s_start = anchor * sf
            s_end = min((anchor + n_wagons) * sf, d.n_splits)
            if s_start < s_end:
                tensor[CH.TRAIN_DEPARTURE_URGENCY, track_row, s_start:s_end, 0] = urgency

    def _fill_time_progress(
        self,
        tensor: np.ndarray,
        now: datetime,
    ) -> None:
        """Fill uniform time-of-day progress: seconds_since_midnight / 86400."""
        if now is None:
            return
        seconds = now.hour * 3600 + now.minute * 60 + now.second
        tensor[CH.TIME_PROGRESS, :, :, :] = seconds / SECONDS_PER_DAY

    def _fill_export_pickup_demand(
        self,
        tensor: np.ndarray,
        trains: Dict[str, Train],
        now: datetime,
    ) -> None:
        """Mark yard containers wanted by present trains' pickup lists.

        Value = train departure urgency (0.1→1.0) so the CNN sees both
        'this container is wanted' AND 'how urgently.'

        Only yard rows are marked — rail containers are already on the
        train and don't need this signal.
        """
        if now is None:
            return
        d = self.dims

        # Collect all pickup IDs from present trains, with urgency
        pickup_urgency: Dict[str, float] = {}
        for train_id, train in trains.items():
            dep = getattr(train, "departure_time", None)
            if dep is None:
                continue
            hours_until = max(0.0, (dep - now).total_seconds() / 3600.0)
            urgency = max(0.1, 1.0 - min(1.0, hours_until / MAX_TRAIN_DEPARTURE_HOURS))
            for cid in train.get_all_pickup_container_ids():
                # Keep highest urgency if container appears in multiple trains
                pickup_urgency[cid] = max(pickup_urgency.get(cid, 0.0), urgency)

        if not pickup_urgency:
            return

        # Mark yard containers that are in pickup lists
        y0 = d.yard_row_start
        for rec in self.yard.iter_records():
            cid = rec.container.container_id
            urg = pickup_urgency.get(cid)
            if urg is not None:
                r = y0 + rec.placement.row
                s0 = rec.placement.abs_start
                s1 = min(s0 + rec.n_splits, d.n_splits)
                t = rec.placement.tier
                tensor[CH.EXPORT_PICKUP_DEMAND, r, s0:s1, t] = urg

    def _preferred_bay_for_truck(self, truck: Truck) -> Optional[int]:
        """Compute preferred bay for queue positioning.

        Terminal trucks → right edge (near swap body storage zone).
        Pickup trucks → median bay of their wanted containers in yard.
        Delivery trucks → goods zone anchor of their container.
        """
        # Terminal truck: park near swap body zone (right edge)
        if getattr(truck, "is_terminal_truck", False):
            return self.dims.n_bays - 1

        # Pickup: find container locations in yard
        pickup_ids = getattr(truck, "pickup_container_ids", None)
        if pickup_ids:
            bays = []
            for cid in pickup_ids:
                pl = self.yard.get_placement(cid)
                if pl:
                    bays.append(pl.bay)
            if bays:
                bays.sort()
                return bays[len(bays) // 2]

            # Containers not in yard yet — check trains for anchor
            for train_id in list(self.rail._train_to_slot.keys()):
                slot = self.rail.get_slot(train_id)
                if slot:
                    return slot.anchor_bay
            return None

        # Delivery: use goods zone anchor
        containers = getattr(truck, "containers", None)
        if containers:
            return self._goods_anchor_bay(containers[0])

        return None

    def _goods_anchor_bay(self, container) -> int:
        """Approximate goods zone anchor (matches terminal_manager logic)."""
        n_bays = self.dims.n_bays
        gt = getattr(container, "goods_type", "Regular")

        if gt == "Reefer":
            return 0  # left edge
        if gt == "DangerousGoods":
            return n_bays // 2  # center
        if getattr(container, "is_swap_body", False) or getattr(container, "is_trailer", False):
            return n_bays - 1  # right edge (row-based, but bay anchor suffices)
        return n_bays // 3  # regular: ~1/3 mark


# ══════════════════════════════════════════════════════════════════════════
# Pure helper functions
# ══════════════════════════════════════════════════════════════════════════



def _container_type_value(c) -> float:
    """Categorical float for container type."""
    if getattr(c, "is_swap_body", False) or getattr(c, "is_trailer", False):
        return TYPE_SWAP
    gt = getattr(c, "goods_type", "Regular")
    if gt == "Reefer":
        return TYPE_REEFER
    if gt == "DangerousGoods":
        return TYPE_DANGEROUS
    return TYPE_REGULAR


def _direction_value(c) -> float:
    """0.0 = Import, 1.0 = Export, 0.5 = unknown."""
    direction = getattr(c, "direction", None)
    if direction is None:
        return 0.5
    d_str = direction.value if hasattr(direction, "value") else str(direction)
    if d_str == "Export":
        return 1.0
    if d_str == "Import":
        return 0.0
    return 0.5


def _container_hash(entity_id: str) -> float:
    """Deterministic hash in [0.1, 1.0]."""
    h = hash(entity_id) & 0xFFFFFFFF
    h = (h * _HASH_PRIME) % _HASH_MOD
    return 0.1 + 0.9 * (h / _HASH_MOD)


def _compute_urgency(c, now: datetime) -> float:
    """Normalised departure urgency in [0, 1]."""
    dep = getattr(c, "departure_date", None)
    if dep is None:
        return DEFAULT_URGENCY
    days = max(0.0, (dep - now).total_seconds() / SECONDS_PER_DAY)
    return min(1.0, days / MAX_DEPARTURE_DAYS)


def _container_splits(container, split_factor: int) -> int:
    """Number of splits a container occupies.

    Thin wrapper around :func:`_container_n_splits` from terminal_env.
    Kept as a local helper to avoid a circular import (encoder is imported
    by the env module).
    """
    length_ft = getattr(container, "length_ft", None) or getattr(container, "length", 20)
    sub_bay_ft = 40.0 / split_factor
    return max(1, int(np.ceil(length_ft / sub_bay_ft)))


def _fill_blocking(
    tensor: np.ndarray,
    urgency_grid: np.ndarray,
    yard_row_start: int,
    n_yard: int,
    S: int,
    T: int,
) -> None:
    """Blocking severity for yard rows only."""
    if T < 2:
        return
    occ = tensor[CH.OCCUPANCY, yard_row_start:yard_row_start + n_yard]
    for t in range(1, T):
        both = (occ[:, :, t] > 0.5) & (occ[:, :, t - 1] > 0.5)
        above_urg = urgency_grid[:, :, t]
        below_urg = urgency_grid[:, :, t - 1]
        valid = both & (above_urg >= 0) & (below_urg >= 0)
        severity = np.where(valid & (above_urg > below_urg), above_urg - below_urg, 0.0)
        tensor[CH.BLOCKS_URGENT, yard_row_start:yard_row_start + n_yard, :, t] = severity


def _collect_truck_demand(trucks: Dict[str, Truck]) -> Dict[str, float]:
    """Map container_id → demand value for any truck that wants it.

    Parked trucks get demand 1.0 (immediate pickup needed).
    Queued (unparked) trucks get demand 0.5, giving the agent an early
    signal to prepare containers before the truck is parked.
    """
    demand: Dict[str, float] = {}
    for tk in trucks.values():
        pickup_ids = getattr(tk, "pickup_container_ids", None)
        if not pickup_ids:
            continue
        is_parked = getattr(tk, "parking_spot", None) is not None
        val = 1.0 if is_parked else 0.5
        for cid in pickup_ids:
            # Keep highest demand if multiple trucks want same container
            if cid not in demand or val > demand[cid]:
                demand[cid] = val
    return demand


def _collect_train_demand(
    trains: Dict[str, Train],
    rail: RailYard,
    n_bays: int,
) -> Dict[str, float]:
    """Map container_id → normalised train anchor bay.

    The value encodes which train the container is destined for as a
    normalised position in [0.2, 1.0].  The 0.2 floor ensures that even
    low-anchor trains (bay 0–5) produce a clearly non-zero signal the CNN
    can distinguish from empty positions (0.0).
    """
    demand: Dict[str, float] = {}
    for tid, train in trains.items():
        anchor = rail.get_anchor_bay(tid)
        if anchor is None:
            continue
        # Map anchor ∈ [0, n_bays-1] → value ∈ [0.2, 1.0]
        raw = anchor / max(n_bays - 1, 1)        # 0.0 – 1.0
        val = 0.2 + 0.8 * raw                     # 0.2 – 1.0
        for cid in train.get_all_pickup_container_ids():
            demand[cid] = val
    return demand