# simulation/env/unified_state_encoder.py
"""10-channel unified state encoder for Spatial DQN.

Output shape: (C, R_uni, S, T) where
  C     = 10 channels
  R_uni = n_tracks + n_parking + n_yard + n_queue (default 15)
  S     = n_bays * split_factor (default 1160)
  T     = n_tiers (default 5)

Layout:
  Rows 0..6   Rail tracks    — containers on train wagons
  Row  7      Parking        — parked trucks with containers
  Rows 8..12  Yard           — stacked storage containers
  Rows 13..14 Queue          — waiting-to-park trucks at target bay
"""
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set

from simulation.core.facilities.yard import OptimizedStorageYard
from simulation.core.facilities.railyard import OptimizedRailYard
from simulation.core.vehicles.train import Train
from simulation.core.vehicles.truck import Truck
from simulation.core.enums import TruckStatus
from simulation.rl.multihead_dqn.config import UnifiedDims


# ── Channel specification ────────────────────────────────────────────────

@dataclass(frozen=True)
class UnifiedChannelSpec:
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

    NUM_CHANNELS: int = 10


CH = UnifiedChannelSpec()

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

# ── Hash constants ───────────────────────────────────────────────────────

_HASH_PRIME: int = 2654435761
_HASH_MOD: int = 2**32


# ══════════════════════════════════════════════════════════════════════════
# Encoder
# ══════════════════════════════════════════════════════════════════════════

class UnifiedStateEncoder:
    """Encodes terminal state as (C, R_uni, S, T) unified spatial tensor.

    All terminal entities (trains, yard, trucks, queue) share one
    coordinate system so the CNN can learn cross-modal spatial relationships.
    """

    __slots__ = ("yard", "rail", "parking", "dims", "_sf")

    def __init__(
        self,
        yard: OptimizedStorageYard,
        rail: OptimizedRailYard,
        parking=None,
        dims: Optional[UnifiedDims] = None,
    ):
        self.yard = yard
        self.rail = rail
        self.parking = parking
        self.dims = dims or UnifiedDims(
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
    ) -> np.ndarray:
        """Build unified state tensor (C, R_uni, S, T)."""
        d = self.dims
        S = d.n_splits
        T = d.n_tiers
        tensor = np.zeros((CH.NUM_CHANNELS, d.R_unified, S, T), dtype=np.float32)

        # Pre-compute demand maps
        truck_demand = _collect_truck_demand(trucks)
        train_demand = _collect_train_demand(trains, self.rail, d.n_bays)

        # Fill each region
        self._fill_rail_rows(tensor, trains, now)
        self._fill_parking_row(tensor, trucks, now)
        self._fill_yard_rows(tensor, truck_demand, train_demand, now)
        self._fill_queue_rows(tensor, trucks, now)

        return tensor

    # ── Rail rows (0..n_tracks-1) ─────────────────────────────────────

    def _fill_rail_rows(
        self,
        tensor: np.ndarray,
        trains: Dict[str, Train],
        now: datetime,
    ) -> None:
        """Encode containers on train wagons into rail track rows.

        Each wagon maps to ~1 bay. Container placed at wagon's bay offset
        within the train's anchor position.
        """
        d = self.dims
        sf = self._sf

        for train_id, train in trains.items():
            slot = self.rail.get_slot(train_id)
            if slot is None:
                continue

            track_row = self._track_id_to_row(slot.track_id)
            if track_row is None or track_row >= d.rail_row_end:
                continue

            anchor = slot.anchor_bay

            for wagon_idx, wagon in enumerate(train.wagons):
                wagon_bay = anchor + wagon_idx
                if wagon_bay < 0 or wagon_bay >= d.n_bays:
                    continue

                # Encode each container on this wagon
                split_cursor = 0
                for container in wagon.containers.values():
                    n_splits = _container_splits(container, sf)
                    s0 = wagon_bay * sf + split_cursor
                    s1 = min(s0 + n_splits, d.n_splits)
                    if s0 >= d.n_splits:
                        break

                    tensor[CH.OCCUPANCY, track_row, s0:s1, 0] = 1.0
                    tensor[CH.CONTAINER_START, track_row, s0, 0] = 1.0
                    tensor[CH.CONTAINER_TYPE, track_row, s0:s1, 0] = _container_type_value(container)
                    tensor[CH.ACCESSIBLE, track_row, s0:s1, 0] = 1.0  # all rail containers accessible
                    tensor[CH.DEPARTURE_URGENCY, track_row, s0:s1, 0] = _compute_urgency(container, now)
                    tensor[CH.DIRECTION, track_row, s0:s1, 0] = _direction_value(container)
                    tensor[CH.CONTAINER_HASH, track_row, s0:s1, 0] = _container_hash(container.container_id)

                    split_cursor += n_splits

            # Also encode pickup slots (containers the train wants but aren't loaded yet)
            for cid in train.get_all_pickup_container_ids():
                if train.container_locations.get(cid) is not None:
                    continue  # already on train, encoded above
                # Mark demand on the track row at anchor position
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
        """Encode parked trucks at their parking split positions."""
        d = self.dims
        pr = d.parking_row_start
        sf = self._sf

        for tk in trucks.values():
            spot = getattr(tk, "parking_spot", None)
            if spot is None:
                continue

            bay = getattr(spot, "bay", None)
            split_offset = getattr(spot, "split", 0) or 0
            if bay is None:
                continue

            s = bay * sf + split_offset
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
        """Encode queued (not-yet-parked) trucks at their preferred bay.

        Each queued truck is placed at its target bay position so the CNN
        can see WHERE demand is concentrating. Multiple trucks targeting the
        same bay stack across queue rows (row 0, row 1, ...).
        """
        d = self.dims
        sf = self._sf
        q0 = d.queue_row_start
        n_queue = d.n_queue_rows

        # Track occupied splits per queue row for stacking
        occupied: Dict[int, set] = {i: set() for i in range(n_queue)}

        for tk in trucks.values():
            if not self._is_queued(tk):
                continue

            preferred_bay = self._preferred_bay_for_truck(tk)
            if preferred_bay is None:
                preferred_bay = d.n_bays // 2  # center fallback
            preferred_bay = max(0, min(preferred_bay, d.n_bays - 1))

            s = preferred_bay * sf  # anchor at start of bay

            # Find first queue row with space at this split
            placed = False
            for qi in range(n_queue):
                if s not in occupied[qi]:
                    occupied[qi].add(s)
                    row = q0 + qi
                    placed = True
                    break

            if not placed:
                # Overflow: jitter ±1 split
                for offset in range(1, sf):
                    for sign in (1, -1):
                        s_jitter = s + sign * offset
                        if s_jitter < 0 or s_jitter >= d.n_splits:
                            continue
                        for qi in range(n_queue):
                            if s_jitter not in occupied[qi]:
                                occupied[qi].add(s_jitter)
                                row = q0 + qi
                                s = s_jitter
                                placed = True
                                break
                        if placed:
                            break
                    if placed:
                        break

            if not placed:
                continue  # very rare: queue totally full at this bay

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
    ) -> np.ndarray:
        """Unified source mask (R_uni, S, T): True where an actionable entity starts.

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

        # Rail: import containers on slotted trains
        for train_id, train in trains.items():
            slot = self.rail.get_slot(train_id)
            if slot is None:
                continue
            track_row = self._track_id_to_row(slot.track_id)
            if track_row is None or track_row >= d.rail_row_end:
                continue

            anchor = slot.anchor_bay
            for wagon_idx, wagon in enumerate(train.wagons):
                wagon_bay = anchor + wagon_idx
                if wagon_bay < 0 or wagon_bay >= d.n_bays:
                    continue
                split_cursor = 0
                for container in wagon.containers.values():
                    n_splits = _container_splits(container, self._sf)
                    s0 = wagon_bay * self._sf + split_cursor
                    if s0 >= d.n_splits:
                        break
                    # Only Import containers are source-selectable from rail.
                    # Use the container's actual direction attribute — NOT the
                    # pickup list, which gets cleared after loading and would
                    # cause loaded Export containers to be misclassified as sources.
                    if _is_import_container(container):
                        mask[track_row, s0, 0] = True
                    split_cursor += n_splits

        # Parking: delivery trucks with containers to unload
        # Only mark trucks that carry EXPORT containers (genuine deliveries).
        # Pickup trucks that were just loaded (Import containers collected)
        # must NOT be sources — they should depart, not get unloaded.
        for tk in trucks.values():
            spot = getattr(tk, "parking_spot", None)
            if spot is None:
                continue
            containers = getattr(tk, "containers", None)
            if not containers:
                continue
            # Check if the truck actually has containers to deliver TO the yard.
            # Delivery trucks carry Export containers (arrived from outside).
            # Pickup trucks that just collected Import containers should NOT be sources.
            if not any(_is_delivery_container(c) for c in containers):
                continue
            bay = getattr(spot, "bay", None)
            split_offset = getattr(spot, "split", 0) or 0
            if bay is None:
                continue
            s = bay * self._sf + split_offset
            if 0 <= s < d.n_splits:
                mask[d.parking_row_start, s, 0] = True

        # Queue: waiting trucks (ready to park)
        # Re-compute positions (same logic as _fill_queue_rows but only need starts)
        occupied: Dict[int, set] = {i: set() for i in range(d.n_queue_rows)}
        for tk in trucks.values():
            if not self._is_queued(tk):
                continue
            preferred_bay = self._preferred_bay_for_truck(tk)
            if preferred_bay is None:
                preferred_bay = d.n_bays // 2
            preferred_bay = max(0, min(preferred_bay, d.n_bays - 1))
            s = preferred_bay * self._sf
            for qi in range(d.n_queue_rows):
                if s not in occupied[qi]:
                    occupied[qi].add(s)
                    mask[d.queue_row_start + qi, s, 0] = True
                    break

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

def _is_import_container(container) -> bool:
    """True if container's intrinsic direction is Import."""
    d = getattr(container, "direction", None)
    if d is None:
        return False
    return (d.value if hasattr(d, "value") else str(d)) == "Import"


def _is_delivery_container(container) -> bool:
    """True if container's intrinsic direction is Export (delivery to yard).

    Delivery trucks bring Export containers from outside to the terminal.
    Pickup trucks collect Import containers from the yard.
    """
    d = getattr(container, "direction", None)
    if d is None:
        return False
    return (d.value if hasattr(d, "value") else str(d)) == "Export"


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
    """Number of splits a container occupies."""
    length_ft = getattr(container, "length_ft", None) or getattr(container, "length", 20)
    sub_bay_ft = 40.0 / split_factor  # BAY_LENGTH_FT / split_factor
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
    rail: OptimizedRailYard,
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