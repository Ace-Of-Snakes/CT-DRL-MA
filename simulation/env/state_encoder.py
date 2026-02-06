# simulation/env/state_encoder.py
"""12-channel split-level state encoder for Multi-Head DQN.

Output shape: (C, R_state, S, T) where
  C = 12 channels
  R_state = n_rows + n_parking_rows (yard rows + parking row)
  S = total_splits (n_bays * split_factor)
  T = n_tiers

Parking row encodes truck positions at split resolution so the CNN
can form direct spatial links between containers and their target trucks.

TRUCK_DEMAND is binary (1.0) -- the parking row provides the spatial
"where" signal, while TRUCK_DEMAND provides the "who wants it" signal.
TRAIN_DEMAND retains normalised anchor bay encoding.
"""
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime

from simulation.core.facilities.yard import OptimizedStorageYard
from simulation.core.facilities.railyard import OptimizedRailYard
from simulation.core.vehicles.train import Train
from simulation.core.vehicles.truck import Truck


# -- Parking spot helper ------------------------------------------------------

def _parking_spot_bay(spot) -> Optional[int]:
    """Extract bay from parking_spot (string 'P_bay_split' or object with .bay)."""
    if spot is None:
        return None
    # Object with .bay attribute
    bay = getattr(spot, "bay", None)
    if bay is not None:
        return int(bay)
    # String format: "prefix_bay_split"
    if isinstance(spot, str):
        parts = spot.rsplit("_", 2)
        if len(parts) >= 2:
            try:
                return int(parts[-2])
            except (ValueError, IndexError):
                pass
    return None


# -- Channel specification ---------------------------------------------------

@dataclass(frozen=True)
class ChannelSpec:
    """Named channel indices for the state tensor."""
    OCCUPANCY: int = 0
    CONTAINER_START: int = 1
    CONTAINER_TYPE: int = 2
    ACCESSIBLE: int = 3
    DEPARTURE_URGENCY: int = 4
    BLOCKS_URGENT: int = 5
    TRAIN_HEAT: int = 6
    TRUCK_HEAT: int = 7
    # Demand channels
    TRUCK_DEMAND: int = 8   # 1.0 if a parked truck wants this container (binary)
    TRAIN_DEMAND: int = 9   # train_anchor / n_bays if a train wants this container
    DIRECTION: int = 10     # 0.0 = import, 1.0 = export
    CONTAINER_HASH: int = 11  # Unique per-container fingerprint [0, 1]

    @staticmethod
    def num_channels() -> int:
        return 12


CH = ChannelSpec()
NUM_CHANNELS = CH.num_channels()

# -- Container type categorical values ----------------------------------------
TYPE_REGULAR: float = 0.25
TYPE_REEFER: float = 0.50
TYPE_DANGEROUS: float = 0.75
TYPE_SWAP: float = 1.00

# -- Normalisation / heat constants -------------------------------------------
MAX_DEPARTURE_DAYS: float = 30.0
SECONDS_PER_DAY: float = 86_400.0
TRAIN_HEAT_SIGMA_BAYS: float = 3.0
TRUCK_HEAT_SIGMA_BAYS: float = 2.0
DEFAULT_URGENCY: float = 1.0
MAX_TRUCK_WAIT_HOURS: float = 4.0

# -- Hash constants -----------------------------------------------------------
_HASH_PRIME: int = 2654435761
_HASH_MOD: int = 2**32


class SplitLevelStateEncoder:
    """Encodes terminal state as (C, R_state, S, T) tensor at split resolution.

    R_state = n_rows + n_parking_rows. The extra parking row(s) encode
    truck positions at split resolution so the CNN sees them spatially
    aligned with yard containers.
    """

    __slots__ = ("yard", "rail", "parking", "_splits_f", "_split_factor",
                 "_n_parking_rows")

    def __init__(self, yard: OptimizedStorageYard, rail: OptimizedRailYard,
                 parking=None, n_parking_rows: int = 1):
        self.yard = yard
        self.rail = rail
        self.parking = parking
        self._split_factor = yard.split_factor
        self._splits_f = np.arange(yard.total_splits, dtype=np.float32)
        self._n_parking_rows = n_parking_rows

    # -- Public API -----------------------------------------------------------

    def encode(
        self,
        trains: Dict[str, Train],
        trucks: Dict[str, Truck],
        now: datetime,
    ) -> np.ndarray:
        """Build full state tensor (C, R_state, S, T)."""
        R = self.yard.n_rows
        R_state = R + self._n_parking_rows
        S = self.yard.total_splits
        T = self.yard.n_tiers
        n_bays = self.yard.n_bays

        tensor = np.zeros((NUM_CHANNELS, R_state, S, T), dtype=np.float32)

        # Collect demand maps ONCE: container_id -> 1.0 (binary)
        truck_demand = _collect_truck_demand_binary(trucks)
        train_demand = _collect_train_demand_spatial(trains, self.rail, n_bays)

        # -- Ch 0  OCCUPANCY (yard rows only) --------------------------
        tensor[CH.OCCUPANCY, :R] = self.yard.occupancy_mask.transpose(1, 2, 0).astype(
            np.float32
        )

        # -- Per-container channels (single pass) ---------------------------
        urgency_grid = np.full((R, S, T), -1.0, dtype=np.float32)

        for rec in self.yard.iter_records():
            c = rec.container
            pl = rec.placement
            r, t = pl.row, pl.tier
            s0 = pl.abs_start
            s1 = min(s0 + rec.n_splits, S)
            cid = c.container_id

            # Ch 1  CONTAINER_START
            tensor[CH.CONTAINER_START, r, s0, t] = 1.0

            # Ch 2  CONTAINER_TYPE
            tensor[CH.CONTAINER_TYPE, r, s0:s1, t] = _container_type_value(c)

            # Ch 3  ACCESSIBLE
            if rec.is_accessible:
                tensor[CH.ACCESSIBLE, r, s0:s1, t] = 1.0

            # Ch 4  DEPARTURE_URGENCY
            urg = _compute_urgency(c, now)
            tensor[CH.DEPARTURE_URGENCY, r, s0:s1, t] = urg
            urgency_grid[r, s0, t] = urg

            # Ch 8  TRUCK_DEMAND -- binary demand
            if cid in truck_demand:
                tensor[CH.TRUCK_DEMAND, r, s0:s1, t] = truck_demand[cid]

            # Ch 9  TRAIN_DEMAND -- normalised train anchor position
            if cid in train_demand:
                tensor[CH.TRAIN_DEMAND, r, s0:s1, t] = train_demand[cid]

            # Ch 10 DIRECTION
            tensor[CH.DIRECTION, r, s0:s1, t] = _direction_value(c)

            # Ch 11 CONTAINER_HASH
            tensor[CH.CONTAINER_HASH, r, s0:s1, t] = _container_hash(cid)

        # -- Ch 5  BLOCKS_URGENT -------------------------------------------
        _fill_blocking(tensor, urgency_grid, R, S, T)

        # -- Ch 6  TRAIN_HEAT (from rail anchor positions) -----------------
        train_anchors = self._collect_train_anchors(trains)
        if train_anchors:
            heat = _gaussian_heat(
                train_anchors, self._splits_f, self._split_factor,
                TRAIN_HEAT_SIGMA_BAYS, S,
            )
            tensor[CH.TRAIN_HEAT, :, :, :] = heat[np.newaxis, :, np.newaxis]

        # -- Ch 7  TRUCK_HEAT (from actual PARKING positions) --------------
        truck_parking_bays = self._collect_truck_parking_bays(trucks)
        if truck_parking_bays:
            heat = _gaussian_heat(
                truck_parking_bays, self._splits_f, self._split_factor,
                TRUCK_HEAT_SIGMA_BAYS, S,
            )
            tensor[CH.TRUCK_HEAT, :, :, :] = heat[np.newaxis, :, np.newaxis]

        # -- Parking row(s): encode truck positions at split resolution ----
        self._fill_parking_row(tensor, R, S, trucks, now)

        return tensor

    # -- Mask helpers ---------------------------------------------------------

    def get_occupancy_mask(self) -> np.ndarray:
        return self.yard.occupancy_mask.transpose(1, 2, 0)

    def get_validity_mask(
        self,
        n_splits_needed: int,
        goods_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Validity mask for placement, padded to (R_state, S, T).

        Parking row is always False (containers can't be placed there).
        """
        R = self.yard.n_rows
        R_state = R + self._n_parking_rows
        S = self.yard.total_splits
        T = self.yard.n_tiers
        occ = self.yard.occupancy_mask

        valid = np.zeros((R, S, T), dtype=bool)

        for t in range(T):
            if t > 0:
                support = self.yard._get_support_mask(t, n_splits_needed)
            else:
                support = None

            for r in range(R):
                free = (~occ[t, r, :]).astype(np.int32)
                cs = np.empty(S + 1, dtype=np.int32)
                cs[0] = 0
                np.cumsum(free, out=cs[1:])

                end_idx = np.arange(n_splits_needed, S + 1)
                start_idx = end_idx - n_splits_needed
                contiguous = (cs[end_idx] - cs[start_idx]) == n_splits_needed
                valid[r, : S - n_splits_needed + 1, t] = contiguous

                if support is not None:
                    sup_row = support[r, :]
                    for s in range(S - n_splits_needed + 1):
                        if valid[r, s, t] and not np.all(
                            sup_row[s : s + n_splits_needed]
                        ):
                            valid[r, s, t] = False

        if goods_mask is not None:
            valid &= goods_mask

        # Pad with False for parking rows (no placement allowed)
        padded = np.zeros((R_state, S, T), dtype=bool)
        padded[:R] = valid
        return padded

    # -- Parking row encoding ------------------------------------------------

    def _fill_parking_row(
        self,
        tensor: np.ndarray,
        yard_rows: int,
        total_splits: int,
        trucks: Dict[str, Truck],
        now: datetime,
    ) -> None:
        """Fill parking row(s) in the state tensor.

        Encodes each parked truck at tier 0 of row `yard_rows`:
          OCCUPANCY:         1.0 at truck's (bay*sf + split)
          DIRECTION:         0.0 = pickup (Import), 1.0 = delivery (Export)
          DEPARTURE_URGENCY: min(1.0, wait_hours / MAX_WAIT)
          TRUCK_DEMAND:      1.0 if truck has pickup containers
          CONTAINER_HASH:    hash of truck_id for identity
        """
        pr = yard_rows  # first parking row index
        sf = self._split_factor

        for tk in trucks.values():
            spot = getattr(tk, "parking_spot", None)
            if spot is None:
                continue

            # Resolve parking split position
            bay = getattr(spot, "bay", None)
            split_offset = getattr(spot, "split", 0) or 0
            if bay is None:
                bay = _parking_spot_bay(spot)
            if bay is None:
                continue

            s = bay * sf + split_offset
            if s < 0 or s >= total_splits:
                continue

            # Occupancy
            tensor[CH.OCCUPANCY, pr, s, 0] = 1.0

            # Direction: pickup truck = Import = 0.0, delivery = Export = 1.0
            has_pickup = bool(getattr(tk, "pickup_container_ids", None))
            has_delivery = bool(getattr(tk, "containers", None))
            if has_delivery and not has_pickup:
                tensor[CH.DIRECTION, pr, s, 0] = 1.0
            else:
                tensor[CH.DIRECTION, pr, s, 0] = 0.0

            # Departure urgency (wait time)
            arr = getattr(tk, "arrival_time", None)
            if arr and now and arr < now:
                wait_h = (now - arr).total_seconds() / 3600.0
                tensor[CH.DEPARTURE_URGENCY, pr, s, 0] = min(
                    1.0, wait_h / MAX_TRUCK_WAIT_HOURS,
                )

            # Truck demand: 1.0 if this truck wants to pick up containers
            if has_pickup:
                tensor[CH.TRUCK_DEMAND, pr, s, 0] = 1.0

            # Container hash (truck identity)
            tensor[CH.CONTAINER_HASH, pr, s, 0] = _container_hash(
                getattr(tk, "truck_id", str(id(tk))),
            )

    # -- Internal helpers -----------------------------------------------------

    def _collect_train_anchors(self, trains: Dict[str, Train]) -> List[int]:
        anchors: List[int] = []
        for tid in trains:
            bay = self.rail.get_anchor_bay(tid)
            if bay is not None:
                anchors.append(bay)
        return anchors

    def _collect_truck_parking_bays(self, trucks: Dict[str, Truck]) -> List[int]:
        bays: List[int] = []
        for tk in trucks.values():
            bay = _parking_spot_bay(getattr(tk, "parking_spot", None))
            if bay is not None:
                bays.append(bay)
        return bays


# -- Spatial demand collection -----------------------------------------------

def _collect_truck_demand_binary(
    trucks: Dict[str, Truck],
) -> Dict[str, float]:
    """Map container_id -> 1.0 if any parked truck wants it.

    Binary signal replaces the weak normalised-bay encoding.
    Truck spatial position is now encoded in the parking row instead.
    """
    demand: Dict[str, float] = {}
    for tk in trucks.values():
        if getattr(tk, "parking_spot", None) is None:
            continue
        pickup_ids = getattr(tk, "pickup_container_ids", None)
        if pickup_ids:
            for cid in pickup_ids:
                demand[cid] = 1.0
    return demand


def _collect_train_demand_spatial(
    trains: Dict[str, Train], rail, n_bays: int,
) -> Dict[str, float]:
    """Map container_id -> normalised train anchor bay."""
    demand: Dict[str, float] = {}
    for tid, train in trains.items():
        anchor = rail.get_anchor_bay(tid)
        if anchor is None:
            continue
        val = max(0.1, anchor / max(n_bays - 1, 1))
        for cid in train.get_all_pickup_container_ids():
            demand[cid] = val
    return demand


# -- Module-level helpers (pure functions) ------------------------------------

def _container_type_value(c) -> float:
    if getattr(c, "is_swap_body", False) or getattr(c, "is_trailer", False):
        return TYPE_SWAP
    gt = getattr(c, "goods_type", "Regular")
    if gt == "Reefer":
        return TYPE_REEFER
    if gt == "DangerousGoods":
        return TYPE_DANGEROUS
    return TYPE_REGULAR


def _direction_value(c) -> float:
    direction = getattr(c, "direction", None)
    if direction is None:
        return 0.5
    d_str = direction.value if hasattr(direction, "value") else str(direction)
    if d_str == "Export":
        return 1.0
    if d_str == "Import":
        return 0.0
    return 0.5


def _container_hash(container_id: str) -> float:
    """Deterministic hash normalised to [0.1, 1.0].

    Clamped above 0.1 so non-zero always means 'container present'.
    """
    h = hash(container_id) & 0xFFFFFFFF
    h = (h * _HASH_PRIME) % _HASH_MOD
    return 0.1 + 0.9 * (h / _HASH_MOD)


def _compute_urgency(c, now: datetime) -> float:
    dep = getattr(c, "departure_date", None)
    if dep is None:
        return DEFAULT_URGENCY
    days = max(0.0, (dep - now).total_seconds() / SECONDS_PER_DAY)
    return min(1.0, days / MAX_DEPARTURE_DAYS)


def _fill_blocking(tensor, urgency_grid, R, S, T):
    """Blocking severity on yard rows only (excludes parking row)."""
    if T < 2:
        return
    occ = tensor[CH.OCCUPANCY, :R]  # yard rows only
    for t in range(1, T):
        both = (occ[:, :, t] > 0.5) & (occ[:, :, t - 1] > 0.5)
        above_urg = urgency_grid[:, :, t]
        below_urg = urgency_grid[:, :, t - 1]
        valid = both & (above_urg >= 0) & (below_urg >= 0)
        severity = np.where(valid & (above_urg > below_urg), above_urg - below_urg, 0.0)
        tensor[CH.BLOCKS_URGENT, :R, :, t] = severity


def _gaussian_heat(anchor_bays, splits_f, split_factor, sigma_bays, total_splits):
    heat = np.zeros(total_splits, dtype=np.float32)
    sigma_splits = sigma_bays * split_factor
    half_center = split_factor / 2.0
    for bay in anchor_bays:
        center = bay * split_factor + half_center
        center = min(max(0.0, center), total_splits - 1.0)
        dist = np.abs(splits_f - center)
        heat += np.exp(-0.5 * (dist / sigma_splits) ** 2)
    mx = heat.max()
    if mx > 0:
        heat /= mx
    return heat