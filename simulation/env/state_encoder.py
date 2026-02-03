# simulation/env/state_encoder.py
"""12-channel split-level state encoder for Multi-Head DQN.

Output shape: (C, R, S, T) where
  C = 12 channels
  R = n_rows
  S = total_splits (n_bays * split_factor)
  T = n_tiers

Channels 0-7:  original spatial/urgency features
Channels 8-11: demand signals and container identity
"""
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime

from simulation.core.facilities.yard import OptimizedStorageYard
from simulation.core.facilities.railyard import OptimizedRailYard
from simulation.core.vehicles.train import Train
from simulation.core.vehicles.truck import Truck


# -- Channel specification ---------------------------------------------------
@dataclass(frozen=True)
class ChannelSpec:
    """Named channel indices for the state tensor."""
    # Original channels
    OCCUPANCY: int = 0
    CONTAINER_START: int = 1
    CONTAINER_TYPE: int = 2
    ACCESSIBLE: int = 3
    DEPARTURE_URGENCY: int = 4
    BLOCKS_URGENT: int = 5
    TRAIN_HEAT: int = 6
    TRUCK_HEAT: int = 7     # Now: actual truck PARKING location heat
    # New channels
    TRUCK_DEMAND: int = 8   # 1.0 if a parked truck wants this container
    TRAIN_DEMAND: int = 9   # 1.0 if a train wants this container
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
DEFAULT_URGENCY: float = 1.0  # far away = not urgent

# -- Hash constants -----------------------------------------------------------
_HASH_PRIME: int = 2654435761  # Knuth multiplicative hash constant
_HASH_MOD: int = 2**32


class SplitLevelStateEncoder:
    """Encodes terminal state as (C, R, S, T) tensor at split resolution."""

    __slots__ = ("yard", "rail", "_splits_f", "_split_factor")

    def __init__(self, yard: OptimizedStorageYard, rail: OptimizedRailYard):
        self.yard = yard
        self.rail = rail
        self._split_factor = yard.split_factor
        self._splits_f = np.arange(yard.total_splits, dtype=np.float32)

    # -- Public API -----------------------------------------------------------

    def encode(
        self,
        trains: Dict[str, Train],
        trucks: Dict[str, Truck],
        now: datetime,
    ) -> np.ndarray:
        """Build full state tensor (C, R, S, T)."""
        R = self.yard.n_rows
        S = self.yard.total_splits
        T = self.yard.n_tiers

        tensor = np.zeros((NUM_CHANNELS, R, S, T), dtype=np.float32)

        # -- Collect demand sets ONCE before container loop -----------------
        truck_wanted_ids = _collect_truck_demand(trucks)
        train_wanted_ids = _collect_train_demand(trains)

        # -- Ch 0  OCCUPANCY ------------------------------------------------
        tensor[CH.OCCUPANCY] = self.yard.occupancy_mask.transpose(1, 2, 0).astype(
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

            # Ch 8  TRUCK_DEMAND - "a parked truck wants this container"
            if cid in truck_wanted_ids:
                tensor[CH.TRUCK_DEMAND, r, s0:s1, t] = 1.0

            # Ch 9  TRAIN_DEMAND - "a train wants this container"
            if cid in train_wanted_ids:
                tensor[CH.TRAIN_DEMAND, r, s0:s1, t] = 1.0

            # Ch 10 DIRECTION - import vs export
            tensor[CH.DIRECTION, r, s0:s1, t] = _direction_value(c)

            # Ch 11 CONTAINER_HASH - unique fingerprint for differentiation
            tensor[CH.CONTAINER_HASH, r, s0:s1, t] = _container_hash(cid)

        # -- Ch 5  BLOCKS_URGENT  (vectorised over tiers) -------------------
        _fill_blocking(tensor, urgency_grid, R, S, T)

        # -- Ch 6  TRAIN_HEAT (from rail anchor positions) ------------------
        train_anchors = self._collect_train_anchors(trains)
        if train_anchors:
            heat = _gaussian_heat(
                train_anchors, self._splits_f, self._split_factor,
                TRAIN_HEAT_SIGMA_BAYS, S,
            )
            tensor[CH.TRAIN_HEAT, :, :, :] = heat[np.newaxis, :, np.newaxis]

        # -- Ch 7  TRUCK_HEAT (from actual PARKING positions) ---------------
        truck_parking_bays = self._collect_truck_parking_bays(trucks)
        if truck_parking_bays:
            heat = _gaussian_heat(
                truck_parking_bays, self._splits_f, self._split_factor,
                TRUCK_HEAT_SIGMA_BAYS, S,
            )
            tensor[CH.TRUCK_HEAT, :, :, :] = heat[np.newaxis, :, np.newaxis]

        return tensor

    # -- Mask helpers ---------------------------------------------------------

    def get_occupancy_mask(self) -> np.ndarray:
        """Occupancy as (R, S, T) bool for container selection masking."""
        return self.yard.occupancy_mask.transpose(1, 2, 0)

    def get_validity_mask(
        self,
        n_splits_needed: int,
        goods_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Valid placement positions as (R, S, T) bool."""
        R = self.yard.n_rows
        S = self.yard.total_splits
        T = self.yard.n_tiers
        occ = self.yard.occupancy_mask  # (T, R, S) bool

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

        return valid

    # -- Internal helpers -----------------------------------------------------

    def _collect_train_anchors(self, trains: Dict[str, Train]) -> List[int]:
        """Gather anchor bays for all active trains."""
        anchors: List[int] = []
        for tid in trains:
            bay = self.rail.get_anchor_bay(tid)
            if bay is not None:
                anchors.append(bay)
        return anchors

    def _collect_truck_parking_bays(self, trucks: Dict[str, Truck]) -> List[int]:
        """Gather ACTUAL parking spot bays for parked trucks.

        Previously used container locations (circular/useless).
        Now uses the truck's real parking position.
        """
        bays: List[int] = []
        for tk in trucks.values():
            spot = getattr(tk, "parking_spot", None)
            if spot is not None:
                bay = getattr(spot, "bay", None)
                if bay is not None:
                    bays.append(bay)
        return bays


# -- Demand collection -------------------------------------------------------


def _collect_truck_demand(trucks: Dict[str, Truck]) -> Set[str]:
    """Collect all container IDs wanted by PARKED trucks."""
    wanted: Set[str] = set()
    for tk in trucks.values():
        if not getattr(tk, "parking_spot", None):
            continue  # Only parked trucks create actionable demand
        pickup_ids = getattr(tk, "pickup_container_ids", None)
        if pickup_ids:
            wanted.update(pickup_ids)
    return wanted


def _collect_train_demand(trains: Dict[str, Train]) -> Set[str]:
    """Collect all container IDs wanted by active trains."""
    wanted: Set[str] = set()
    for train in trains.values():
        wanted.update(train.get_all_pickup_container_ids())
    return wanted


# -- Module-level helpers (pure functions) ------------------------------------


def _container_type_value(c) -> float:
    """Categorical scalar for container type."""
    if getattr(c, "is_swap_body", False) or getattr(c, "is_trailer", False):
        return TYPE_SWAP
    gt = getattr(c, "goods_type", "Regular")
    if gt == "Reefer":
        return TYPE_REEFER
    if gt == "DangerousGoods":
        return TYPE_DANGEROUS
    return TYPE_REGULAR


def _direction_value(c) -> float:
    """Encode container direction: 0.0 = import, 1.0 = export."""
    direction = getattr(c, "direction", None)
    if direction is None:
        return 0.5  # Unknown
    d_str = direction.value if hasattr(direction, "value") else str(direction)
    if d_str == "Export":
        return 1.0
    if d_str == "Import":
        return 0.0
    return 0.5


def _container_hash(container_id: str) -> float:
    """Deterministic hash normalised to [0, 1].

    Gives each container a unique 'color' so the CNN can
    distinguish adjacent containers of same type/urgency.
    """
    h = hash(container_id) & 0xFFFFFFFF
    h = (h * _HASH_PRIME) % _HASH_MOD
    return h / _HASH_MOD


def _compute_urgency(c, now: datetime) -> float:
    """Normalised departure urgency: 0.0 = imminent, 1.0 = far away."""
    dep = getattr(c, "departure_date", None)
    if dep is None:
        return DEFAULT_URGENCY
    days = max(0.0, (dep - now).total_seconds() / SECONDS_PER_DAY)
    return min(1.0, days / MAX_DEPARTURE_DAYS)


def _fill_blocking(
    tensor: np.ndarray,
    urgency_grid: np.ndarray,
    R: int,
    S: int,
    T: int,
) -> None:
    """Compute BLOCKS_URGENT channel from urgency grid."""
    if T < 2:
        return

    occ = tensor[CH.OCCUPANCY]

    for t in range(1, T):
        both = (occ[:, :, t] > 0.5) & (occ[:, :, t - 1] > 0.5)
        above_urg = urgency_grid[:, :, t]
        below_urg = urgency_grid[:, :, t - 1]
        valid = both & (above_urg >= 0) & (below_urg >= 0)
        severity = np.where(valid & (above_urg > below_urg), above_urg - below_urg, 0.0)
        tensor[CH.BLOCKS_URGENT, :, :, t] = severity


def _gaussian_heat(
    anchor_bays: List[int],
    splits_f: np.ndarray,
    split_factor: int,
    sigma_bays: float,
    total_splits: int,
) -> np.ndarray:
    """Vectorised Gaussian heat over splits from a list of anchor bays."""
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