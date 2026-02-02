# simulation/env/state_encoder.py
"""8-channel split-level state encoder for Multi-Head DQN.

Output shape: (C, R, S, T) where
  C = 8 channels
  R = n_rows
  S = total_splits (n_bays * split_factor)
  T = n_tiers
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from simulation.core.facilities.yard import OptimizedStorageYard
from simulation.core.facilities.railyard import OptimizedRailYard
from simulation.core.vehicles.train import Train
from simulation.core.vehicles.truck import Truck


# ── Channel specification ──────────────────────────────────────
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

    @staticmethod
    def num_channels() -> int:
        return 8


CH = ChannelSpec()
NUM_CHANNELS = CH.num_channels()

# ── Container type categorical values ──────────────────────────
TYPE_REGULAR: float = 0.25
TYPE_REEFER: float = 0.50
TYPE_DANGEROUS: float = 0.75
TYPE_SWAP: float = 1.00

# ── Normalisation / heat constants ─────────────────────────────
MAX_DEPARTURE_DAYS: float = 30.0
SECONDS_PER_DAY: float = 86_400.0
TRAIN_HEAT_SIGMA_BAYS: float = 3.0
TRUCK_HEAT_SIGMA_BAYS: float = 2.0
DEFAULT_URGENCY: float = 1.0  # far away = not urgent


class SplitLevelStateEncoder:
    """Encodes terminal state as (C, R, S, T) tensor at split resolution."""

    __slots__ = ("yard", "rail", "_splits_f", "_split_factor")

    def __init__(self, yard: OptimizedStorageYard, rail: OptimizedRailYard):
        self.yard = yard
        self.rail = rail
        self._split_factor = yard.split_factor
        # Pre-allocated coordinate vector for vectorised heat
        self._splits_f = np.arange(yard.total_splits, dtype=np.float32)

    # ── Public API ─────────────────────────────────────────────

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

        # ── Ch 0  OCCUPANCY  (from yard mask, transpose T,R,S → R,S,T) ──
        tensor[CH.OCCUPANCY] = self.yard.occupancy_mask.transpose(1, 2, 0).astype(
            np.float32
        )

        # ── Per-container channels (single pass) ─────────────────
        # Urgency grid for blocking analysis (default = far away)
        urgency_grid = np.full((R, S, T), -1.0, dtype=np.float32)

        for rec in self.yard.iter_records():
            c = rec.container
            pl = rec.placement
            r, t = pl.row, pl.tier
            s0 = pl.abs_start
            s1 = min(s0 + rec.n_splits, S)

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
            urgency_grid[r, s0, t] = urg  # store at start for blocking

        # ── Ch 5  BLOCKS_URGENT  (vectorised over tiers) ─────────
        _fill_blocking(tensor, urgency_grid, R, S, T)

        # ── Ch 6  TRAIN_HEAT ─────────────────────────────────────
        train_anchors = self._collect_train_anchors(trains)
        if train_anchors:
            heat = _gaussian_heat(
                train_anchors, self._splits_f, self._split_factor,
                TRAIN_HEAT_SIGMA_BAYS, S,
            )
            tensor[CH.TRAIN_HEAT, :, :, :] = heat[np.newaxis, :, np.newaxis]

        # ── Ch 7  TRUCK_HEAT ─────────────────────────────────────
        truck_bays = self._collect_truck_bays(trucks)
        if truck_bays:
            heat = _gaussian_heat(
                truck_bays, self._splits_f, self._split_factor,
                TRUCK_HEAT_SIGMA_BAYS, S,
            )
            tensor[CH.TRUCK_HEAT, :, :, :] = heat[np.newaxis, :, np.newaxis]

        return tensor

    # ── Mask helpers (used by agent for action masking) ─────────

    def get_occupancy_mask(self) -> np.ndarray:
        """Occupancy as (R, S, T) bool for container selection masking."""
        return self.yard.occupancy_mask.transpose(1, 2, 0)

    def get_validity_mask(
        self,
        n_splits_needed: int,
        goods_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Valid placement positions as (R, S, T) bool.

        Uses vectorised cumsum to check contiguous free splits.
        """
        R = self.yard.n_rows
        S = self.yard.total_splits
        T = self.yard.n_tiers
        occ = self.yard.occupancy_mask  # (T, R, S) bool

        valid = np.zeros((R, S, T), dtype=bool)

        for t in range(T):
            # Support check: tier > 0 needs full support below
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

                # Apply support mask
                if support is not None:
                    sup_row = support[r, :]
                    for s in range(S - n_splits_needed + 1):
                        if valid[r, s, t] and not np.all(
                            sup_row[s : s + n_splits_needed]
                        ):
                            valid[r, s, t] = False

        # Apply goods-type zoning mask
        if goods_mask is not None:
            valid &= goods_mask

        return valid

    # ── Internal helpers ───────────────────────────────────────

    def _collect_train_anchors(self, trains: Dict[str, Train]) -> List[int]:
        """Gather anchor bays for all active trains."""
        anchors: List[int] = []
        for tid in trains:
            bay = self.rail.get_anchor_bay(tid)
            if bay is not None:
                anchors.append(bay)
        return anchors

    def _collect_truck_bays(self, trucks: Dict[str, Truck]) -> List[int]:
        """Estimate target bays from truck pickup containers."""
        bays: List[int] = []
        default_bay = self.yard.n_bays // 2
        for tk in trucks.values():
            pickup_ids = getattr(tk, "pickup_container_ids", None)
            if not pickup_ids:
                bays.append(default_bay)
                continue
            found: List[int] = []
            for cid in pickup_ids:
                pl = self.yard.get_placement(cid)
                if pl:
                    found.append(pl.bay)
            if found:
                found.sort()
                bays.append(found[len(found) // 2])
            else:
                bays.append(default_bay)
        return bays


# ── Module-level helpers (pure functions) ──────────────────────


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
    """Compute BLOCKS_URGENT channel from urgency grid.

    A container at tier t blocks if its urgency > the urgency of the
    container at the same (row, start_split) at tier t-1.
    Blocking severity = difference in normalised urgency.
    """
    if T < 2:
        return

    occ = tensor[CH.OCCUPANCY]  # (R, S, T)

    for t in range(1, T):
        # Positions where both current tier and below are occupied
        both = (occ[:, :, t] > 0.5) & (occ[:, :, t - 1] > 0.5)
        # Urgency values at start positions (only valid where grid >= 0)
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