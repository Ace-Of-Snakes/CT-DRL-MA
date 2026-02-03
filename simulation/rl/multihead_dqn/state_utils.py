# simulation/rl/multihead_dqn/state_utils.py
"""State encoding utilities -- re-exports canonical ChannelSpec.

The production encoder lives in simulation.env.state_encoder.
This module provides a lightweight SplitLevelEncoder for standalone
agent tests that don't require OptimizedStorageYard.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple

# "" Canonical channel specification (single source of truth) """
from simulation.env.state_encoder import (
    ChannelSpec,
    CH,
    NUM_CHANNELS,
    _container_type_value,
    _compute_urgency,
    _gaussian_heat,
    DEFAULT_URGENCY,
    TRAIN_HEAT_SIGMA_BAYS,
    TRUCK_HEAT_SIGMA_BAYS,
)


class SplitLevelEncoder:
    """Lightweight encoder for standalone agent testing.

    Production code should use SplitLevelStateEncoder from
    simulation.env.state_encoder instead.
    """

    __slots__ = (
        "n_rows", "n_bays", "n_tiers", "split_factor",
        "total_splits", "_splits_f",
    )

    def __init__(
        self, n_rows: int, n_bays: int, n_tiers: int, split_factor: int
    ):
        self.n_rows = n_rows
        self.n_bays = n_bays
        self.n_tiers = n_tiers
        self.split_factor = split_factor
        self.total_splits = n_bays * split_factor
        self._splits_f = np.arange(self.total_splits, dtype=np.float32)

    def encode(
        self,
        containers: Dict,
        container_length_map: Dict[float, int],
        current_time,
        train_anchors: Optional[Dict[str, int]] = None,
        truck_bays: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Encode yard state from raw dicts -- for testing only."""
        R, S, T = self.n_rows, self.total_splits, self.n_tiers
        tensor = np.zeros((NUM_CHANNELS, R, S, T), dtype=np.float32)

        urgency_grid = np.full((R, S, T), -1.0, dtype=np.float32)

        for _cid, rec in containers.items():
            c = rec.container
            pl = rec.placement
            r, t = pl.row, pl.tier
            s0 = pl.bay * self.split_factor + pl.start_split
            s1 = min(s0 + rec.n_splits, S)

            tensor[CH.OCCUPANCY, r, s0:s1, t] = 1.0
            tensor[CH.CONTAINER_START, r, s0, t] = 1.0
            tensor[CH.CONTAINER_TYPE, r, s0:s1, t] = _container_type_value(c)
            if rec.is_accessible:
                tensor[CH.ACCESSIBLE, r, s0:s1, t] = 1.0

            urg = _compute_urgency(c, current_time)
            tensor[CH.DEPARTURE_URGENCY, r, s0:s1, t] = urg
            urgency_grid[r, s0, t] = urg

        # Blocking
        occ = tensor[CH.OCCUPANCY]
        for t in range(1, T):
            both = (occ[:, :, t] > 0.5) & (occ[:, :, t - 1] > 0.5)
            above = urgency_grid[:, :, t]
            below = urgency_grid[:, :, t - 1]
            valid = both & (above >= 0) & (below >= 0)
            tensor[CH.BLOCKS_URGENT, :, :, t] = np.where(
                valid & (above > below), above - below, 0.0
            )

        # Heat channels
        if train_anchors:
            heat = _gaussian_heat(
                list(train_anchors.values()), self._splits_f,
                self.split_factor, TRAIN_HEAT_SIGMA_BAYS, S,
            )
            tensor[CH.TRAIN_HEAT, :, :, :] = heat[np.newaxis, :, np.newaxis]

        if truck_bays:
            heat = _gaussian_heat(
                truck_bays, self._splits_f,
                self.split_factor, TRUCK_HEAT_SIGMA_BAYS, S,
            )
            tensor[CH.TRUCK_HEAT, :, :, :] = heat[np.newaxis, :, np.newaxis]

        return tensor

    @staticmethod
    def get_occupancy_mask(tensor: np.ndarray) -> np.ndarray:
        """Extract occupancy mask from state tensor."""
        return tensor[CH.OCCUPANCY] > 0.5

    def get_validity_mask(
        self,
        occupancy_mask: np.ndarray,
        n_splits_needed: int,
        goods_mask: Optional[np.ndarray] = None,
        support_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Valid placement mask using vectorised cumsum."""
        R, S, T = occupancy_mask.shape
        valid = np.zeros((R, S, T), dtype=bool)

        for t in range(T):
            for r in range(R):
                free = (~occupancy_mask[r, :, t]).astype(np.int32)
                cs = np.empty(S + 1, dtype=np.int32)
                cs[0] = 0
                np.cumsum(free, out=cs[1:])

                end_idx = np.arange(n_splits_needed, S + 1)
                start_idx = end_idx - n_splits_needed
                contiguous = (cs[end_idx] - cs[start_idx]) == n_splits_needed
                valid[r, :S - n_splits_needed + 1, t] = contiguous

                if support_mask is not None and t > 0:
                    for s in range(S - n_splits_needed + 1):
                        if valid[r, s, t] and not np.all(
                            support_mask[r, s:s + n_splits_needed, t]
                        ):
                            valid[r, s, t] = False

        if goods_mask is not None:
            valid &= goods_mask
        return valid