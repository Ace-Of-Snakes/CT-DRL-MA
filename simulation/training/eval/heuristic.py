# simulation/training/eval/heuristic.py
"""Priority-based heuristic baseline agent.

A simple rule-based agent that mirrors the DQN ``act()`` interface.
It scans the source mask by priority order and picks the first valid
source → destination pair.  No learning, no neural network — just a
deterministic priority queue.  Useful as a lower-bound baseline.

Priority order:
    1. YARD exports with train demand  → YARD_TO_TRAIN
    2. RAIL imports                     → TRAIN_TO_YARD
    3. QUEUE trucks                     → PARK_TRUCK
    4. PARKING trucks with containers   → TRUCK_TO_YARD
    5. YARD containers with truck demand→ YARD_TO_TRUCK
    6. YARD reshuffles (fallback)       → YARD_TO_YARD
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from simulation.env.state_encoder import CH
from simulation.rl.base_agent import ActionResult
from simulation.rl.multihead_dqn.config import DEFAULT_S_STRIDE, Dims


# Type alias matching the real agent
DestMaskFn = Callable[[int, int, int], Tuple[np.ndarray, int]]


class HeuristicAgent:
    """Priority-based heuristic agent (no learning).

    Implements the same ``act()`` interface as :class:`BaseSpatialDQNAgent`
    so it can be used directly with :class:`TutorialRunner` and
    :func:`run_eval_matrix`.
    """

    is_trainable: bool = False

    def __init__(self, dims: Dims):
        self.dims = dims
        self._s_stride = DEFAULT_S_STRIDE
        self._s_down = dims.n_splits // self._s_stride

        # Compatibility attributes
        self.epsilon_override: Optional[float] = None
        self.step_count: int = 0

    # ── act: core interface ───────────────────────────────────────

    def act(
        self,
        state: np.ndarray,
        source_mask: np.ndarray,
        dest_mask_fn: DestMaskFn,
        epsilon: Optional[float] = None,
    ) -> ActionResult:
        """Select action via priority scan.

        Args:
            state: ``(C, R, S, T)`` float32 tensor.
            source_mask: ``(R, S, T)`` bool array — actionable sources.
            dest_mask_fn: Takes ``(row, split, tier)`` → ``(mask, n_splits)``.
            epsilon: Ignored (no exploration).

        Returns:
            :class:`ActionResult` with source/dest positions and regions.
        """
        if not source_mask.any():
            return ActionResult()

        d = self.dims

        # Try each priority in order; return the first that succeeds.
        for scanner in self._scanners():
            result = scanner(state, source_mask, dest_mask_fn)
            if result is not None:
                return result

        # Nothing found at all — idle
        return ActionResult()

    # ── Priority scanners ─────────────────────────────────────────

    def _scanners(self):
        """Return ordered list of scanning functions."""
        return [
            self._scan_yard_exports,
            self._scan_rail_imports,
            self._scan_queue_trucks,
            self._scan_parking_trucks,
            self._scan_yard_pickups,
            self._scan_yard_reshuffles,
        ]

    def _scan_yard_exports(self, state, source_mask, dest_mask_fn):
        """YARD exports with train demand → YARD_TO_TRAIN."""
        d = self.dims
        for row in range(d.yard_row_start, d.yard_row_end):
            for split in range(d.n_splits):
                for tier in range(d.n_tiers):
                    if not source_mask[row, split, tier]:
                        continue
                    # Must be export direction AND demanded by a train
                    if (state[CH.DIRECTION, row, split, tier] > 0.7
                            and state[CH.EXPORT_PICKUP_DEMAND, row, split, tier] > 0.05):
                        result = self._try_dest(
                            state, dest_mask_fn, row, split, tier)
                        if result is not None:
                            return result
        return None

    def _scan_rail_imports(self, state, source_mask, dest_mask_fn):
        """RAIL imports → TRAIN_TO_YARD."""
        d = self.dims
        for row in range(d.rail_row_start, d.rail_row_end):
            for split in range(d.n_splits):
                for tier in range(d.n_tiers):
                    if not source_mask[row, split, tier]:
                        continue
                    result = self._try_dest(
                        state, dest_mask_fn, row, split, tier)
                    if result is not None:
                        return result
        return None

    def _scan_queue_trucks(self, state, source_mask, dest_mask_fn):
        """QUEUE trucks → PARK_TRUCK."""
        d = self.dims
        for row in range(d.queue_row_start, d.queue_row_end):
            for split in range(d.n_splits):
                for tier in range(d.n_tiers):
                    if not source_mask[row, split, tier]:
                        continue
                    result = self._try_dest(
                        state, dest_mask_fn, row, split, tier)
                    if result is not None:
                        return result
        return None

    def _scan_parking_trucks(self, state, source_mask, dest_mask_fn):
        """PARKING delivery trucks → TRUCK_TO_YARD."""
        d = self.dims
        for row in range(d.parking_row_start, d.parking_row_end):
            for split in range(d.n_splits):
                for tier in range(d.n_tiers):
                    if not source_mask[row, split, tier]:
                        continue
                    result = self._try_dest(
                        state, dest_mask_fn, row, split, tier)
                    if result is not None:
                        return result
        return None

    def _scan_yard_pickups(self, state, source_mask, dest_mask_fn):
        """YARD containers demanded by truck → YARD_TO_TRUCK."""
        d = self.dims
        for row in range(d.yard_row_start, d.yard_row_end):
            for split in range(d.n_splits):
                for tier in range(d.n_tiers):
                    if not source_mask[row, split, tier]:
                        continue
                    if state[CH.TRUCK_DEMAND, row, split, tier] > 0.05:
                        result = self._try_dest(
                            state, dest_mask_fn, row, split, tier)
                        if result is not None:
                            return result
        return None

    def _scan_yard_reshuffles(self, state, source_mask, dest_mask_fn):
        """Fallback: any accessible yard container → YARD_TO_YARD."""
        d = self.dims
        for row in range(d.yard_row_start, d.yard_row_end):
            for split in range(d.n_splits):
                for tier in range(d.n_tiers):
                    if not source_mask[row, split, tier]:
                        continue
                    result = self._try_dest(
                        state, dest_mask_fn, row, split, tier)
                    if result is not None:
                        return result
        return None

    # ── Destination selection ─────────────────────────────────────

    def _try_dest(
        self,
        state: np.ndarray,
        dest_mask_fn: DestMaskFn,
        src_row: int,
        src_split: int,
        src_tier: int,
    ) -> Optional[ActionResult]:
        """Call dest_mask_fn for a source and pick the first valid dest.

        Returns an :class:`ActionResult` or ``None`` if no valid
        destination exists.
        """
        dest_mask, n_splits = dest_mask_fn(src_row, src_split, src_tier)
        if dest_mask is None or not dest_mask.any():
            return None

        # dest_mask shape: (R, S_down, T)
        # Find first valid destination
        candidates = np.argwhere(dest_mask)
        if len(candidates) == 0:
            return None

        # Pick first candidate (deterministic)
        dst_row, dst_s_down, dst_tier = candidates[0]

        # Resolve regions
        d = self.dims
        src_region = d.region_of(src_row)
        dst_region = d.region_of(int(dst_row))

        # Resolve move type
        move_type = _resolve_move_type(src_region, dst_region)

        # Resolve full-resolution destination split
        dst_split = int(dst_s_down) * self._s_stride
        # For occupied destinations (train, truck), find the actual entity
        if dst_region in ("RAIL", "PARKING"):
            # Scan the stride window for the entity start
            s_start = int(dst_s_down) * self._s_stride
            s_end = min(s_start + self._s_stride, d.n_splits)
            for s in range(s_start, s_end):
                if state[CH.CONTAINER_START, int(dst_row), s, int(dst_tier)] > 0.5:
                    dst_split = s
                    break
                if state[CH.OCCUPANCY, int(dst_row), s, int(dst_tier)] > 0.5:
                    dst_split = s
                    break

        return ActionResult(
            source_pos=(int(src_row), int(src_split), int(src_tier)),
            dest_pos=(int(dst_row), int(dst_split), int(dst_tier)),
            source_region=src_region,
            dest_region=dst_region,
            move_type=move_type,
            source_pos_down=(
                int(src_row),
                int(src_split) // self._s_stride,
                int(src_tier),
            ),
            dest_pos_down=(int(dst_row), int(dst_s_down), int(dst_tier)),
        )

    # ── Compatibility stubs ───────────────────────────────────────
    # These are called by TutorialRunner / run_eval_matrix but are
    # meaningless for a non-learning agent.

    def remember(self, transition) -> None:
        """No replay buffer."""

    def reset_nstep(self) -> None:
        """No n-step returns."""

    def optimize(self) -> float:
        """No learning."""
        return 0.0

    def save(self, path: str) -> None:
        """Nothing to save."""

    def load(self, path: str, **kwargs) -> None:
        """Nothing to load."""

    def set_noise_scale(self, scale: float) -> None:
        """No noisy layers."""

    def set_tutorial_epsilon(self, epoch: int) -> None:
        """No epsilon schedule — deterministic agent."""

    def set_epsilon_override(self, eps: float) -> None:
        self.epsilon_override = eps

    def clear_epsilon_override(self) -> None:
        self.epsilon_override = None

    def _get_epsilon(self) -> float:
        return 0.0


# ── Move type lookup ──────────────────────────────────────────────

_MOVE_TYPE_MAP = {
    ("QUEUE", "PARKING"): "PARK_TRUCK",
    ("RAIL", "YARD"): "TRAIN_TO_YARD",
    ("PARKING", "YARD"): "TRUCK_TO_YARD",
    ("YARD", "YARD"): "YARD_TO_YARD",
    ("YARD", "RAIL"): "YARD_TO_TRAIN",
    ("YARD", "PARKING"): "YARD_TO_TRUCK",
    ("RAIL", "PARKING"): "TRAIN_TO_TRUCK",
    ("PARKING", "RAIL"): "TRUCK_TO_TRAIN",
}


def _resolve_move_type(src_region: str, dst_region: str) -> Optional[str]:
    return _MOVE_TYPE_MAP.get((src_region, dst_region))
