# multihead_dqn/replay_buffer.py
"""Memory-efficient replay buffer for Spatial DQN.

Transitions store only (source_pos_down, dest_pos_down) instead of
the old 6-field structure (action_type, container_pos, dest_type,
placement_pos, vehicle_idx, parking_idx).
"""
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple
from copy import copy
import numpy as np
import random


@dataclass
class Transition:
    """Single transition for the unified 2-head architecture.

    Positions are at DOWNSAMPLED resolution (s_down = s // stride)
    for direct indexing into Q-value tensors during training.
    """
    state: np.ndarray                                       # (C, R_uni, S, T)
    source_pos_down: Tuple[int, int, int]                   # (row, s_down, tier)
    dest_pos_down: Optional[Tuple[int, int, int]] = None    # (row, s_down, tier)
    reward: float = 0.0
    next_state: Optional[np.ndarray] = None                 # (C, R_uni, S, T)
    done: bool = False


# ── Compression ──────────────────────────────────────────────────────────

def _compress(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Downcast float32 state to float16."""
    if arr is None:
        return None
    return arr.astype(np.float16) if arr.dtype == np.float32 else arr


def _decompress(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Upcast float16 back to float32 for torch."""
    if arr is None:
        return None
    return arr.astype(np.float32) if arr.dtype == np.float16 else arr


# ── N-Step Return Buffer ─────────────────────────────────────────────────

class NStepBuffer:
    """Accumulates *n* transitions, computes the n-step discounted return,
    then emits a single ``(s_0, a_0, R_n, s_n, done_n)`` transition for
    the replay buffer.

    * On each ``push`` the buffer may return 0 or more ready transitions.
    * When a terminal transition arrives (``done=True``) the buffer is
      flushed so that shorter partial returns are also emitted.
    """

    def __init__(self, n: int, gamma: float):
        self.n = n
        self.gamma = gamma
        self.buffer: deque = deque(maxlen=n)

    # ── public API ───────────────────────────────────────────────────

    def push(self, transition: Transition) -> List[Transition]:
        """Push a 1-step transition.  Returns ready n-step transitions."""
        self.buffer.append(transition)

        # Episode ended → flush everything (partial windows included).
        if transition.done:
            return self._flush()

        # Buffer full → emit oldest as n-step, then drop it.
        if len(self.buffer) == self.n:
            result = [self._make_nstep()]
            self.buffer.popleft()
            return result

        return []

    def flush(self) -> List[Transition]:
        """Emit all remaining partial-window transitions (call between
        episodes / scenarios)."""
        return self._flush()

    def reset(self):
        """Discard buffer contents without emitting."""
        self.buffer.clear()

    # ── internals ────────────────────────────────────────────────────

    def _make_nstep(self) -> Transition:
        """Compute n-step return from the current buffer window."""
        R = 0.0
        for i in reversed(range(len(self.buffer))):
            t = self.buffer[i]
            R = t.reward + self.gamma * R * (1.0 - float(t.done))
        return Transition(
            state=self.buffer[0].state,
            source_pos_down=self.buffer[0].source_pos_down,
            dest_pos_down=self.buffer[0].dest_pos_down,
            reward=R,
            next_state=self.buffer[-1].next_state,
            done=self.buffer[-1].done,
        )

    def _flush(self) -> List[Transition]:
        results: List[Transition] = []
        while self.buffer:
            results.append(self._make_nstep())
            self.buffer.popleft()
        return results


# ── Replay Buffers ───────────────────────────────────────────────────────

class ReplayBuffer:
    """Circular replay buffer with float16 state compression."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Transition] = []
        self.position = 0

    def push(self, transition: Transition):
        """Add transition, compressing states to float16."""
        transition.state = _compress(transition.state)
        transition.next_state = _compress(transition.next_state)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample batch, decompressing states to float32 COPIES."""
        n = min(batch_size, len(self.buffer))
        indices = random.sample(range(len(self.buffer)), n)
        batch = []
        for i in indices:
            t = copy(self.buffer[i])
            t.state = _decompress(t.state)
            t.next_state = _decompress(t.next_state)
            batch.append(t)
        return batch

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size


class PrioritizedReplayBuffer:
    """Prioritized experience replay with float16 compression."""

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0

        self.buffer: List[Optional[Transition]] = [None] * capacity
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.max_priority = 1.0

    @property
    def beta(self) -> float:
        progress = min(1.0, self.frame / self.beta_frames)
        return self.beta_start + progress * (1.0 - self.beta_start)

    def push(self, transition: Transition):
        """Add transition with max priority."""
        transition.state = _compress(transition.state)
        transition.next_state = _compress(transition.next_state)

        self.buffer[self.position] = transition
        self.priorities[self.position] = self.max_priority ** self.alpha

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int,
    ) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """Sample with importance-sampling weights."""
        self.frame += 1
        n = min(batch_size, self.size)

        probs = self.priorities[:self.size] / self.priorities[:self.size].sum()
        indices = np.random.choice(self.size, size=n, replace=False, p=probs)

        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        transitions = []
        for i in indices:
            t = copy(self.buffer[i])
            t.state = _decompress(t.state)
            t.next_state = _decompress(t.next_state)
            transitions.append(t)

        return transitions, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        priorities = (np.abs(td_errors) + 1e-6) ** self.alpha
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())

    def __len__(self) -> int:
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        return self.size >= batch_size