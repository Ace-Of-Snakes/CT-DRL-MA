# multihead_dqn/replay_buffer.py
"""Memory-efficient replay buffer for Multi-Head DQN.

States are stored as float16 to halve memory usage (the yard tensor
at split resolution is ~1.4 MB per state in float32).
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import random

from simulation.rl.multihead_dqn.config import ActionType, DestinationType


@dataclass
class Transition:
    """Complete transition capturing all decision stages."""
    # State (stored as float16 for memory efficiency)
    state: np.ndarray               # (C, R, S, T) float16

    # Stage 1: Action type
    action_type: int                # ActionType enum value

    # Stage 2: Container or Parking selection
    container_pos: Optional[Tuple[int, int, int]] = None
    parking_idx: int = -1

    # Stage 3: Destination type (only if MOVE)
    dest_type: int = -1

    # Stage 4: Placement or Vehicle (only if MOVE)
    placement_pos: Optional[Tuple[int, int, int]] = None
    vehicle_idx: int = -1

    # Outcome
    reward: float = 0.0
    next_state: Optional[np.ndarray] = None  # float16
    done: bool = False

    # Masks (stored sparsely)
    occupancy_mask_indices: Optional[np.ndarray] = None
    validity_mask_indices: Optional[np.ndarray] = None
    vehicle_mask: Optional[np.ndarray] = None
    parking_mask: Optional[np.ndarray] = None


def _compress(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Downcast float32 state to float16."""
    if arr is None:
        return None
    if arr.dtype == np.float32:
        return arr.astype(np.float16)
    return arr


def _decompress(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Upcast float16 back to float32 for torch."""
    if arr is None:
        return None
    if arr.dtype == np.float16:
        return arr.astype(np.float32)
    return arr


class ReplayBuffer:
    """Replay buffer with float16 state compression."""

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
        """Sample batch, decompressing states to float32 COPIES (not in-place)."""
        from copy import copy
        n = min(batch_size, len(self.buffer))
        indices = random.sample(range(len(self.buffer)), n)
        batch = []
        for i in indices:
            t = copy(self.buffer[i])  # shallow copy
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
        beta_frames: int = 100_000
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
        """Add transition with max priority, compressing states."""
        transition.state = _compress(transition.state)
        transition.next_state = _compress(transition.next_state)

        self.buffer[self.position] = transition
        self.priorities[self.position] = self.max_priority ** self.alpha

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """Sample with importance-sampling weights, decompressing to COPIES."""
        from copy import copy
        self.frame += 1
        n = min(batch_size, self.size)

        probs = self.priorities[:self.size] / self.priorities[:self.size].sum()
        indices = np.random.choice(self.size, size=n, replace=False, p=probs)

        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        transitions = []
        for i in indices:
            t = copy(self.buffer[i])  # shallow copy
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