# multihead_dqn/replay_buffer.py
"""Replay buffer for Multi-Head DQN."""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import random

from simulation.rl.multihead_dqn.config import ActionType, DestinationType


@dataclass
class Transition:
    """
    Complete transition capturing all decision stages.
    
    Decision flow:
    1. action_type: MOVE_CONTAINER or SLOT_PARKING
    2a. If MOVE: container_pos (which container)
    2b. If PARK: parking_idx (which parking action)
    3. If MOVE: dest_type (YARD/TRAIN/TRUCK)
    4a. If YARD: placement_pos
    4b. If TRAIN/TRUCK: vehicle_idx
    """
    # State (stored as numpy for memory efficiency)
    state: np.ndarray               # (C, R, S, T)
    
    # Stage 1: Action type
    action_type: int                # ActionType enum value
    
    # Stage 2: Container or Parking selection
    container_pos: Optional[Tuple[int, int, int]] = None  # (row, split, tier) if MOVE
    parking_idx: int = -1           # Index in parking options if PARK
    
    # Stage 3: Destination type (only if MOVE)
    dest_type: int = -1             # DestinationType enum value
    
    # Stage 4: Placement or Vehicle (only if MOVE)
    placement_pos: Optional[Tuple[int, int, int]] = None  # (row, split, tier) if YARD
    vehicle_idx: int = -1           # Index in vehicle list if TRAIN/TRUCK
    
    # Outcome
    reward: float = 0.0
    next_state: Optional[np.ndarray] = None
    done: bool = False
    
    # Masks for reconstruction (stored sparsely)
    occupancy_mask_indices: Optional[np.ndarray] = None  # Indices where occupied
    validity_mask_indices: Optional[np.ndarray] = None   # Indices where placement valid
    vehicle_mask: Optional[np.ndarray] = None            # (V,) bool
    parking_mask: Optional[np.ndarray] = None            # (P,) bool


@dataclass
class TransitionBatch:
    """Batched transitions for training."""
    states: np.ndarray              # (B, C, R, S, T)
    action_types: np.ndarray        # (B,) int
    container_positions: np.ndarray # (B, 3) int, -1 for non-MOVE
    parking_indices: np.ndarray     # (B,) int, -1 for non-PARK
    dest_types: np.ndarray          # (B,) int
    placement_positions: np.ndarray # (B, 3) int, -1 for non-YARD
    vehicle_indices: np.ndarray     # (B,) int
    rewards: np.ndarray             # (B,)
    next_states: np.ndarray         # (B, C, R, S, T)
    dones: np.ndarray               # (B,) bool


class ReplayBuffer:
    """Standard replay buffer with uniform sampling."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Transition] = []
        self.position = 0
    
    def push(self, transition: Transition):
        """Add transition to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Transition]:
        """Sample random batch of transitions."""
        n = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, n)
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return len(self.buffer) >= batch_size


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer."""
    
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
        """Linearly anneal beta from beta_start to 1.0."""
        progress = min(1.0, self.frame / self.beta_frames)
        return self.beta_start + progress * (1.0 - self.beta_start)
    
    def push(self, transition: Transition):
        """Add transition with max priority."""
        self.buffer[self.position] = transition
        self.priorities[self.position] = self.max_priority ** self.alpha
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """
        Sample batch with importance sampling weights.
        Returns: (transitions, indices, weights)
        """
        self.frame += 1
        n = min(batch_size, self.size)
        
        # Compute sampling probabilities
        probs = self.priorities[:self.size] / self.priorities[:self.size].sum()
        indices = np.random.choice(self.size, size=n, replace=False, p=probs)
        
        # Importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        transitions = [self.buffer[i] for i in indices]
        return transitions, indices, weights.astype(np.float32)
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        priorities = (np.abs(td_errors) + 1e-6) ** self.alpha
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())
    
    def __len__(self) -> int:
        return self.size
    
    def is_ready(self, batch_size: int) -> bool:
        return self.size >= batch_size