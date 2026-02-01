# multihead_dqn/config.py
"""Configuration for Multi-Head DQN agent."""
from dataclasses import dataclass, field
from typing import List, Tuple
from enum import IntEnum
import torch


class ActionType(IntEnum):
    """Top-level action types."""
    MOVE_CONTAINER = 0
    SLOT_PARKING = 1


class DestinationType(IntEnum):
    """Destination types for container moves."""
    YARD = 0
    TRAIN = 1
    TRUCK = 2


@dataclass
class YardDims:
    """Yard dimensions at split resolution."""
    n_rows: int
    n_splits: int  # total_splits = n_bays * split_factor
    n_tiers: int
    n_bays: int
    split_factor: int
    
    @property
    def spatial_shape(self) -> Tuple[int, int, int]:
        """Returns (R, S, T) for tensor shapes."""
        return (self.n_rows, self.n_splits, self.n_tiers)
    
    @property
    def total_positions(self) -> int:
        return self.n_rows * self.n_splits * self.n_tiers


@dataclass
class BackboneConfig:
    """CNN backbone configuration."""
    in_channels: int = 8
    hidden_channels: List[int] = field(default_factory=lambda: [32, 64, 64])
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3])
    use_batchnorm: bool = True
    dropout: float = 0.0


@dataclass
class HeadConfig:
    """Configuration for decision heads."""
    global_hidden: int = 128
    spatial_features: int = 64  # feature dim before spatial head
    vehicle_feat_dim: int = 8   # features per train/truck
    max_vehicles: int = 20      # max trains + trucks to consider


@dataclass
class DQNConfig:
    """Training configuration."""
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 32
    replay_size: int = 100_000
    target_tau: float = 0.005
    grad_clip: float = 1.0
    
    # Epsilon schedule
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000
    
    # Multi-step returns (optional)
    n_step: int = 1
    
    # Prioritized replay (optional)
    use_per: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 100_000


@dataclass
class MultiHeadDQNConfig:
    """Complete configuration."""
    yard: YardDims = None  # Must be set
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    heads: HeadConfig = field(default_factory=HeadConfig)
    training: DQNConfig = field(default_factory=DQNConfig)
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    def __post_init__(self):
        if self.yard is None:
            raise ValueError("YardDims must be provided")