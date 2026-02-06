# multihead_dqn/config.py
"""Configuration for Factored-CNN Multi-Head DQN agent."""
from dataclasses import dataclass, field
from typing import Tuple
from enum import IntEnum
import torch


class ActionType(IntEnum):
    """Top-level action types."""
    MOVE_CONTAINER = 0
    SLOT_PARKING = 1
    IMPORT_VEHICLE = 2


class DestinationType(IntEnum):
    """Destination types for container moves."""
    YARD = 0
    TRAIN = 1
    TRUCK = 2


@dataclass
class YardDims:
    """Yard dimensions."""
    n_rows: int
    n_splits: int
    n_tiers: int
    n_bays: int
    split_factor: int  # splits per bay (typically 20)
    n_parking_rows: int = 1  # extra rows appended for parking encoding

    @property
    def n_state_rows(self) -> int:
        """Total rows in state tensor (yard + parking)."""
        return self.n_rows + self.n_parking_rows

    @property
    def spatial_shape(self) -> Tuple[int, int, int]:
        return (self.n_rows, self.n_splits, self.n_tiers)

    @property
    def total_positions(self) -> int:
        return self.n_rows * self.n_splits * self.n_tiers


@dataclass
class CNNConfig:
    """Factored CNN backbone configuration.

    Kernel design:
      (1, container_kernel, 1) along S Ã¢â‚¬â€ matches container shape
      (cross_kernel, 1, cross_kernel) across RÃƒâ€”T Ã¢â‚¬â€ stacking context
      (1, refine_kernel, 1) along S Ã¢â‚¬â€ neighborhood awareness

    Total receptive field:
      S Ã¢â€°Ë† 57 splits (~3 bays),  R = 3 rows,  T = 3 tiers
    """
    n_state_channels: int = 12
    stage1_channels: int = 32        # first conv output channels
    feat_channels: int = 64          # backbone output channels
    global_dim: int = 128            # after occupied-only pooling
    s_stride: int = 4                # downsample factor along splits
    gn_groups: int = 8               # GroupNorm groups
    # Factored kernel sizes
    container_kernel: int = 21       # (1, k, 1) Ã¢â‚¬â€ 20ft container = 20 splits
    cross_kernel: int = 3            # (r, 1, t) Ã¢â‚¬â€ stacking / cross-row
    refine_kernel: int = 5           # (1, k, 1) Ã¢â‚¬â€ neighborhood

    @property
    def container_pad(self) -> int:
        """Symmetric padding for container-axis kernels."""
        return self.container_kernel // 2

    @property
    def cross_pad(self) -> int:
        """Symmetric padding for cross-row/tier kernels."""
        return self.cross_kernel // 2

    @property
    def refine_pad(self) -> int:
        """Symmetric padding for refinement kernel."""
        return self.refine_kernel // 2


@dataclass
class HeadConfig:
    """Configuration for decision heads."""
    hidden: int = 64                 # hidden size for MLP heads
    vehicle_feat_dim: int = 8        # external vehicle feature dim
    dueling: bool = True             # dueling V+A for fixed-size heads
    proximity_bays: int = 3          # OperationsDefaults.PROXIMITY_SEARCH_BAYS


@dataclass
class DQNConfig:
    """Training configuration."""
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 32
    replay_size: int = 5_000
    target_tau: float = 0.005
    grad_clip: float = 1.0

    # Epsilon (curriculum stages, step-based)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000

    # Tutorial epsilon (epoch-based, fast decay)
    tutorial_epsilon_start: float = 0.9
    tutorial_epsilon_end: float = 0.05
    tutorial_epsilon_epochs: int = 80

    double_dqn: bool = True
    n_step: int = 3

    # Dest-type exploration floor (prevents catastrophic forgetting
    # when main epsilon is low and vehicle destinations are rare)
    dest_epsilon_floor: float = 0.20

    # Auxiliary dest_type loss weight
    dest_aux_weight: float = 0.5

    # Prioritized replay
    use_per: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 100_000


@dataclass
class MultiHeadDQNConfig:
    """Complete agent configuration."""
    yard: YardDims = None
    cnn: CNNConfig = field(default_factory=CNNConfig)
    heads: HeadConfig = field(default_factory=HeadConfig)
    training: DQNConfig = field(default_factory=DQNConfig)
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        if self.yard is None:
            raise ValueError("YardDims must be provided")
        if self.yard.n_splits % self.cnn.s_stride != 0:
            raise ValueError(
                f"n_splits ({self.yard.n_splits}) must be divisible by "
                f"s_stride ({self.cnn.s_stride})"
            )

    @property
    def s_down(self) -> int:
        """Downsampled split count after CNN stride."""
        return self.yard.n_splits // self.cnn.s_stride