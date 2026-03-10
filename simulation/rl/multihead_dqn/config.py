# multihead_dqn/config.py
"""Configuration for Spatial DQN agent."""
from dataclasses import dataclass, field
from typing import Tuple
import torch


# ── Shared constants ──────────────────────────────────────────────────────
DEFAULT_S_STRIDE: int = 4  # Spatial stride for downsampling splits axis

# ── Spatial dimensions ────────────────────────────────────────────────────

@dataclass
class Dims:
    """Dimensions for the spatial state matrix.

    Layout (row order):
        Rail tracks  → Parking row → Yard rows → Queue rows

    The CNN (3,1,3) kernel bridges adjacent regions:
        Rail ↔ Parking ↔ Yard ↔ Queue
    """
    n_tracks: int = 7
    n_parking_rows: int = 1
    n_yard_rows: int = 5
    n_queue_rows: int = 2
    n_bays: int = 58
    split_factor: int = 20
    n_tiers: int = 5

    # ── Derived dimensions ────────────────────────────────────────────

    @property
    def R_unified(self) -> int:
        """Total rows in unified state matrix."""
        return self.n_tracks + self.n_parking_rows + self.n_yard_rows + self.n_queue_rows

    @property
    def n_splits(self) -> int:
        """Total split positions along bay axis."""
        return self.n_bays * self.split_factor

    @property
    def spatial_shape(self) -> Tuple[int, int, int]:
        """(R, S, T) shape of the unified state matrix."""
        return (self.R_unified, self.n_splits, self.n_tiers)

    # ── Region row ranges ─────────────────────────────────────────────

    @property
    def rail_row_start(self) -> int:
        return 0

    @property
    def rail_row_end(self) -> int:
        """Exclusive end of rail rows."""
        return self.n_tracks

    @property
    def parking_row_start(self) -> int:
        return self.n_tracks

    @property
    def parking_row_end(self) -> int:
        """Exclusive end of parking rows."""
        return self.n_tracks + self.n_parking_rows

    @property
    def yard_row_start(self) -> int:
        return self.n_tracks + self.n_parking_rows

    @property
    def yard_row_end(self) -> int:
        """Exclusive end of yard rows."""
        return self.n_tracks + self.n_parking_rows + self.n_yard_rows

    @property
    def queue_row_start(self) -> int:
        return self.n_tracks + self.n_parking_rows + self.n_yard_rows

    @property
    def queue_row_end(self) -> int:
        """Exclusive end of queue rows."""
        return self.R_unified

    # ── Convenience ranges ────────────────────────────────────────────

    @property
    def rail_rows(self) -> range:
        return range(self.rail_row_start, self.rail_row_end)

    @property
    def parking_rows(self) -> range:
        return range(self.parking_row_start, self.parking_row_end)

    @property
    def yard_rows(self) -> range:
        return range(self.yard_row_start, self.yard_row_end)

    @property
    def queue_rows(self) -> range:
        return range(self.queue_row_start, self.queue_row_end)

    def region_of(self, row: int) -> str:
        """Return region name for a given row index."""
        if row < self.rail_row_end:
            return "RAIL"
        if row < self.parking_row_end:
            return "PARKING"
        if row < self.yard_row_end:
            return "YARD"
        return "QUEUE"


# ── Legacy YardDims (kept for backward compat) ───────────────────────────

@dataclass
class YardDims:
    """Yard dimensions (DEPRECATED — use Dims)."""
    n_rows: int
    n_splits: int
    n_tiers: int
    n_bays: int
    split_factor: int
    n_parking_rows: int = 1

    @property
    def n_state_rows(self) -> int:
        return self.n_rows + self.n_parking_rows

    @property
    def spatial_shape(self) -> Tuple[int, int, int]:
        return (self.n_rows, self.n_splits, self.n_tiers)

    @property
    def total_positions(self) -> int:
        return self.n_rows * self.n_splits * self.n_tiers


# ── CNN backbone config ──────────────────────────────────────────────────

@dataclass
class CNNConfig:
    """Factored CNN backbone configuration.

    Kernel design:
      (1, container_kernel, 1) along S → container profile extraction
      (cross_kernel, 1, cross_kernel) across R×T → stacking + cross-region
      (1, refine_kernel, 1) along S → neighbourhood refinement
    """
    n_state_channels: int = 14   # 14 channels (10 base + crane, train urgency, time, export demand)
    stage1_channels: int = 32
    feat_channels: int = 64
    global_dim: int = 128
    s_stride: int = DEFAULT_S_STRIDE
    gn_groups: int = 8
    container_kernel: int = 21   # (1, k, 1) — 20ft = 20 splits
    cross_kernel: int = 3        # (r, 1, t) — stacking / cross-row
    refine_kernel: int = 5       # (1, k, 1) — neighbourhood

    # Region-aware backbone kernel sizes
    container_region_kernel: int = 11  # (1, k, 1) — smallest container = 10 splits + 1
    bay_region_kernel: int = 21        # (1, k, 1) — one bay = 20 splits + 1

    @property
    def container_pad(self) -> int:
        return self.container_kernel // 2

    @property
    def cross_pad(self) -> int:
        return self.cross_kernel // 2

    @property
    def refine_pad(self) -> int:
        return self.refine_kernel // 2

    @property
    def container_region_pad(self) -> int:
        return self.container_region_kernel // 2

    @property
    def bay_region_pad(self) -> int:
        return self.bay_region_kernel // 2


# ── Head config ──────────────────────────────────────────────────────────

@dataclass
class HeadConfig:
    """Configuration for decision heads."""
    hidden: int = 64
    dueling: bool = True
    proximity_bays: int = 3


# ── Training config ──────────────────────────────────────────────────────

@dataclass
class DQNConfig:
    """Training configuration."""
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 32
    replay_size: int = 3_000
    target_tau: float = 0.005
    grad_clip: float = 1.0

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000

    tutorial_epsilon_start: float = 0.9
    tutorial_epsilon_end: float = 0.05
    tutorial_epsilon_epochs: int = 80

    double_dqn: bool = True
    n_step: int = 3

    # Dest head auxiliary loss weight (reward prediction, no bootstrap)
    dest_aux_weight: float = 0.5

    # Prioritized replay
    use_per: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 100_000

    # ── Variant-specific hyperparameters ─────────────────────────
    # Munchausen DQN
    munchausen_alpha: float = 0.9
    munchausen_tau: float = 0.03
    munchausen_clip: float = -1.0

    # QR-DQN
    n_quantiles: int = 32

    # IQN
    iqn_n_quantiles_train: int = 32
    iqn_n_quantiles_eval: int = 32
    iqn_embedding_dim: int = 64

    # NoisyNet
    noisy_sigma0: float = 0.5
    noisy_sigma_min: float = 0.1     # Floor for noise-scale decay (1.0 → this)
    noisy_decay_epochs: int = 60     # Linear decay schedule length (epochs)

    # Spectral / Layer norm
    use_spectral_norm: bool = False
    use_layer_norm: bool = False


# ── Top-level agent config ───────────────────────────────────────────────

@dataclass
class MultiHeadDQNConfig:
    """Complete agent configuration (unified spatial version)."""
    yard: YardDims = None
    unified: Dims = field(default_factory=Dims)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    heads: HeadConfig = field(default_factory=HeadConfig)
    training: DQNConfig = field(default_factory=DQNConfig)
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        if self.yard is None:
            raise ValueError("YardDims must be provided")
        n_splits = self.unified.n_splits
        if n_splits % self.cnn.s_stride != 0:
            raise ValueError(
                f"n_splits ({n_splits}) must be divisible by "
                f"s_stride ({self.cnn.s_stride})"
            )

    @property
    def s_down(self) -> int:
        """Downsampled split count after CNN stride."""
        return self.unified.n_splits // self.cnn.s_stride