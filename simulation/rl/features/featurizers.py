# simulation/rl/features/featurizers.py
"""Feature extraction for hierarchical DQN."""
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Set
from enum import Enum
import numpy as np
import torch


class SourceType(str, Enum):
    """Source types for moveable containers."""
    YARD = "YARD"
    TRAIN = "TRAIN"
    TRUCK = "TRUCK"


class DestinationType(str, Enum):
    """Destination types for container moves."""
    YARD = "YARD"
    TRAIN = "TRAIN"
    TRUCK = "TRUCK"


@dataclass
class MoveableContainer:
    """Container available for selection in Stage 1."""
    container_id: str
    source_type: SourceType
    source_id: Optional[str]  # train_id or truck_id if not yard
    
    # Position (for yard containers)
    row: int = 0
    bay: int = 0
    tier: int = 0
    start_split: int = 0
    
    # Container properties
    length_m: float = 12.2
    goods_type: str = "Regular"  # Regular, Reefer, DangerousGoods
    is_swap_body: bool = False
    is_trailer: bool = False
    days_until_departure: float = 30.0
    
    # Context
    source_anchor_bay: int = 0  # Train anchor bay or truck parking bay
    opens_access_below: bool = False  # Removing this opens access to container below


@dataclass
class Destination:
    """Destination available for selection in Stage 2."""
    dest_type: DestinationType
    dest_id: Optional[str] = None  # train_id or truck_id
    
    # For yard destinations
    row: int = 0
    bay: int = 0
    tier: int = 0
    start_split: int = 0
    
    # Context
    zone_match: bool = True  # Does goods type match zone
    heat_proximity: float = 0.5  # Distance to nearest train anchor (normalized)


@dataclass
class ParkingAction:
    """Parking move (not two-stage)."""
    truck_id: str
    spot: str
    preferred_bay: Optional[int]
    delta_bay: int


class ContainerFeaturizer:
    """Featurize containers for Stage 1 scoring."""
    
    FEAT_DIM = 16
    
    def __init__(self, n_rows: int, n_bays: int, n_tiers: int, split_factor: int):
        self.n_rows = max(1, n_rows)
        self.n_bays = max(1, n_bays)
        self.n_tiers = max(1, n_tiers)
        self.split_factor = max(1, split_factor)
    
    @staticmethod
    def feat_dim() -> int:
        return ContainerFeaturizer.FEAT_DIM
    
    def featurize_single(self, cont: MoveableContainer) -> np.ndarray:
        """Featurize a single container to 16-dim vector."""
        f = np.zeros(self.FEAT_DIM, dtype=np.float32)
        
        # Position (4 dims) - normalized
        f[0] = cont.row / max(1, self.n_rows - 1) if self.n_rows > 1 else 0.5
        f[1] = cont.bay / max(1, self.n_bays - 1) if self.n_bays > 1 else 0.5
        f[2] = cont.tier / max(1, self.n_tiers - 1) if self.n_tiers > 1 else 0.5
        f[3] = cont.start_split / max(1, self.split_factor - 1) if self.split_factor > 1 else 0.5
        
        # Length (1 dim) - normalized to typical range [6, 16] meters
        f[4] = (cont.length_m - 6.0) / 10.0
        
        # Goods type one-hot (3 dims)
        if cont.goods_type == "Reefer":
            f[5] = 1.0
        elif cont.goods_type == "DangerousGoods":
            f[6] = 1.0
        else:  # Regular
            f[7] = 1.0
        
        # Special flags (2 dims)
        f[8] = 1.0 if cont.is_swap_body else 0.0
        f[9] = 1.0 if cont.is_trailer else 0.0
        
        # Urgency (1 dim) - clamped to [0, 1]
        f[10] = min(1.0, max(0.0, cont.days_until_departure / 30.0))
        
        # Source type one-hot (3 dims)
        if cont.source_type == SourceType.YARD:
            f[11] = 1.0
        elif cont.source_type == SourceType.TRAIN:
            f[12] = 1.0
        else:  # TRUCK
            f[13] = 1.0
        
        # Source anchor bay (1 dim)
        f[14] = cont.source_anchor_bay / max(1, self.n_bays - 1) if self.n_bays > 1 else 0.5
        
        # Opens access below (1 dim)
        f[15] = 1.0 if cont.opens_access_below else 0.0
        
        return f
    
    def featurize_batch(self, containers: List[MoveableContainer]) -> torch.Tensor:
        """Featurize multiple containers to [K, 16] tensor."""
        if not containers:
            return torch.zeros((0, self.FEAT_DIM), dtype=torch.float32)
        
        feats = np.stack([self.featurize_single(c) for c in containers], axis=0)
        return torch.from_numpy(feats)


class DestinationFeaturizer:
    """Featurize destinations for Stage 2 scoring."""
    
    FEAT_DIM = 12
    
    def __init__(self, n_rows: int, n_bays: int, n_tiers: int, split_factor: int):
        self.n_rows = max(1, n_rows)
        self.n_bays = max(1, n_bays)
        self.n_tiers = max(1, n_tiers)
        self.split_factor = max(1, split_factor)
    
    @staticmethod
    def feat_dim() -> int:
        return DestinationFeaturizer.FEAT_DIM
    
    def featurize_single(
        self,
        dest: Destination,
        source_bay: int,
        source_tier: int
    ) -> np.ndarray:
        """Featurize a single destination to 12-dim vector."""
        f = np.zeros(self.FEAT_DIM, dtype=np.float32)
        
        # Target position (4 dims) - normalized
        f[0] = dest.row / max(1, self.n_rows - 1) if self.n_rows > 1 else 0.5
        f[1] = dest.bay / max(1, self.n_bays - 1) if self.n_bays > 1 else 0.5
        f[2] = dest.tier / max(1, self.n_tiers - 1) if self.n_tiers > 1 else 0.5
        f[3] = dest.start_split / max(1, self.split_factor - 1) if self.split_factor > 1 else 0.5
        
        # Destination type one-hot (3 dims)
        if dest.dest_type == DestinationType.YARD:
            f[4] = 1.0
        elif dest.dest_type == DestinationType.TRAIN:
            f[5] = 1.0
        else:  # TRUCK
            f[6] = 1.0
        
        # Bay distance (1 dim) - normalized
        f[7] = abs(dest.bay - source_bay) / max(1, self.n_bays - 1) if self.n_bays > 1 else 0.0
        
        # Tier delta (1 dim) - signed, normalized
        tier_delta = dest.tier - source_tier
        f[8] = tier_delta / max(1, self.n_tiers - 1) if self.n_tiers > 1 else 0.0
        
        # Zone match (1 dim)
        f[9] = 1.0 if dest.zone_match else 0.0
        
        # Heat proximity (1 dim) - already normalized in Destination
        f[10] = dest.heat_proximity
        
        # Ground level (1 dim) - placing at tier 0
        f[11] = 1.0 if dest.tier == 0 else 0.0
        
        return f
    
    def featurize_batch(
        self,
        destinations: List[Destination],
        source_bay: int,
        source_tier: int
    ) -> torch.Tensor:
        """Featurize multiple destinations to [K, 12] tensor."""
        if not destinations:
            return torch.zeros((0, self.FEAT_DIM), dtype=torch.float32)
        
        feats = np.stack([
            self.featurize_single(d, source_bay, source_tier) 
            for d in destinations
        ], axis=0)
        return torch.from_numpy(feats)


class ParkingFeaturizer:
    """Featurize parking actions for direct scoring."""
    
    FEAT_DIM = 4
    
    def __init__(self, n_bays: int):
        self.n_bays = max(1, n_bays)
    
    @staticmethod
    def feat_dim() -> int:
        return ParkingFeaturizer.FEAT_DIM
    
    def featurize_single(self, parking: ParkingAction) -> np.ndarray:
        """Featurize a parking action to 4-dim vector."""
        f = np.zeros(self.FEAT_DIM, dtype=np.float32)
        
        # Preferred bay (normalized)
        if parking.preferred_bay is not None:
            f[0] = parking.preferred_bay / max(1, self.n_bays - 1)
        else:
            f[0] = 0.5
        
        # Delta bay (normalized to [-1, 1] range then shifted to [0, 1])
        f[1] = (parking.delta_bay + 2) / 4.0  # Maps -2..+2 to 0..1
        
        # Is optimal (delta == 0)
        f[2] = 1.0 if parking.delta_bay == 0 else 0.0
        
        # Has preferred bay
        f[3] = 1.0 if parking.preferred_bay is not None else 0.0
        
        return f
    
    def featurize_batch(self, parkings: List[ParkingAction]) -> torch.Tensor:
        """Featurize multiple parking actions."""
        if not parkings:
            return torch.zeros((0, self.FEAT_DIM), dtype=torch.float32)
        
        feats = np.stack([self.featurize_single(p) for p in parkings], axis=0)
        return torch.from_numpy(feats)
