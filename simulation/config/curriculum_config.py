# simulation/config/curriculum_config.py
"""Curriculum learning configuration for hierarchical DQN training."""
from dataclasses import dataclass, field
from typing import List
import torch


@dataclass
class CurriculumConfig:
    """Configuration for curriculum-based training."""
    
    # Curriculum scaling
    start_imports: int = 20
    increment: int = 20
    max_imports: int = 220
    days_per_stage: int = 365
    
    # Network dimensions
    state_hidden: int = 128
    container_feat_dim: int = 16
    destination_feat_dim: int = 12
    scorer_hidden: int = 64
    
    # DQN hyperparameters
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 32
    replay_size: int = 100_000
    target_tau: float = 0.005
    
    # Epsilon schedule (per stage)
    epsilon_start: float = 0.3
    epsilon_end: float = 0.02
    epsilon_decay_steps: int = 50_000
    epsilon_reset_per_stage: bool = True
    
    # Penalties and rewards
    no_destination_penalty: float = -1.0
    invalid_selection_penalty: float = -0.5
    max_retries_per_timestep: int = 10
    
    # Hardware
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    # Logging
    log_interval_days: int = 10
    
    @property
    def num_stages(self) -> int:
        """Total number of curriculum stages."""
        return (self.max_imports - self.start_imports) // self.increment + 1
    
    def imports_for_stage(self, stage: int) -> int:
        """Get import count for a specific stage."""
        return min(self.start_imports + stage * self.increment, self.max_imports)
    
    def exports_for_stage(self, stage: int, export_ratio: float = 0.75) -> int:
        """Get export count for a specific stage."""
        return int(self.imports_for_stage(stage) * export_ratio)
    
    def stage_schedule(self, export_ratio: float = 0.75) -> List[dict]:
        """Get full curriculum schedule."""
        return [
            {
                "stage": i,
                "imports": self.imports_for_stage(i),
                "exports": self.exports_for_stage(i, export_ratio),
                "days": self.days_per_stage,
            }
            for i in range(self.num_stages)
        ]


@dataclass 
class HierarchicalDQNConfig:
    """Network architecture configuration."""
    
    # Backbone (3D CNN)
    in_channels: int = 21
    backbone_channels: List[int] = field(default_factory=lambda: [32, 64])
    state_hidden: int = 128
    
    # Container scorer
    container_feat_dim: int = 16
    container_hidden: int = 64
    
    # Destination scorer  
    destination_feat_dim: int = 12
    destination_hidden: int = 64
    
    # Combined
    scorer_hidden: int = 64
    
    # Training
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 32
    replay_size: int = 100_000
    target_tau: float = 0.005
    grad_clip: float = 1.0
    
    # Epsilon
    epsilon_start: float = 0.3
    epsilon_end: float = 0.02
    epsilon_decay_steps: int = 50_000
    
    # Device
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
