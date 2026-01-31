# simulation/rl/policy/hierarchical_networks.py
"""Neural network components for hierarchical DQN."""
import torch
import torch.nn as nn
from typing import Tuple

from simulation.config.curriculum_config import HierarchicalDQNConfig


class SharedBackbone(nn.Module):
    """
    3D CNN backbone for state encoding.
    Input: [B, C, R, BAYS, T] 
    Output: [B, state_hidden]
    """
    
    def __init__(self, in_channels: int = 21, hidden: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.proj = nn.Linear(64, hidden)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: State tensor [B, C, R, BAYS, T]
        Returns:
            State embedding [B, hidden]
        """
        h = self.conv(x).flatten(1)  # [B, 64]
        return torch.tanh(self.proj(h))  # [B, hidden]


class ContainerScorer(nn.Module):
    """
    Stage 1: Score containers for selection.
    Input: state_emb [B, H_s], container_feats [K, F_c]
    Output: Q-values [K]
    """
    
    def __init__(
        self,
        state_dim: int = 128,
        container_feat_dim: int = 16,
        hidden: int = 64
    ):
        super().__init__()
        self.container_encoder = nn.Sequential(
            nn.Linear(container_feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.q_head = nn.Sequential(
            nn.Linear(state_dim + hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
    
    def forward(
        self,
        state_emb: torch.Tensor,
        container_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            state_emb: [1, H_s] (single state)
            container_feats: [K, F_c] (K containers)
        Returns:
            Q-values [K]
        """
        K = container_feats.size(0)
        if K == 0:
            return torch.zeros(0, device=state_emb.device)
        
        # Encode containers
        cont_emb = self.container_encoder(container_feats)  # [K, H]
        
        # Expand state to match containers
        state_exp = state_emb.expand(K, -1)  # [K, H_s]
        
        # Concatenate and score
        combined = torch.cat([state_exp, cont_emb], dim=-1)  # [K, H_s + H]
        q_values = self.q_head(combined).squeeze(-1)  # [K]
        
        return q_values


class DestinationScorer(nn.Module):
    """
    Stage 2: Score destinations for selected container.
    Input: state_emb [1, H_s], container_feat [1, F_c], dest_feats [K, F_d]
    Output: Q-values [K]
    """
    
    def __init__(
        self,
        state_dim: int = 128,
        container_feat_dim: int = 16,
        destination_feat_dim: int = 12,
        hidden: int = 64
    ):
        super().__init__()
        # Encode destination + container context together
        self.dest_encoder = nn.Sequential(
            nn.Linear(destination_feat_dim + container_feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.q_head = nn.Sequential(
            nn.Linear(state_dim + hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
    
    def forward(
        self,
        state_emb: torch.Tensor,
        container_feat: torch.Tensor,
        dest_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            state_emb: [1, H_s]
            container_feat: [1, F_c] or [F_c]
            dest_feats: [K, F_d]
        Returns:
            Q-values [K]
        """
        K = dest_feats.size(0)
        if K == 0:
            return torch.zeros(0, device=state_emb.device)
        
        # Ensure container_feat is [1, F_c]
        if container_feat.dim() == 1:
            container_feat = container_feat.unsqueeze(0)
        
        # Expand container feat to match destinations
        cont_exp = container_feat.expand(K, -1)  # [K, F_c]
        
        # Concatenate destination + container features
        dest_context = torch.cat([dest_feats, cont_exp], dim=-1)  # [K, F_d + F_c]
        
        # Encode
        dest_emb = self.dest_encoder(dest_context)  # [K, H]
        
        # Expand state
        state_exp = state_emb.expand(K, -1)  # [K, H_s]
        
        # Concatenate and score
        combined = torch.cat([state_exp, dest_emb], dim=-1)  # [K, H_s + H]
        q_values = self.q_head(combined).squeeze(-1)  # [K]
        
        return q_values


class ParkingScorer(nn.Module):
    """
    Score parking actions (competes with containers in Stage 1).
    Input: state_emb [1, H_s], parking_feats [K, F_p]
    Output: Q-values [K]
    """
    
    def __init__(
        self,
        state_dim: int = 128,
        parking_feat_dim: int = 4,
        hidden: int = 32
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(parking_feat_dim, hidden),
            nn.ReLU(inplace=True),
        )
        self.q_head = nn.Sequential(
            nn.Linear(state_dim + hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
    
    def forward(
        self,
        state_emb: torch.Tensor,
        parking_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            state_emb: [1, H_s]
            parking_feats: [K, F_p]
        Returns:
            Q-values [K]
        """
        K = parking_feats.size(0)
        if K == 0:
            return torch.zeros(0, device=state_emb.device)
        
        park_emb = self.encoder(parking_feats)  # [K, H]
        state_exp = state_emb.expand(K, -1)  # [K, H_s]
        combined = torch.cat([state_exp, park_emb], dim=-1)
        q_values = self.q_head(combined).squeeze(-1)
        
        return q_values


class HierarchicalQNetwork(nn.Module):
    """
    Complete hierarchical Q-network combining all components.
    """
    
    def __init__(self, cfg: HierarchicalDQNConfig = None):
        super().__init__()
        cfg = cfg or HierarchicalDQNConfig()
        
        self.backbone = SharedBackbone(cfg.in_channels, cfg.state_hidden)
        self.container_scorer = ContainerScorer(
            cfg.state_hidden,
            cfg.container_feat_dim,
            cfg.scorer_hidden
        )
        self.destination_scorer = DestinationScorer(
            cfg.state_hidden,
            cfg.container_feat_dim,
            cfg.destination_feat_dim,
            cfg.scorer_hidden
        )
        self.parking_scorer = ParkingScorer(
            cfg.state_hidden,
            parking_feat_dim=4,
            hidden=32
        )
    
    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """Encode state tensor to embedding."""
        return self.backbone(state)
    
    def score_containers(
        self,
        state_emb: torch.Tensor,
        container_feats: torch.Tensor
    ) -> torch.Tensor:
        """Score containers for Stage 1."""
        return self.container_scorer(state_emb, container_feats)
    
    def score_destinations(
        self,
        state_emb: torch.Tensor,
        container_feat: torch.Tensor,
        dest_feats: torch.Tensor
    ) -> torch.Tensor:
        """Score destinations for Stage 2."""
        return self.destination_scorer(state_emb, container_feat, dest_feats)
    
    def score_parking(
        self,
        state_emb: torch.Tensor,
        parking_feats: torch.Tensor
    ) -> torch.Tensor:
        """Score parking actions for Stage 1."""
        return self.parking_scorer(state_emb, parking_feats)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test network dimensions
    cfg = HierarchicalDQNConfig()
    net = HierarchicalQNetwork(cfg)
    
    print(f"Total parameters: {count_parameters(net):,}")
    print(f"  Backbone: {count_parameters(net.backbone):,}")
    print(f"  Container scorer: {count_parameters(net.container_scorer):,}")
    print(f"  Destination scorer: {count_parameters(net.destination_scorer):,}")
    print(f"  Parking scorer: {count_parameters(net.parking_scorer):,}")
    
    # Test forward pass
    batch_size = 1
    R, B, T, C = 5, 58, 5, 21
    
    state = torch.randn(batch_size, C, R, B, T)
    state_emb = net.encode_state(state)
    print(f"\nState embedding shape: {state_emb.shape}")
    
    # Container scoring
    K_cont = 50
    cont_feats = torch.randn(K_cont, 16)
    q_cont = net.score_containers(state_emb, cont_feats)
    print(f"Container Q-values shape: {q_cont.shape}")
    
    # Destination scoring
    K_dest = 100
    cont_feat = cont_feats[0]
    dest_feats = torch.randn(K_dest, 12)
    q_dest = net.score_destinations(state_emb, cont_feat, dest_feats)
    print(f"Destination Q-values shape: {q_dest.shape}")
    
    # Parking scoring
    K_park = 5
    park_feats = torch.randn(K_park, 4)
    q_park = net.score_parking(state_emb, park_feats)
    print(f"Parking Q-values shape: {q_park.shape}")
