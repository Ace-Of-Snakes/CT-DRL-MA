# multihead_dqn/networks.py
"""Neural network components for Multi-Head DQN."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

from simulation.rl.multihead_dqn.config import BackboneConfig, HeadConfig, YardDims, ActionType, DestinationType


class ConvBlock(nn.Module):
    """Single 3D conv block with optional batchnorm."""
    
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        use_bn: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        padding = kernel_size // 2
        layers = [nn.Conv3d(in_ch, out_ch, kernel_size, padding=padding)]
        if use_bn:
            layers.append(nn.BatchNorm3d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout3d(dropout))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNBackbone(nn.Module):
    """
    3D CNN backbone for spatial feature extraction.
    Input: (B, C, R, S, T)
    Output: feature_map (B, F, R, S, T), global_feat (B, F)
    """
    
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        channels = [cfg.in_channels] + cfg.hidden_channels
        
        blocks = []
        for i in range(len(cfg.hidden_channels)):
            blocks.append(ConvBlock(
                channels[i],
                channels[i + 1],
                kernel_size=cfg.kernel_sizes[i],
                use_bn=cfg.use_batchnorm,
                dropout=cfg.dropout
            ))
        self.conv_blocks = nn.Sequential(*blocks)
        self.out_channels = cfg.hidden_channels[-1]
        
        # Global pooling for context vector
        self.global_pool = nn.AdaptiveAvgPool3d(1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (feature_map, global_features)."""
        feat_map = self.conv_blocks(x)  # (B, F, R, S, T)
        global_feat = self.global_pool(feat_map).flatten(1)  # (B, F)
        return feat_map, global_feat


class ActionTypeHead(nn.Module):
    """Decides: MOVE_CONTAINER vs SLOT_PARKING."""
    
    def __init__(self, in_features: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, len(ActionType))
        )
    
    def forward(self, global_feat: torch.Tensor) -> torch.Tensor:
        """Returns Q-values for action types: (B, 2)."""
        return self.net(global_feat)


class ContainerSelectionHead(nn.Module):
    """
    Selects which container to move via spatial attention.
    Uses feature map directly - occupied positions are valid selections.
    """
    
    def __init__(self, in_channels: int):
        super().__init__()
        # 1x1 conv to produce per-position Q-values
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 2, 1, kernel_size=1)
        )
    
    def forward(
        self,
        feat_map: torch.Tensor,
        occupancy_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            feat_map: (B, F, R, S, T)
            occupancy_mask: (B, R, S, T) bool - True where containers exist
        Returns:
            Q-values: (B, R*S*T) with -inf for unoccupied positions
        """
        q_spatial = self.conv(feat_map).squeeze(1)  # (B, R, S, T)
        
        # Mask unoccupied positions
        q_flat = q_spatial.reshape(q_spatial.size(0), -1)
        mask_flat = occupancy_mask.reshape(occupancy_mask.size(0), -1)
        
        # Apply mask: -inf for invalid positions
        q_flat = q_flat.masked_fill(~mask_flat, float('-inf'))
        return q_flat


class DestTypeHead(nn.Module):
    """Decides destination type: YARD, TRAIN, or TRUCK."""
    
    def __init__(self, global_dim: int, container_feat_dim: int, hidden: int = 64):
        super().__init__()
        # Conditioned on global context + selected container features
        self.net = nn.Sequential(
            nn.Linear(global_dim + container_feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, len(DestinationType))
        )
    
    def forward(
        self,
        global_feat: torch.Tensor,
        container_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            global_feat: (B, G)
            container_feat: (B, F) - features of selected container
        Returns:
            Q-values: (B, 3) for YARD/TRAIN/TRUCK
        """
        combined = torch.cat([global_feat, container_feat], dim=-1)
        return self.net(combined)


class SpatialPlacementHead(nn.Module):
    """
    Outputs Q-values for yard placement positions.
    Conditioned on source container position via relative encoding.
    """
    
    def __init__(self, in_channels: int, global_dim: int):
        super().__init__()
        # Inject global context into spatial features
        self.global_proj = nn.Linear(global_dim, in_channels)
        
        # Spatial processing with relative position awareness
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 2, 1, kernel_size=1)
        )
    
    def forward(
        self,
        feat_map: torch.Tensor,
        global_feat: torch.Tensor,
        source_pos: torch.Tensor,
        validity_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            feat_map: (B, F, R, S, T)
            global_feat: (B, G)
            source_pos: (B, 3) - (row, split, tier) of source container
            validity_mask: (B, R, S, T) bool - True where placement is valid
        Returns:
            Q-values: (B, R*S*T) with -inf for invalid positions
        """
        B, F, R, S, T = feat_map.shape
        device = feat_map.device
        
        # Create relative position encoding
        rel_pos = self._build_relative_encoding(source_pos, R, S, T, device)  # (B, F, R, S, T)
        
        # Inject global context
        g = self.global_proj(global_feat)  # (B, F)
        g = g.view(B, F, 1, 1, 1).expand(-1, -1, R, S, T)
        
        # Combine: features + relative encoding + global
        combined = torch.cat([feat_map + g, rel_pos], dim=1)  # (B, 2F, R, S, T)
        
        q_spatial = self.conv(combined).squeeze(1)  # (B, R, S, T)
        q_flat = q_spatial.reshape(B, -1)  # (B, R*S*T)
        
        # Mask invalid placements
        mask_flat = validity_mask.reshape(B, -1)
        q_flat = q_flat.masked_fill(~mask_flat, float('-inf'))
        
        return q_flat
    
    def _build_relative_encoding(
        self,
        source_pos: torch.Tensor,
        R: int, S: int, T: int,
        device: torch.device
    ) -> torch.Tensor:
        """Build relative position features from source container."""
        B = source_pos.size(0)
        F = 64  # encoding dimension (will be projected)
        
        # Create coordinate grids
        r_coords = torch.arange(R, device=device).float() / max(R - 1, 1)
        s_coords = torch.arange(S, device=device).float() / max(S - 1, 1)
        t_coords = torch.arange(T, device=device).float() / max(T - 1, 1)
        
        # Meshgrid: (R, S, T) each
        rr, ss, tt = torch.meshgrid(r_coords, s_coords, t_coords, indexing='ij')
        
        # Normalize source positions
        src_r = source_pos[:, 0:1].float() / max(R - 1, 1)  # (B, 1)
        src_s = source_pos[:, 1:2].float() / max(S - 1, 1)
        src_t = source_pos[:, 2:3].float() / max(T - 1, 1)
        
        # Compute relative distances: (B, R, S, T)
        rel_r = rr.unsqueeze(0) - src_r.view(B, 1, 1, 1)
        rel_s = ss.unsqueeze(0) - src_s.view(B, 1, 1, 1)
        rel_t = tt.unsqueeze(0) - src_t.view(B, 1, 1, 1)
        
        # Stack as channels: (B, 3, R, S, T)
        rel_encoding = torch.stack([rel_r, rel_s, rel_t], dim=1)
        
        # Expand to match feature dimension via simple repetition
        # (B, 3, R, S, T) -> (B, F, R, S, T) by repeating and truncating
        n_repeats = (F + 2) // 3
        rel_encoding = rel_encoding.repeat(1, n_repeats, 1, 1, 1)[:, :F]
        
        return rel_encoding


class VehicleSelectionHead(nn.Module):
    """Selects which train/truck to load container onto."""
    
    def __init__(self, global_dim: int, container_feat_dim: int, vehicle_feat_dim: int, hidden: int = 64):
        super().__init__()
        self.vehicle_encoder = nn.Sequential(
            nn.Linear(vehicle_feat_dim, hidden),
            nn.ReLU(inplace=True)
        )
        self.scorer = nn.Sequential(
            nn.Linear(global_dim + container_feat_dim + hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )
    
    def forward(
        self,
        global_feat: torch.Tensor,
        container_feat: torch.Tensor,
        vehicle_feats: torch.Tensor,
        vehicle_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            global_feat: (B, G)
            container_feat: (B, Fc)
            vehicle_feats: (B, V, Fv) - features for each vehicle
            vehicle_mask: (B, V) bool - True for valid vehicles
        Returns:
            Q-values: (B, V) with -inf for invalid vehicles
        """
        B, V, Fv = vehicle_feats.shape
        
        # Encode vehicles
        v_enc = self.vehicle_encoder(vehicle_feats)  # (B, V, H)
        
        # Expand global and container features
        g_exp = global_feat.unsqueeze(1).expand(-1, V, -1)  # (B, V, G)
        c_exp = container_feat.unsqueeze(1).expand(-1, V, -1)  # (B, V, Fc)
        
        # Combine and score
        combined = torch.cat([g_exp, c_exp, v_enc], dim=-1)  # (B, V, G+Fc+H)
        q_values = self.scorer(combined).squeeze(-1)  # (B, V)
        
        # Mask invalid vehicles
        q_values = q_values.masked_fill(~vehicle_mask, float('-inf'))
        
        return q_values


class ParkingHead(nn.Module):
    """Selects which truck to assign parking slot."""
    
    def __init__(self, global_dim: int, parking_feat_dim: int, hidden: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(parking_feat_dim, hidden),
            nn.ReLU(inplace=True)
        )
        self.scorer = nn.Sequential(
            nn.Linear(global_dim + hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )
    
    def forward(
        self,
        global_feat: torch.Tensor,
        parking_feats: torch.Tensor,
        parking_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            global_feat: (B, G)
            parking_feats: (B, P, Fp) - features for pending parking requests
            parking_mask: (B, P) bool - True for valid parking actions
        Returns:
            Q-values: (B, P) with -inf for invalid
        """
        B, P, Fp = parking_feats.shape
        
        p_enc = self.encoder(parking_feats)  # (B, P, H)
        g_exp = global_feat.unsqueeze(1).expand(-1, P, -1)  # (B, P, G)
        
        combined = torch.cat([g_exp, p_enc], dim=-1)
        q_values = self.scorer(combined).squeeze(-1)  # (B, P)
        
        q_values = q_values.masked_fill(~parking_mask, float('-inf'))
        return q_values


class MultiHeadQNetwork(nn.Module):
    """
    Complete Multi-Head Q-Network.
    
    Decision flow:
    1. ActionTypeHead: MOVE_CONTAINER or SLOT_PARKING
    2. If MOVE: ContainerSelectionHead -> DestTypeHead
    3. If YARD dest: SpatialPlacementHead
    4. If TRAIN/TRUCK dest: VehicleSelectionHead
    5. If PARKING: ParkingHead
    """
    
    def __init__(
        self,
        yard: YardDims,
        backbone_cfg: BackboneConfig,
        head_cfg: HeadConfig
    ):
        super().__init__()
        self.yard = yard
        
        # Backbone
        self.backbone = CNNBackbone(backbone_cfg)
        feat_dim = self.backbone.out_channels
        global_dim = head_cfg.global_hidden
        
        # Global projection
        self.global_fc = nn.Linear(feat_dim, global_dim)
        
        # Decision heads
        self.action_type_head = ActionTypeHead(global_dim)
        self.container_head = ContainerSelectionHead(feat_dim)
        self.dest_type_head = DestTypeHead(global_dim, feat_dim)
        self.spatial_head = SpatialPlacementHead(feat_dim, global_dim)
        self.vehicle_head = VehicleSelectionHead(
            global_dim, feat_dim, head_cfg.vehicle_feat_dim
        )
        self.parking_head = ParkingHead(global_dim, head_cfg.vehicle_feat_dim)
    
    def encode_state(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode state through backbone.
        Args:
            state: (B, C, R, S, T)
        Returns:
            feat_map: (B, F, R, S, T)
            global_feat: (B, G)
        """
        feat_map, global_raw = self.backbone(state)
        global_feat = F.relu(self.global_fc(global_raw))
        return feat_map, global_feat
    
    def extract_container_features(
        self,
        feat_map: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract features for selected container positions.
        Args:
            feat_map: (B, F, R, S, T)
            positions: (B, 3) - (row, split, tier) indices
        Returns:
            features: (B, F)
        """
        B, F, R, S, T = feat_map.shape
        
        # Gather features at specified positions
        r_idx = positions[:, 0].long()
        s_idx = positions[:, 1].long()
        t_idx = positions[:, 2].long()
        
        # Index: feat_map[b, :, r, s, t]
        features = feat_map[
            torch.arange(B, device=feat_map.device),
            :,
            r_idx,
            s_idx,
            t_idx
        ]  # (B, F)
        
        return features
    
    def q_action_type(self, global_feat: torch.Tensor) -> torch.Tensor:
        """Q-values for MOVE_CONTAINER vs SLOT_PARKING."""
        return self.action_type_head(global_feat)
    
    def q_container_selection(
        self,
        feat_map: torch.Tensor,
        occupancy_mask: torch.Tensor
    ) -> torch.Tensor:
        """Q-values for container selection (flattened spatial)."""
        return self.container_head(feat_map, occupancy_mask)
    
    def q_dest_type(
        self,
        global_feat: torch.Tensor,
        container_feat: torch.Tensor
    ) -> torch.Tensor:
        """Q-values for destination type."""
        return self.dest_type_head(global_feat, container_feat)
    
    def q_placement(
        self,
        feat_map: torch.Tensor,
        global_feat: torch.Tensor,
        source_pos: torch.Tensor,
        validity_mask: torch.Tensor
    ) -> torch.Tensor:
        """Q-values for yard placement (flattened spatial)."""
        return self.spatial_head(feat_map, global_feat, source_pos, validity_mask)
    
    def q_vehicle(
        self,
        global_feat: torch.Tensor,
        container_feat: torch.Tensor,
        vehicle_feats: torch.Tensor,
        vehicle_mask: torch.Tensor
    ) -> torch.Tensor:
        """Q-values for train/truck selection."""
        return self.vehicle_head(global_feat, container_feat, vehicle_feats, vehicle_mask)
    
    def q_parking(
        self,
        global_feat: torch.Tensor,
        parking_feats: torch.Tensor,
        parking_mask: torch.Tensor
    ) -> torch.Tensor:
        """Q-values for parking action selection."""
        return self.parking_head(global_feat, parking_feats, parking_mask)