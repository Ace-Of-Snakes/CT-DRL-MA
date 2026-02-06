# multihead_dqn/networks.py
"""Factored-CNN Multi-Head Q-Network.

Architecture rationale:
  Containers are 1D objects (1 row Ã— 20-53 splits Ã— 1 tier) embedded in
  a 3D grid (5 Ã— 1160 Ã— 5).  Standard (3,3,3) kernels waste parameters
  mixing noise across rows/tiers and see only ~7 of a container's 20-40
  splits.  Instead we factorise the convolution:

  Stage 1: (1, 21, 1) kernels along S axis â€” extract container profiles.
           Two layers give RF = 41 splits (one 40ft container).
           Second layer strides Ã—4 â†’ (5, 290, 5) = 7 250 positions.

  Stage 2: (3, 1, 3) kernel across RÃ—T â€” stacking & cross-row context.
           Each (row, tier) can see its neighbours (blocking, support).

  Stage 3: (1, 5, 1) kernel along S â€” neighbourhood refinement.
           Total RF_S â‰ˆ 57 splits (~3 bays).

  Global:  Occupied-only max+mean pooling (no spatial dilution).
           At 0.5% occupancy, naive avg-pool dilutes 200Ã—; ours doesn't.

  Container selection: 1Ã—1 conv â†’ per-position Q-values, masked by
           CONTAINER_START at downsampled resolution.

  Per-container embedding: index feat_map at selected position.
           64-dim feature already encodes all container+context info.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from simulation.rl.multihead_dqn.config import (
    CNNConfig, HeadConfig, YardDims, ActionType, DestinationType,
)

# State tensor channel indices (must match state_encoder.ChannelSpec)
_CH_OCCUPANCY = 0
_CH_CONTAINER_START = 1


# ================================================================
# Encoded state container
# ================================================================

@dataclass
class EncodedState:
    """Outputs from CNN backbone encoding."""
    global_feat: torch.Tensor    # (B, global_dim)
    feat_map: torch.Tensor       # (B, feat_channels, R, S_down, T)


# ================================================================
# CNN Backbone
# ================================================================

class FactoredCNNBackbone(nn.Module):
    """Factored 1D CNN for container terminal yard states.

    Processes (B, C, R, S, T) â†’ (B, feat_ch, R, S_down, T).
    Each dimension gets the kernel shape that matches its patterns:
      S (splits): long 1D kernels matching container lengths
      RÃ—T (row/tier): short 2D kernels for stacking context
    """

    def __init__(self, cfg: CNNConfig):
        super().__init__()
        C_in = cfg.n_state_channels
        C1 = cfg.stage1_channels
        C2 = cfg.feat_channels
        k = cfg.container_kernel
        pk = cfg.container_pad
        rk = cfg.cross_kernel
        rp = cfg.cross_pad
        fk = cfg.refine_kernel
        fp = cfg.refine_pad
        G = cfg.gn_groups

        # Stage 1a: (1,21,1) no stride â€” RF_S = 21 (one 20ft container)
        self.conv1a = nn.Conv3d(C_in, C1, (1, k, 1), padding=(0, pk, 0))
        self.gn1a = nn.GroupNorm(G, C1)

        # Stage 1b: (1,21,1) stride 4 â€” RF_S = 41 (one 40ft container)
        self.conv1b = nn.Conv3d(
            C1, C2, (1, k, 1),
            stride=(1, cfg.s_stride, 1),
            padding=(0, pk, 0),
        )
        self.gn1b = nn.GroupNorm(G, C2)

        # Stage 2: (3,1,3) â€” cross-row/tier context (stacking, blocking)
        self.conv2 = nn.Conv3d(C2, C2, (rk, 1, rk), padding=(rp, 0, rp))
        self.gn2 = nn.GroupNorm(G, C2)

        # Stage 3: (1,5,1) â€” neighbourhood refinement, RF_S â‰ˆ 57
        self.conv3 = nn.Conv3d(C2, C2, (1, fk, 1), padding=(0, fp, 0))
        self.gn3 = nn.GroupNorm(G, C2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, R, S, T) â†’ (B, feat_channels, R, S_down, T)."""
        x = F.relu(self.gn1a(self.conv1a(x)), inplace=True)
        x = F.relu(self.gn1b(self.conv1b(x)), inplace=True)
        x = F.relu(self.gn2(self.conv2(x)), inplace=True)
        x = F.relu(self.gn3(self.conv3(x)), inplace=True)
        return x


class OccupiedPooling(nn.Module):
    """Max + mean pooling over occupied positions only.

    At typical occupancy (0.5%), naive AdaptiveAvgPool dilutes
    container features 200Ã—. This module pools ONLY over positions
    where containers exist, preserving signal strength.
    """

    def __init__(self, feat_channels: int, global_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * feat_channels, global_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat_map: torch.Tensor,
                occ_mask: torch.Tensor) -> torch.Tensor:
        """Pool feat_map using occupancy mask.

        Args:
            feat_map: (B, C, R, S_down, T)
            occ_mask: (B, R, S_down, T) bool
        Returns:
            (B, global_dim)
        """
        mask = occ_mask.unsqueeze(1)                  # (B, 1, R, S_down, T)
        mask_f = mask.float()

        # Mean over occupied positions
        n_occ = mask_f.sum(dim=(2, 3, 4)).clamp(min=1.0)     # (B, 1)
        g_mean = (feat_map * mask_f).sum(dim=(2, 3, 4)) / n_occ  # (B, C)

        # Max over occupied positions
        g_max = torch.where(mask, feat_map, torch.full_like(feat_map, -1e9))
        g_max = g_max.amax(dim=(2, 3, 4))                     # (B, C)
        g_max = g_max * (n_occ > 0).float()                    # zero if empty

        return self.fc(torch.cat([g_mean, g_max], dim=1))


# ================================================================
# Decision Heads
# ================================================================

class DuelingHead(nn.Module):
    """Dueling V+A for fixed-size action spaces."""

    def __init__(self, in_features: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.value(x)
        a = self.advantage(x)
        return v + a - a.mean(dim=-1, keepdim=True)


class ActionTypeHead(nn.Module):
    """MOVE_CONTAINER / SLOT_PARKING / IMPORT_VEHICLE."""

    def __init__(self, global_dim: int, hidden: int = 64, dueling: bool = True):
        super().__init__()
        n = len(ActionType)
        self.net = (
            DuelingHead(global_dim, n, hidden) if dueling
            else nn.Sequential(
                nn.Linear(global_dim, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, n),
            )
        )

    def forward(self, global_feat: torch.Tensor) -> torch.Tensor:
        return self.net(global_feat)


class ContainerSelectionHead(nn.Module):
    """Per-position Q-values via 1Ã—1 conv on the CNN feat_map.

    Outputs one Q-value per (row, s_down, tier) position. Masked
    by CONTAINER_START so only actual containers are selectable.
    """

    def __init__(self, feat_channels: int):
        super().__init__()
        mid = feat_channels // 2
        self.conv = nn.Sequential(
            nn.Conv3d(feat_channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, 1, 1),
        )

    def forward(self, feat_map: torch.Tensor,
                container_mask: torch.Tensor) -> torch.Tensor:
        """Score each position for container selection.

        Args:
            feat_map: (B, C, R, S_down, T)
            container_mask: (B, R, S_down, T) bool â€” True at CONTAINER_START
        Returns:
            q_flat: (B, R*S_down*T) with -inf for non-container positions
        """
        q = self.conv(feat_map).squeeze(1)               # (B, R, S_down, T)
        q_flat = q.reshape(q.size(0), -1)
        mask_flat = container_mask.reshape(container_mask.size(0), -1)
        return q_flat.masked_fill(~mask_flat, float("-inf"))


class DestTypeHead(nn.Module):
    """YARD / TRAIN / TRUCK destination type."""

    def __init__(self, global_dim: int, feat_dim: int,
                 hidden: int = 64, dueling: bool = True):
        super().__init__()
        n = len(DestinationType)
        combined = global_dim + feat_dim
        self.net = (
            DuelingHead(combined, n, hidden) if dueling
            else nn.Sequential(
                nn.Linear(combined, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, n),
            )
        )

    def forward(self, global_feat: torch.Tensor,
                container_feat: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([global_feat, container_feat], dim=-1))


class ProximityPlacementHead(nn.Module):
    """Split-level placement within a proximity window.

    Instead of scoring all 29,000 positions or 1,450 bay-level positions,
    scores only the Â±proximity_bays window around a reference bay at full
    split resolution.  Matches OperationsDefaults.PROXIMITY_SEARCH_BAYS.

    Factorised: Q(r, s, t) = Q_split(s) + Q_row_tier(r, t).
    Output: R Ã— window_splits Ã— T positions.
    """

    _HIDDEN_SPLIT: int = 128
    _HIDDEN_RT: int = 64

    def __init__(self, input_dim: int, proximity_bays: int,
                 split_factor: int, n_rows: int, n_tiers: int):
        super().__init__()
        self.proximity_bays = proximity_bays
        self.split_factor = split_factor
        self.n_rows = n_rows
        self.n_tiers = n_tiers
        self.window_bays = 2 * proximity_bays + 1
        self.window_splits = self.window_bays * split_factor

        self.split_net = nn.Sequential(
            nn.Linear(input_dim, self._HIDDEN_SPLIT),
            nn.ReLU(inplace=True),
            nn.Linear(self._HIDDEN_SPLIT, self.window_splits),
        )
        self.row_tier_net = nn.Sequential(
            nn.Linear(input_dim, self._HIDDEN_RT),
            nn.ReLU(inplace=True),
            nn.Linear(self._HIDDEN_RT, n_rows * n_tiers),
        )

    def forward(self, global_feat: torch.Tensor,
                container_feat: torch.Tensor,
                validity_window: torch.Tensor) -> torch.Tensor:
        """Score positions within the proximity window.

        Args:
            global_feat: (B, G)
            container_feat: (B, F)
            validity_window: (B, R, window_splits, T) bool
        Returns:
            q_flat: (B, R*window_splits*T) with -inf for invalid
        """
        x = torch.cat([global_feat, container_feat], dim=-1)
        B = x.size(0)
        R, W, T = self.n_rows, self.window_splits, self.n_tiers

        split_q = self.split_net(x)                              # (B, W)
        rt_q = self.row_tier_net(x).view(B, R, T)               # (B, R, T)

        q_grid = split_q.view(B, 1, W, 1) + rt_q.view(B, R, 1, T)
        q_flat = q_grid.reshape(B, -1)
        mask_flat = validity_window.reshape(B, -1)
        return q_flat.masked_fill(~mask_flat, float("-inf"))


class VehicleSelectionHead(nn.Module):
    """Scores each vehicle for loading or import."""

    def __init__(self, global_dim: int, feat_dim: int,
                 vehicle_feat_dim: int, hidden: int = 64):
        super().__init__()
        self.vehicle_enc = nn.Sequential(
            nn.Linear(vehicle_feat_dim, hidden), nn.ReLU(inplace=True),
        )
        self.scorer = nn.Sequential(
            nn.Linear(global_dim + feat_dim + hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_feat, container_feat, vehicle_feats, vehicle_mask):
        B, V, _ = vehicle_feats.shape
        v_enc = self.vehicle_enc(vehicle_feats)                  # (B, V, H)
        g_exp = global_feat.unsqueeze(1).expand(-1, V, -1)
        c_exp = container_feat.unsqueeze(1).expand(-1, V, -1)
        combined = torch.cat([g_exp, c_exp, v_enc], dim=-1)
        q = self.scorer(combined).squeeze(-1)                    # (B, V)
        return q.masked_fill(~vehicle_mask, float("-inf"))


class ParkingHead(nn.Module):
    """Scores each parking slot for a truck."""

    def __init__(self, global_dim: int, parking_feat_dim: int, hidden: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(parking_feat_dim, hidden), nn.ReLU(inplace=True),
        )
        self.scorer = nn.Sequential(
            nn.Linear(global_dim + hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_feat, parking_feats, parking_mask):
        B, P, _ = parking_feats.shape
        p_enc = self.encoder(parking_feats)                      # (B, P, H)
        g_exp = global_feat.unsqueeze(1).expand(-1, P, -1)
        combined = torch.cat([g_exp, p_enc], dim=-1)
        q = self.scorer(combined).squeeze(-1)                    # (B, P)
        return q.masked_fill(~parking_mask, float("-inf"))


# ================================================================
# Complete Q-Network
# ================================================================

class FactoredCNNQNetwork(nn.Module):
    """Factored-CNN Q-Network for container terminal operations.

    Pipeline:
      State (C, R, S, T)
        â†’ FactoredCNNBackbone  â†’ feat_map (F, R, S_down, T)
        â†’ OccupiedPooling      â†’ global_feat (G)
        â†’ Decision heads       â†’ Q-values per action stage

    Container embedding = feat_map[:, r, s_down, t] at the selected
    position â€” the CNN features already encode all per-container info
    (type, urgency, blocking, demand) plus spatial context (neighbours,
    vehicle proximity, congestion).
    """

    def __init__(self, yard: YardDims, cnn_cfg: CNNConfig, head_cfg: HeadConfig):
        super().__init__()
        self.yard = yard
        self.cnn_cfg = cnn_cfg

        Fc = cnn_cfg.feat_channels
        G = cnn_cfg.global_dim

        # Backbone + pooling
        self.backbone = FactoredCNNBackbone(cnn_cfg)
        self.pool = OccupiedPooling(Fc, G)

        # Decision heads
        self.action_type_head = ActionTypeHead(G, head_cfg.hidden, head_cfg.dueling)
        self.container_head = ContainerSelectionHead(Fc)
        self.dest_type_head = DestTypeHead(G, Fc, head_cfg.hidden, head_cfg.dueling)
        # Placement uses n_rows (yard only), NOT n_state_rows (yard + parking)
        self.placement_head = ProximityPlacementHead(
            G + Fc, head_cfg.proximity_bays, yard.split_factor,
            yard.n_rows, yard.n_tiers,
        )
        self.vehicle_head = VehicleSelectionHead(G, Fc, head_cfg.vehicle_feat_dim)
        self.parking_head = ParkingHead(G, head_cfg.vehicle_feat_dim)

    # ----------------------------------------------------------------
    # Encoding
    # ----------------------------------------------------------------

    def _downsample_occ(self, state: torch.Tensor) -> torch.Tensor:
        """Downsample occupancy channel to match feat_map resolution.

        Uses max-pool so any occupied split in a stride-window â†’ True.

        Args:
            state: (B, C, R, S, T)
        Returns:
            (B, R, S_down, T) bool
        """
        occ = state[:, _CH_OCCUPANCY: _CH_OCCUPANCY + 1]  # (B, 1, R, S, T)
        stride = self.cnn_cfg.s_stride
        occ_down = F.max_pool3d(occ, (1, stride, 1), stride=(1, stride, 1))
        return occ_down.squeeze(1) > 0.5

    def encode_state(self, state: torch.Tensor) -> EncodedState:
        """Full state â†’ CNN features + global embedding.

        Args:
            state: (B, C, R, S, T)
        Returns:
            EncodedState with feat_map and global_feat
        """
        feat_map = self.backbone(state)
        occ_down = self._downsample_occ(state)
        global_feat = self.pool(feat_map, occ_down)
        return EncodedState(global_feat=global_feat, feat_map=feat_map)

    def extract_container_feat(self, feat_map: torch.Tensor,
                               pos_down: torch.Tensor) -> torch.Tensor:
        """Extract per-container feature from feat_map.

        Differentiable indexing â€” gradients flow back through backbone.

        Args:
            feat_map: (B, C, R, S_down, T)
            pos_down: (B, 3) long â€” (row, s_down, tier)
        Returns:
            (B, feat_channels)
        """
        B = feat_map.size(0)
        idx = torch.arange(B, device=feat_map.device)
        return feat_map[idx, :, pos_down[:, 0], pos_down[:, 1], pos_down[:, 2]]

    # ----------------------------------------------------------------
    # Q-value accessors (clean API for agent)
    # ----------------------------------------------------------------

    def q_action_type(self, global_feat):
        return self.action_type_head(global_feat)

    def q_container_selection(self, feat_map, container_mask):
        return self.container_head(feat_map, container_mask)

    def q_dest_type(self, global_feat, container_feat):
        return self.dest_type_head(global_feat, container_feat)

    def q_placement(self, global_feat, container_feat, validity_window):
        return self.placement_head(global_feat, container_feat, validity_window)

    def q_vehicle(self, global_feat, container_feat, vehicle_feats, vehicle_mask):
        return self.vehicle_head(global_feat, container_feat, vehicle_feats, vehicle_mask)

    def q_parking(self, global_feat, parking_feats, parking_mask):
        return self.parking_head(global_feat, parking_feats, parking_mask)