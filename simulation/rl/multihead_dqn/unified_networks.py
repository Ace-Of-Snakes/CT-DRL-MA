# multihead_dqn/unified_networks.py
"""Unified Spatial Q-Network — 2-head architecture.

Architecture:
  State (C, R_uni, S, T)
    → FactoredCNNBackbone  → feat_map (F, R_uni, S_down, T)
    → OccupiedPooling      → global_feat (G)
    → SourceHead           → Q-values per source position
    → extract source feat  → source_feat (F)
    → UnifiedDestHead      → Q-values per dest position

Move type = f(source_region, dest_region) — inferred, not predicted.
Vehicle identity = resolved from spatial coordinates — no separate head.

Compared to the old 6-head design:
  REMOVED: ActionTypeHead, DestTypeHead, VehicleSelectionHead, ParkingHead
  KEPT:    FactoredCNNBackbone (processes taller R=15 image)
           OccupiedPooling (pools across all regions)
           DuelingHead (utility for small fixed-size outputs)
  NEW:     UnifiedDestHead (source-conditioned destination scoring)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from simulation.rl.multihead_dqn.config import CNNConfig, HeadConfig, UnifiedDims


# Channel indices (must match unified_state_encoder.UnifiedChannelSpec)
_CH_OCCUPANCY = 0
_CH_CONTAINER_START = 1


# ══════════════════════════════════════════════════════════════════════════
# Encoded state container
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class EncodedState:
    """Outputs from CNN backbone encoding."""
    global_feat: torch.Tensor    # (B, global_dim)
    feat_map: torch.Tensor       # (B, feat_channels, R_uni, S_down, T)


# ══════════════════════════════════════════════════════════════════════════
# CNN Backbone (unchanged architecture, processes larger R)
# ══════════════════════════════════════════════════════════════════════════

class FactoredCNNBackbone(nn.Module):
    """Factored 1D CNN for container terminal states.

    Processes (B, C, R_uni, S, T) → (B, feat_ch, R_uni, S_down, T).
    """

    def __init__(self, cfg: CNNConfig):
        super().__init__()
        C_in = cfg.n_state_channels
        C1 = cfg.stage1_channels
        C2 = cfg.feat_channels
        G = cfg.gn_groups

        # Stage 1a: (1,21,1) — container profile, RF_S = 21
        self.conv1a = nn.Conv3d(
            C_in, C1, (1, cfg.container_kernel, 1),
            padding=(0, cfg.container_pad, 0),
        )
        self.gn1a = nn.GroupNorm(G, C1)

        # Stage 1b: (1,21,1) stride 4 — RF_S = 41, downsample
        self.conv1b = nn.Conv3d(
            C1, C2, (1, cfg.container_kernel, 1),
            stride=(1, cfg.s_stride, 1),
            padding=(0, cfg.container_pad, 0),
        )
        self.gn1b = nn.GroupNorm(G, C2)

        # Stage 2: (3,1,3) — cross-row/tier context (NOW bridges all regions)
        self.conv2 = nn.Conv3d(
            C2, C2, (cfg.cross_kernel, 1, cfg.cross_kernel),
            padding=(cfg.cross_pad, 0, cfg.cross_pad),
        )
        self.gn2 = nn.GroupNorm(G, C2)

        # Stage 3: (1,5,1) — neighbourhood refinement
        self.conv3 = nn.Conv3d(
            C2, C2, (1, cfg.refine_kernel, 1),
            padding=(0, cfg.refine_pad, 0),
        )
        self.gn3 = nn.GroupNorm(G, C2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, R_uni, S, T) → (B, feat_channels, R_uni, S_down, T)."""
        x = F.relu(self.gn1a(self.conv1a(x)), inplace=True)
        x = F.relu(self.gn1b(self.conv1b(x)), inplace=True)
        x = F.relu(self.gn2(self.conv2(x)), inplace=True)
        x = F.relu(self.gn3(self.conv3(x)), inplace=True)
        return x


# ══════════════════════════════════════════════════════════════════════════
# Occupied-only pooling (unchanged)
# ══════════════════════════════════════════════════════════════════════════

class OccupiedPooling(nn.Module):
    """Max + mean pooling over occupied positions only."""

    def __init__(self, feat_channels: int, global_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * feat_channels, global_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, feat_map: torch.Tensor, occ_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool feat_map using occupancy mask.

        Args:
            feat_map: (B, C, R, S_down, T)
            occ_mask: (B, R, S_down, T) bool
        Returns:
            (B, global_dim)
        """
        mask = occ_mask.unsqueeze(1).float()  # (B, 1, R, S_down, T)
        n_occ = mask.sum(dim=(2, 3, 4)).clamp(min=1.0)  # (B, 1)

        g_mean = (feat_map * mask).sum(dim=(2, 3, 4)) / n_occ
        g_max = torch.where(
            occ_mask.unsqueeze(1), feat_map,
            torch.full_like(feat_map, -1e9),
        ).amax(dim=(2, 3, 4))
        g_max = g_max * (n_occ > 0).float()

        return self.fc(torch.cat([g_mean, g_max], dim=1))


# ══════════════════════════════════════════════════════════════════════════
# Decision Heads
# ══════════════════════════════════════════════════════════════════════════

class SourceHead(nn.Module):
    """Per-position Q-values via 1×1 conv for source selection.

    Replaces ContainerSelectionHead. Now selects from ANY region:
    yard containers, rail containers, delivery trucks, queued trucks.
    """

    def __init__(self, feat_channels: int):
        super().__init__()
        mid = feat_channels // 2
        self.conv = nn.Sequential(
            nn.Conv3d(feat_channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, 1, 1),
        )

    def forward(
        self, feat_map: torch.Tensor, source_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score each position for source selection.

        Args:
            feat_map: (B, C, R_uni, S_down, T)
            source_mask: (B, R_uni, S_down, T) bool
        Returns:
            q_flat: (B, R_uni*S_down*T) with -inf for invalid
        """
        q = self.conv(feat_map).squeeze(1)  # (B, R_uni, S_down, T)
        q_flat = q.reshape(q.size(0), -1)
        mask_flat = source_mask.reshape(source_mask.size(0), -1)
        return q_flat.masked_fill(~mask_flat, float("-inf"))


class IdleSourceWrapper(nn.Module):
    """Wraps any source head, appending a learnable IDLE Q-value.

    The IDLE action lets a crane voluntarily do nothing when no productive
    move is available.  The idle Q-value is derived from mean-pooled spatial
    features, independent of the inner head's architecture.

    Output shape: inner output with +1 on the last (action) dimension.
    Works with standard (B, N), distributional (B, N_q, N), and
    IQN tuple ((B, N_tau, N), tau) outputs.
    """

    def __init__(self, inner: nn.Module, feat_channels: int):
        super().__init__()
        self.inner = inner
        mid = max(feat_channels // 4, 8)
        self.idle_mlp = nn.Sequential(
            nn.Linear(feat_channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, 1),
        )
        # Pessimistic init: IDLE must *earn* its way up from a negative
        # Q-value so it doesn't compete with spatial actions from the start.
        # This is critical for NoisyNet where the spatial head carries
        # exploration noise but the IDLE MLP uses plain nn.Linear — when
        # noise is zeroed (greedy eval) IDLE would otherwise dominate.
        nn.init.constant_(self.idle_mlp[-1].bias, -5.0)

    IDLE_Q_CEIL = -1.0  # IDLE should never look attractive vs productive moves

    def _idle_q(self, feat_map: torch.Tensor) -> torch.Tensor:
        """Compute IDLE Q-value from mean-pooled spatial features. (B, 1).

        Clamped to IDLE_Q_CEIL so the agent always prefers a real move when
        one is available.  The clamp is differentiable on the active side
        (q < ceil) so gradients still flow during training.
        """
        pooled = feat_map.mean(dim=(2, 3, 4))  # (B, C)
        return self.idle_mlp(pooled).clamp(max=self.IDLE_Q_CEIL)  # (B, 1)

    def forward(
        self, feat_map: torch.Tensor, source_mask: torch.Tensor, **kwargs,
    ) -> torch.Tensor:
        result = self.inner(feat_map, source_mask, **kwargs)
        q_idle = self._idle_q(feat_map)  # (B, 1)

        # IQN returns a tuple (q_quantiles, tau)
        if isinstance(result, tuple):
            q_spatial, tau = result
            # (B, N_tau, N) → (B, N_tau, N+1)
            q_idle_exp = q_idle.unsqueeze(1).expand(-1, q_spatial.size(1), -1)
            return torch.cat([q_spatial, q_idle_exp], dim=-1), tau

        # Distributional: (B, N_q, N) → (B, N_q, N+1)
        if result.dim() == 3:
            q_idle_exp = q_idle.unsqueeze(1).expand(-1, result.size(1), -1)
            return torch.cat([result, q_idle_exp], dim=-1)

        # Standard: (B, N) → (B, N+1)
        return torch.cat([result, q_idle], dim=-1)

    def q_mean(
        self, feat_map: torch.Tensor, source_mask: torch.Tensor, **kwargs,
    ) -> torch.Tensor:
        """Mean Q-values with IDLE appended. (B, N+1)."""
        if hasattr(self.inner, "q_mean"):
            q_spatial = self.inner.q_mean(feat_map, source_mask, **kwargs)
        else:
            q_spatial = self.inner(feat_map, source_mask, **kwargs)
        q_idle = self._idle_q(feat_map)
        return torch.cat([q_spatial, q_idle], dim=-1)

    # ── Proxy methods for NoisyNet compatibility ──────────────────

    def reset_noise(self):
        """Forward reset_noise to inner head if it supports it."""
        if hasattr(self.inner, "reset_noise"):
            self.inner.reset_noise()


class UnifiedDestHead(nn.Module):
    """Source-conditioned destination scoring via additive feat_map fusion.

    Given a selected source entity, scores every position in the unified
    grid as a potential destination. The source embedding is broadcast-added
    to the feat_map, then scored with 1×1 conv.

    This naturally handles all move types:
      - Yard→Yard (restack): both source and dest in yard rows
      - Yard→Rail (train load): dest in rail row at wagon position
      - Rail→Yard (train import): dest in yard rows
      - Parking→Yard (truck import): dest in yard rows
      - Queue→Parking (park truck): dest in parking row
      - Yard→Parking (truck load): dest at truck's parking position

    The dest_mask filters to only valid targets per move context.
    """

    def __init__(self, feat_channels: int, global_dim: int):
        super().__init__()
        # Project (global_feat + source_feat) → feat_channels for additive fusion
        self.query_net = nn.Sequential(
            nn.Linear(global_dim + feat_channels, feat_channels),
            nn.ReLU(inplace=True),
        )
        # Score the augmented feat_map
        mid = feat_channels // 2
        self.score_conv = nn.Sequential(
            nn.Conv3d(feat_channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, 1, 1),
        )

    def forward(
        self,
        feat_map: torch.Tensor,
        global_feat: torch.Tensor,
        source_feat: torch.Tensor,
        dest_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score each position for destination selection.

        Args:
            feat_map: (B, F, R_uni, S_down, T)
            global_feat: (B, G)
            source_feat: (B, F) — extracted from feat_map at source position
            dest_mask: (B, R_uni, S_down, T) bool
        Returns:
            q_flat: (B, R_uni*S_down*T) with -inf for invalid
        """
        B = feat_map.size(0)

        # Build query from source context
        query = self.query_net(
            torch.cat([global_feat, source_feat], dim=-1),
        )  # (B, F)

        # Additive fusion: broadcast query across spatial dims
        augmented = feat_map + query.view(B, -1, 1, 1, 1)  # (B, F, R, S, T)

        # Score each position
        q = self.score_conv(augmented).squeeze(1)  # (B, R_uni, S_down, T)
        q_flat = q.reshape(B, -1)
        mask_flat = dest_mask.reshape(B, -1)
        return q_flat.masked_fill(~mask_flat, float("-inf"))


# ══════════════════════════════════════════════════════════════════════════
# Complete Unified Q-Network
# ══════════════════════════════════════════════════════════════════════════

class UnifiedQNetwork(nn.Module):
    """Unified Spatial Q-Network with 2-head architecture.

    Pipeline:
      State (C, R_uni, S, T)
        → CNN Backbone (swappable) → feat_map (F, R_uni, S_down, T)
        → OccupiedPooling          → global_feat (G)
        → Source Head (swappable)   → Q(source)  [masked by source_mask]
        → extract source_feat      → (F,)
        → Dest Head (swappable)    → Q(dest)    [masked by dest_mask]

    All three components (backbone, source_head, dest_head) are swappable
    via constructor injection. Defaults to standard components if not provided.

    Two Q-values per transition: Q_source and Q_dest.
    Move type inferred from coordinates, not predicted.
    """

    def __init__(
        self,
        dims: UnifiedDims,
        cnn_cfg: CNNConfig,
        head_cfg: HeadConfig,
        backbone: nn.Module = None,
        pool: nn.Module = None,
        source_head: nn.Module = None,
        dest_head: nn.Module = None,
    ):
        super().__init__()
        self.dims = dims
        self.cnn_cfg = cnn_cfg

        Fc = cnn_cfg.feat_channels
        G = cnn_cfg.global_dim

        # Swappable components (defaults = baseline)
        self.backbone = backbone if backbone is not None else FactoredCNNBackbone(cnn_cfg)
        self.pool = pool if pool is not None else OccupiedPooling(Fc, G)
        inner_src = source_head if source_head is not None else SourceHead(Fc)
        self.source_head = IdleSourceWrapper(inner_src, Fc)
        self.dest_head = dest_head if dest_head is not None else UnifiedDestHead(Fc, G)

    # ── Encoding ──────────────────────────────────────────────────────

    def _downsample_occ(self, state: torch.Tensor) -> torch.Tensor:
        """Downsample occupancy channel to match feat_map resolution."""
        occ = state[:, _CH_OCCUPANCY:_CH_OCCUPANCY + 1]  # (B, 1, R, S, T)
        stride = self.cnn_cfg.s_stride
        occ_down = F.max_pool3d(occ, (1, stride, 1), stride=(1, stride, 1))
        return occ_down.squeeze(1) > 0.5

    def encode_state(self, state: torch.Tensor) -> EncodedState:
        """State → CNN features + global embedding."""
        feat_map = self.backbone(state)
        occ_down = self._downsample_occ(state)
        global_feat = self.pool(feat_map, occ_down)
        return EncodedState(global_feat=global_feat, feat_map=feat_map)

    def extract_source_feat(
        self, feat_map: torch.Tensor, pos_down: torch.Tensor,
    ) -> torch.Tensor:
        """Extract per-entity feature from feat_map at source position.

        Args:
            feat_map: (B, F, R_uni, S_down, T)
            pos_down: (B, 3) long — (row, s_down, tier)
        Returns:
            (B, feat_channels)
        """
        B = feat_map.size(0)
        idx = torch.arange(B, device=feat_map.device)
        return feat_map[idx, :, pos_down[:, 0], pos_down[:, 1], pos_down[:, 2]]

    # ── Q-value accessors ─────────────────────────────────────────────

    def q_source(
        self, feat_map: torch.Tensor, source_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Q-values for source selection.  (B, N_actions).

        Distributional heads (QR-DQN, IQN) expose a ``q_mean`` method that
        returns the expected Q-values.  If the head has one we call it;
        otherwise the head's ``forward`` already returns the right shape.
        """
        if hasattr(self.source_head, "q_mean"):
            return self.source_head.q_mean(feat_map, source_mask, **kwargs)
        return self.source_head(feat_map, source_mask)

    def q_dest(
        self,
        feat_map: torch.Tensor,
        global_feat: torch.Tensor,
        source_feat: torch.Tensor,
        dest_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Q-values for destination selection.  (B, N_actions)."""
        if hasattr(self.dest_head, "q_mean"):
            return self.dest_head.q_mean(
                feat_map, global_feat, source_feat, dest_mask, **kwargs,
            )
        return self.dest_head(feat_map, global_feat, source_feat, dest_mask)

    # ── Param counting ────────────────────────────────────────────────

    def count_parameters(self) -> dict:
        """Parameter count breakdown by component."""
        def _count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "backbone": _count(self.backbone),
            "pool": _count(self.pool),
            "source_head": _count(self.source_head),
            "dest_head": _count(self.dest_head),
            "total": _count(self),
        }