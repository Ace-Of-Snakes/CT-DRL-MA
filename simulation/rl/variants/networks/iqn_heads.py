# simulation/rl/variants/networks/iqn_heads.py
"""Implicit Quantile Network heads with cosine embedding.

Samples quantile fractions tau ~ U([0,1]) at runtime and uses a cosine
basis embedding to condition the Q-value prediction on the quantile level.

Memory-efficient implementation: applies tau modulation to a compact
global feature vector rather than the full spatial feature map, avoiding
O(B * N_tau * C * R * S * T) intermediates.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineEmbedding(nn.Module):
    """Map quantile fractions tau to feature-sized embeddings via cosine basis."""

    def __init__(self, n_cos: int = 64, embedding_dim: int = 64):
        super().__init__()
        self.n_cos = n_cos
        self.fc = nn.Linear(n_cos, embedding_dim)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """tau: (B, N_tau) -> (B, N_tau, embedding_dim)."""
        i_pi = (
            torch.arange(1, self.n_cos + 1, device=tau.device, dtype=torch.float32)
            * math.pi
        )
        cos_features = torch.cos(tau.unsqueeze(-1) * i_pi)  # (B, N, n_cos)
        return F.relu(self.fc(cos_features))


class IQNSourceHead(nn.Module):
    """IQN source head — memory efficient.

    Instead of expanding the full (B, C, R, Sd, T) feature map across N_tau
    quantile samples, we:
      1. Reduce spatial features to per-position scores: (B, 1, R, Sd, T)
      2. Pool features globally for tau modulation: (B, C)
      3. Combine tau-conditioned scaling with per-position scores

    This keeps memory at O(B*N_tau*C) instead of O(B*N_tau*C*R*Sd*T).
    """

    def __init__(self, feat_channels: int, n_cos: int = 64):
        super().__init__()
        self.feat_channels = feat_channels
        self.cos_embed = CosineEmbedding(n_cos, feat_channels)

        mid = feat_channels // 2
        # Spatial scoring (shared across quantiles)
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(feat_channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, 1, 1),
        )
        # Tau-conditioned scale and bias for each quantile
        self.tau_scale = nn.Linear(feat_channels, 1)
        self.tau_bias = nn.Linear(feat_channels, 1)

    def forward(
        self,
        feat_map: torch.Tensor,
        source_mask: torch.Tensor,
        n_tau: int = 8,
        tau: torch.Tensor = None,
    ) -> tuple:
        """Returns (q_quantiles: (B, N_tau, N_actions), tau: (B, N_tau))."""
        B, C, R, Sd, T = feat_map.shape
        device = feat_map.device

        if tau is None:
            tau = torch.rand(B, n_tau, device=device)
        N = tau.shape[1]

        # Per-position base scores: (B, R*Sd*T)
        base_q = self.spatial_conv(feat_map).squeeze(1).reshape(B, -1)

        # Global feature for tau conditioning: (B, C)
        global_feat = feat_map.mean(dim=(2, 3, 4))

        # Tau embedding: (B, N, C)
        tau_embed = self.cos_embed(tau)

        # Modulate global feature with tau: (B, N, C)
        modulated = global_feat.unsqueeze(1) * tau_embed

        # Per-quantile scale and bias: (B, N, 1)
        scale = self.tau_scale(modulated)   # (B, N, 1)
        bias = self.tau_bias(modulated)     # (B, N, 1)

        # Q-values per quantile: (B, N, N_actions)
        q_flat = scale * base_q.unsqueeze(1) + bias

        mask_flat = source_mask.reshape(B, 1, -1).expand_as(q_flat)
        q_flat = q_flat.masked_fill(~mask_flat, float("-inf"))
        return q_flat, tau

    def q_mean(
        self, feat_map: torch.Tensor, source_mask: torch.Tensor, n_tau: int = 32,
    ) -> torch.Tensor:
        """Mean over quantiles for action selection. (B, N_actions)."""
        q_all, _ = self.forward(feat_map, source_mask, n_tau=n_tau)
        return q_all.mean(dim=1)


class IQNDestHead(nn.Module):
    """IQN destination head with source conditioning — memory efficient."""

    def __init__(self, feat_channels: int, global_dim: int, n_cos: int = 64):
        super().__init__()
        self.feat_channels = feat_channels
        self.cos_embed = CosineEmbedding(n_cos, feat_channels)
        self.query_net = nn.Sequential(
            nn.Linear(global_dim + feat_channels, feat_channels),
            nn.ReLU(inplace=True),
        )
        mid = feat_channels // 2
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(feat_channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, 1, 1),
        )
        self.tau_scale = nn.Linear(feat_channels, 1)
        self.tau_bias = nn.Linear(feat_channels, 1)

    def forward(
        self,
        feat_map: torch.Tensor,
        global_feat: torch.Tensor,
        source_feat: torch.Tensor,
        dest_mask: torch.Tensor,
        n_tau: int = 8,
        tau: torch.Tensor = None,
    ) -> tuple:
        B, C, R, Sd, T = feat_map.shape
        device = feat_map.device

        query = self.query_net(torch.cat([global_feat, source_feat], dim=-1))
        augmented = feat_map + query.view(B, -1, 1, 1, 1)

        if tau is None:
            tau = torch.rand(B, n_tau, device=device)
        N = tau.shape[1]

        base_q = self.spatial_conv(augmented).squeeze(1).reshape(B, -1)
        pool_feat = augmented.mean(dim=(2, 3, 4))

        tau_embed = self.cos_embed(tau)
        modulated = pool_feat.unsqueeze(1) * tau_embed

        scale = self.tau_scale(modulated)
        bias = self.tau_bias(modulated)

        q_flat = scale * base_q.unsqueeze(1) + bias

        mask_flat = dest_mask.reshape(B, 1, -1).expand_as(q_flat)
        q_flat = q_flat.masked_fill(~mask_flat, float("-inf"))
        return q_flat, tau

    def q_mean(
        self,
        feat_map: torch.Tensor,
        global_feat: torch.Tensor,
        source_feat: torch.Tensor,
        dest_mask: torch.Tensor,
        n_tau: int = 32,
    ) -> torch.Tensor:
        q_all, _ = self.forward(
            feat_map, global_feat, source_feat, dest_mask, n_tau=n_tau,
        )
        return q_all.mean(dim=1)
