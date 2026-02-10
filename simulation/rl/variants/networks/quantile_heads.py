# simulation/rl/variants/networks/quantile_heads.py
"""Quantile-regression heads for QR-DQN.

Each position outputs N quantile values instead of a single scalar,
modelling the full return distribution without Vmin/Vmax tuning.
"""
import torch
import torch.nn as nn


class QuantileSourceHead(nn.Module):
    """Per-position quantile Q-values for source selection."""

    def __init__(self, feat_channels: int, n_quantiles: int = 32):
        super().__init__()
        self.n_quantiles = n_quantiles
        mid = feat_channels // 2
        self.conv = nn.Sequential(
            nn.Conv3d(feat_channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, n_quantiles, 1),
        )

    def forward(
        self, feat_map: torch.Tensor, source_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return (B, N_quantiles, N_actions) with -inf for invalid."""
        q = self.conv(feat_map)  # (B, N_q, R, S_d, T)
        B, Nq = q.shape[:2]
        q_flat = q.reshape(B, Nq, -1)
        mask_flat = source_mask.reshape(B, 1, -1).expand_as(q_flat)
        return q_flat.masked_fill(~mask_flat, float("-inf"))

    def q_mean(
        self, feat_map: torch.Tensor, source_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean over quantiles for action selection. (B, N_actions)."""
        q_all = self.forward(feat_map, source_mask)
        # Mean only over finite values (masked positions stay -inf)
        return q_all.mean(dim=1)


class QuantileDestHead(nn.Module):
    """Source-conditioned quantile destination head."""

    def __init__(
        self, feat_channels: int, global_dim: int, n_quantiles: int = 32,
    ):
        super().__init__()
        self.n_quantiles = n_quantiles
        mid = feat_channels // 2

        self.query_net = nn.Sequential(
            nn.Linear(global_dim + feat_channels, feat_channels),
            nn.ReLU(inplace=True),
        )
        self.conv = nn.Sequential(
            nn.Conv3d(feat_channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, n_quantiles, 1),
        )

    def forward(
        self,
        feat_map: torch.Tensor,
        global_feat: torch.Tensor,
        source_feat: torch.Tensor,
        dest_mask: torch.Tensor,
    ) -> torch.Tensor:
        B = feat_map.size(0)
        query = self.query_net(torch.cat([global_feat, source_feat], dim=-1))
        augmented = feat_map + query.view(B, -1, 1, 1, 1)

        q = self.conv(augmented)
        Nq = q.shape[1]
        q_flat = q.reshape(B, Nq, -1)
        mask_flat = dest_mask.reshape(B, 1, -1).expand_as(q_flat)
        return q_flat.masked_fill(~mask_flat, float("-inf"))

    def q_mean(
        self,
        feat_map: torch.Tensor,
        global_feat: torch.Tensor,
        source_feat: torch.Tensor,
        dest_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(feat_map, global_feat, source_feat, dest_mask).mean(dim=1)
