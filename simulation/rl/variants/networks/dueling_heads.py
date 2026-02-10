# simulation/rl/variants/networks/dueling_heads.py
"""Dueling heads: Q(s,a) = V(s) + A(s,a) - mean(A).

Separates state-value estimation from action-advantage estimation,
improving learning when many actions have similar values.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingSourceHead(nn.Module):
    """Per-position Q-values with dueling V+A decomposition."""

    def __init__(self, feat_channels: int):
        super().__init__()
        mid = feat_channels // 2

        # Value stream: spatial features → global pool → scalar V(s)
        self.value_conv = nn.Sequential(
            nn.Conv3d(feat_channels, mid, 1),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Linear(mid, 1)

        # Advantage stream: per-position A(s,a)
        self.adv_conv = nn.Sequential(
            nn.Conv3d(feat_channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, 1, 1),
        )

    def forward(
        self, feat_map: torch.Tensor, source_mask: torch.Tensor,
    ) -> torch.Tensor:
        B = feat_map.size(0)
        mask_flat = source_mask.reshape(B, -1)

        # V(s): pool value features globally
        v_feats = self.value_conv(feat_map)
        v = self.value_fc(v_feats.mean(dim=(2, 3, 4)))  # (B, 1)

        # A(s,a): per-position advantages
        adv = self.adv_conv(feat_map).squeeze(1)  # (B, R, S_d, T)
        adv_flat = adv.reshape(B, -1)

        # Mean advantage over valid actions only
        adv_masked = adv_flat.masked_fill(~mask_flat, 0.0)
        n_valid = mask_flat.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        adv_mean = adv_masked.sum(dim=1, keepdim=True) / n_valid

        q = v + adv_flat - adv_mean
        return q.masked_fill(~mask_flat, float("-inf"))


class DuelingDestHead(nn.Module):
    """Source-conditioned dueling destination head."""

    def __init__(self, feat_channels: int, global_dim: int):
        super().__init__()
        mid = feat_channels // 2

        # Shared query projection (same as UnifiedDestHead)
        self.query_net = nn.Sequential(
            nn.Linear(global_dim + feat_channels, feat_channels),
            nn.ReLU(inplace=True),
        )

        # Value stream
        self.value_conv = nn.Sequential(
            nn.Conv3d(feat_channels, mid, 1),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Linear(mid, 1)

        # Advantage stream
        self.adv_conv = nn.Sequential(
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
        B = feat_map.size(0)
        mask_flat = dest_mask.reshape(B, -1)

        query = self.query_net(torch.cat([global_feat, source_feat], dim=-1))
        augmented = feat_map + query.view(B, -1, 1, 1, 1)

        v_feats = self.value_conv(augmented)
        v = self.value_fc(v_feats.mean(dim=(2, 3, 4)))

        adv = self.adv_conv(augmented).squeeze(1)
        adv_flat = adv.reshape(B, -1)

        adv_masked = adv_flat.masked_fill(~mask_flat, 0.0)
        n_valid = mask_flat.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        adv_mean = adv_masked.sum(dim=1, keepdim=True) / n_valid

        q = v + adv_flat - adv_mean
        return q.masked_fill(~mask_flat, float("-inf"))
