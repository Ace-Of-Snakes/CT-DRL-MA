# simulation/rl/variants/backbones/wider_backbone.py
"""Wider CNN backbone — same 4-layer structure, 2x channel width.

Tests: "Does the backbone need more representational width?"

Layer structure (identical kernels to baseline, doubled channels):
  conv1a: (10 -> 64, 1x21x1)       — container profile
  conv1b: (64 -> 128, 1x21x1, s=4) — downsample + profile
  conv2:  (128 -> 128, 3x1x3)      — cross-region / tier
  conv3:  (128 -> 128, 1x5x1)      — neighbourhood refine
"""
import torch.nn as nn
import torch.nn.functional as F

from simulation.rl.multihead_dqn.config import CNNConfig


class WiderCNNBackbone(nn.Module):
    """FactoredCNNBackbone with 2x channel width.

    Drop-in replacement: (B, C_in, R, S, T) -> (B, feat_ch, R, S_down, T).
    """

    def __init__(self, cfg: CNNConfig):
        super().__init__()
        C_in = cfg.n_state_channels
        C1 = cfg.stage1_channels    # 64 (was 32)
        C2 = cfg.feat_channels      # 128 (was 64)
        G = cfg.gn_groups

        # Stage 1a: (1,21,1) — container profile
        self.conv1a = nn.Conv3d(
            C_in, C1, (1, cfg.container_kernel, 1),
            padding=(0, cfg.container_pad, 0),
        )
        self.gn1a = nn.GroupNorm(G, C1)

        # Stage 1b: (1,21,1) stride 4 — downsample
        self.conv1b = nn.Conv3d(
            C1, C2, (1, cfg.container_kernel, 1),
            stride=(1, cfg.s_stride, 1),
            padding=(0, cfg.container_pad, 0),
        )
        self.gn1b = nn.GroupNorm(G, C2)

        # Stage 2: (3,1,3) — cross-region/tier
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

    def forward(self, x):
        """(B, C, R, S, T) -> (B, feat_channels, R, S_down, T)."""
        x = F.relu(self.gn1a(self.conv1a(x)), inplace=True)
        x = F.relu(self.gn1b(self.conv1b(x)), inplace=True)
        x = F.relu(self.gn2(self.conv2(x)), inplace=True)
        x = F.relu(self.gn3(self.conv3(x)), inplace=True)
        return x
