# simulation/rl/variants/backbones/narrow_deep_backbone.py
"""Narrow-Deep CNN backbone — half channels, double depth (6 layers).

Tests: "Is it better to be narrow+deep or wide+shallow?"

Layer structure (16/32/64 channels, 6 layers):
  conv1a: (10 -> 16, 1x21x1)       — container profile
  conv1b: (16 -> 32, 1x21x1, s=4)  — downsample + profile
  conv2:  (32 -> 32, 3x1x3)        — cross-region / tier (1st pass)
  conv3:  (32 -> 32, 1x5x1)        — neighbourhood refine (1st pass)
  conv4:  (32 -> 32, 3x1x3)        — cross-region / tier (2nd pass, RF_R = 5)
  conv5:  (32 -> 32, 1x5x1)        — neighbourhood refine (2nd pass, RF_S += 8)

Same receptive field as baseline Deeper but with half the channels.
~30K backbone params (vs ~108K baseline).
"""
import torch.nn as nn
import torch.nn.functional as F

from simulation.rl.multihead_dqn.config import CNNConfig


class NarrowDeepCNNBackbone(nn.Module):
    """Half-width, 6-layer CNN backbone.

    Drop-in replacement: (B, C_in, R, S, T) -> (B, feat_ch, R, S_down, T).
    """

    def __init__(self, cfg: CNNConfig):
        super().__init__()
        C_in = cfg.n_state_channels
        C1 = cfg.stage1_channels    # 16 (was 32)
        C2 = cfg.feat_channels      # 32 (was 64)
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

        # Stage 2: (3,1,3) — cross-region/tier (1st pass)
        self.conv2 = nn.Conv3d(
            C2, C2, (cfg.cross_kernel, 1, cfg.cross_kernel),
            padding=(cfg.cross_pad, 0, cfg.cross_pad),
        )
        self.gn2 = nn.GroupNorm(G, C2)

        # Stage 3: (1,5,1) — neighbourhood refine (1st pass)
        self.conv3 = nn.Conv3d(
            C2, C2, (1, cfg.refine_kernel, 1),
            padding=(0, cfg.refine_pad, 0),
        )
        self.gn3 = nn.GroupNorm(G, C2)

        # Stage 4: (3,1,3) — cross-region/tier (2nd pass, RF_R = 5)
        self.conv4 = nn.Conv3d(
            C2, C2, (cfg.cross_kernel, 1, cfg.cross_kernel),
            padding=(cfg.cross_pad, 0, cfg.cross_pad),
        )
        self.gn4 = nn.GroupNorm(G, C2)

        # Stage 5: (1,5,1) — neighbourhood refine (2nd pass)
        self.conv5 = nn.Conv3d(
            C2, C2, (1, cfg.refine_kernel, 1),
            padding=(0, cfg.refine_pad, 0),
        )
        self.gn5 = nn.GroupNorm(G, C2)

    def forward(self, x):
        """(B, C, R, S, T) -> (B, feat_channels, R, S_down, T)."""
        x = F.relu(self.gn1a(self.conv1a(x)), inplace=True)
        x = F.relu(self.gn1b(self.conv1b(x)), inplace=True)
        x = F.relu(self.gn2(self.conv2(x)), inplace=True)
        x = F.relu(self.gn3(self.conv3(x)), inplace=True)
        x = F.relu(self.gn4(self.conv4(x)), inplace=True)
        x = F.relu(self.gn5(self.conv5(x)), inplace=True)
        return x
