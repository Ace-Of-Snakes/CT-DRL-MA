# simulation/rl/variants/backbones/region_aware_backbone.py
"""Region-Aware CNN backbone — per-region convolution branches + deeper cross-region.

Tests: "Does matching kernel size to physical entity size per region help?"

Architecture overview:
  Phase 1 — Per-region branches (different S-kernels):
    Rail    (rows 0-6):   conv1a + conv1b with kernel=11 (smallest container = 10 splits)
    Parking (row 7):      conv1a + conv1b with kernel=21 (one bay = 20 splits)
    Yard    (rows 8-12):  conv1a + conv1b with kernel=11 (container kernel)
    Queue   (rows 13-14): conv1a + conv1b with kernel=21 (bay kernel)

  Phase 2 — Merge:
    torch.cat([rail, parking, yard, queue], dim=R) → (B, 64, 15, 290, 5)

  Phase 3 — Cross-region layers (deeper + skip):
    conv2: (64 -> 64, 3x1x3) + skip   — cross-region/tier (RF_R = 3)
    conv3: (64 -> 64, 1x5x1) + skip   — neighbourhood refine
    conv4: (64 -> 64, 3x1x3) + skip   — 2nd cross-region (RF_R = 5)

Companion RegionAwarePooling performs per-region max+mean pooling,
preserving region identity in the global feature vector.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from simulation.rl.multihead_dqn.config import CNNConfig, UnifiedDims


class RegionAwareCNNBackbone(nn.Module):
    """Per-region convolution branches + deeper cross-region layers with skips.

    Phase 1 slices the input along R into 4 region groups and applies
    region-specific S-kernels (container-length for rail/yard, bay-length
    for parking/queue).  Phase 2 concatenates back to R=15.  Phase 3 runs
    shared cross-region (3,1,3) layers with skip connections for RF_R=5.

    Drop-in replacement: (B, C_in, R, S, T) -> (B, feat_ch, R, S_down, T).
    """

    def __init__(self, cfg: CNNConfig, dims: UnifiedDims):
        super().__init__()
        self.dims = dims
        C_in = cfg.n_state_channels
        C1 = cfg.stage1_channels
        C2 = cfg.feat_channels
        G = cfg.gn_groups

        # Per-region kernel sizes
        k_cont = cfg.container_region_kernel  # 11
        p_cont = cfg.container_region_pad     # 5
        k_bay = cfg.bay_region_kernel         # 21
        p_bay = cfg.bay_region_pad            # 10

        # ── Rail branch (rows 0:7) — container kernel ──
        self.conv1a_rail = nn.Conv3d(
            C_in, C1, (1, k_cont, 1), padding=(0, p_cont, 0),
        )
        self.gn1a_rail = nn.GroupNorm(G, C1)
        self.conv1b_rail = nn.Conv3d(
            C1, C2, (1, k_cont, 1),
            stride=(1, cfg.s_stride, 1), padding=(0, p_cont, 0),
        )
        self.gn1b_rail = nn.GroupNorm(G, C2)

        # ── Parking branch (row 7:8) — bay kernel ──
        self.conv1a_park = nn.Conv3d(
            C_in, C1, (1, k_bay, 1), padding=(0, p_bay, 0),
        )
        self.gn1a_park = nn.GroupNorm(G, C1)
        self.conv1b_park = nn.Conv3d(
            C1, C2, (1, k_bay, 1),
            stride=(1, cfg.s_stride, 1), padding=(0, p_bay, 0),
        )
        self.gn1b_park = nn.GroupNorm(G, C2)

        # ── Yard branch (rows 8:13) — container kernel ──
        self.conv1a_yard = nn.Conv3d(
            C_in, C1, (1, k_cont, 1), padding=(0, p_cont, 0),
        )
        self.gn1a_yard = nn.GroupNorm(G, C1)
        self.conv1b_yard = nn.Conv3d(
            C1, C2, (1, k_cont, 1),
            stride=(1, cfg.s_stride, 1), padding=(0, p_cont, 0),
        )
        self.gn1b_yard = nn.GroupNorm(G, C2)

        # ── Queue branch (rows 13:15) — bay kernel ──
        self.conv1a_queue = nn.Conv3d(
            C_in, C1, (1, k_bay, 1), padding=(0, p_bay, 0),
        )
        self.gn1a_queue = nn.GroupNorm(G, C1)
        self.conv1b_queue = nn.Conv3d(
            C1, C2, (1, k_bay, 1),
            stride=(1, cfg.s_stride, 1), padding=(0, p_bay, 0),
        )
        self.gn1b_queue = nn.GroupNorm(G, C2)

        # ── Shared cross-region layers (deeper style with skips) ──

        # Stage 2: (3,1,3) — cross-region/tier (1st pass, RF_R = 3) + skip
        self.conv2 = nn.Conv3d(
            C2, C2, (cfg.cross_kernel, 1, cfg.cross_kernel),
            padding=(cfg.cross_pad, 0, cfg.cross_pad),
        )
        self.gn2 = nn.GroupNorm(G, C2)

        # Stage 3: (1,5,1) — neighbourhood refinement + skip
        self.conv3 = nn.Conv3d(
            C2, C2, (1, cfg.refine_kernel, 1),
            padding=(0, cfg.refine_pad, 0),
        )
        self.gn3 = nn.GroupNorm(G, C2)

        # Stage 4: (3,1,3) — cross-region/tier (2nd pass, RF_R = 5) + skip
        self.conv4 = nn.Conv3d(
            C2, C2, (cfg.cross_kernel, 1, cfg.cross_kernel),
            padding=(cfg.cross_pad, 0, cfg.cross_pad),
        )
        self.gn4 = nn.GroupNorm(G, C2)

    def _branch_forward(self, x, conv1a, gn1a, conv1b, gn1b):
        """Apply per-region conv1a + conv1b."""
        x = F.relu(gn1a(conv1a(x)), inplace=True)
        x = F.relu(gn1b(conv1b(x)), inplace=True)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, R=15, S=1160, T=5) → (B, feat_ch=64, R=15, S_down=290, T=5)."""
        d = self.dims

        # Phase 1: Per-region branches
        x_rail = self._branch_forward(
            x[:, :, d.rail_row_start:d.rail_row_end],
            self.conv1a_rail, self.gn1a_rail,
            self.conv1b_rail, self.gn1b_rail,
        )
        x_park = self._branch_forward(
            x[:, :, d.parking_row_start:d.parking_row_end],
            self.conv1a_park, self.gn1a_park,
            self.conv1b_park, self.gn1b_park,
        )
        x_yard = self._branch_forward(
            x[:, :, d.yard_row_start:d.yard_row_end],
            self.conv1a_yard, self.gn1a_yard,
            self.conv1b_yard, self.gn1b_yard,
        )
        x_queue = self._branch_forward(
            x[:, :, d.queue_row_start:d.queue_row_end],
            self.conv1a_queue, self.gn1a_queue,
            self.conv1b_queue, self.gn1b_queue,
        )

        # Phase 2: Merge in row order → (B, 64, 15, 290, 5)
        x = torch.cat([x_rail, x_park, x_yard, x_queue], dim=2)

        # Phase 3: Cross-region layers with skip connections
        residual = x
        x = F.relu(self.gn2(self.conv2(x)), inplace=True)
        x = x + residual

        residual = x
        x = F.relu(self.gn3(self.conv3(x)), inplace=True)
        x = x + residual

        residual = x
        x = F.relu(self.gn4(self.conv4(x)), inplace=True)
        x = x + residual

        return x


class RegionAwarePooling(nn.Module):
    """Per-region max+mean pooling preserving region identity.

    Instead of pooling over all occupied positions globally (losing region
    information), this module computes separate max and mean statistics for
    each of the 4 regions (rail, parking, yard, queue), concatenates them,
    and projects to global_dim.

    Input:  feat_map (B, C, R=15, S_down, T=5), occ_mask (B, R, S_down, T)
    Output: (B, global_dim)
    """

    N_REGIONS = 4

    def __init__(
        self, feat_channels: int, global_dim: int, dims: UnifiedDims,
    ):
        super().__init__()
        self.dims = dims
        # 4 regions × (max + mean) × feat_channels = 8 × feat_channels
        self.fc = nn.Sequential(
            nn.Linear(2 * self.N_REGIONS * feat_channels, global_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, feat_map: torch.Tensor, occ_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool feat_map per-region using occupancy mask.

        Args:
            feat_map: (B, C, R, S_down, T)
            occ_mask: (B, R, S_down, T) bool
        Returns:
            (B, global_dim)
        """
        d = self.dims
        regions = [
            (d.rail_row_start, d.rail_row_end),
            (d.parking_row_start, d.parking_row_end),
            (d.yard_row_start, d.yard_row_end),
            (d.queue_row_start, d.queue_row_end),
        ]

        parts = []
        for r_start, r_end in regions:
            fm_reg = feat_map[:, :, r_start:r_end]      # (B, C, R_reg, S, T)
            mask_reg = occ_mask[:, r_start:r_end]        # (B, R_reg, S, T)
            mask_f = mask_reg.unsqueeze(1).float()       # (B, 1, R_reg, S, T)
            n_occ = mask_f.sum(dim=(2, 3, 4)).clamp(min=1.0)  # (B, 1)

            # Mean over occupied positions
            g_mean = (fm_reg * mask_f).sum(dim=(2, 3, 4)) / n_occ  # (B, C)

            # Max over occupied positions (−inf for unoccupied, then amax)
            g_max = torch.where(
                mask_reg.unsqueeze(1), fm_reg,
                torch.full_like(fm_reg, -1e9),
            ).amax(dim=(2, 3, 4))                        # (B, C)
            g_max = g_max * (n_occ > 0).float()

            parts.extend([g_mean, g_max])

        # (B, 8 * feat_channels) → (B, global_dim)
        return self.fc(torch.cat(parts, dim=1))
