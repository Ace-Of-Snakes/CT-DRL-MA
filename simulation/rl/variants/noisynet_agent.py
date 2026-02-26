# simulation/rl/variants/noisynet_agent.py
"""NoisyNet DQN — parametric noise replaces epsilon-greedy exploration.

Injects factorized Gaussian noise into the scoring layers of both
source and destination heads. Epsilon is set to 0 during normal
operation; exploration comes entirely from network noise.
Uses standard QNetwork with swapped Noisy heads.
"""
import random
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from simulation.rl.base_agent import BaseSpatialDQNAgent
from simulation.rl.multihead_dqn.networks import QNetwork
from simulation.rl.multihead_dqn.replay_buffer import Transition
from simulation.rl.variants.networks.noisy_linear import NoisyLinear


# ── Noisy heads ───────────────────────────────────────────────────────

class NoisySourceHead(nn.Module):
    """SourceHead with NoisyLinear scoring layer."""

    def __init__(self, feat_channels: int, sigma0: float = 0.5):
        super().__init__()
        mid = feat_channels // 2
        self.conv1 = nn.Conv3d(feat_channels, mid, 1)
        self.noisy = NoisyLinear(mid, 1, sigma0=sigma0)

    def forward(self, feat_map: torch.Tensor, source_mask: torch.Tensor):
        B, C, R, Sd, T = feat_map.shape
        x = F.relu(self.conv1(feat_map), inplace=True)
        mid = x.shape[1]
        x = x.permute(0, 2, 3, 4, 1).reshape(-1, mid)
        q = self.noisy(x).view(B, R, Sd, T)
        q_flat = q.reshape(B, -1)
        mask_flat = source_mask.reshape(B, -1)
        return q_flat.masked_fill(~mask_flat, float("-inf"))


class NoisyDestHead(nn.Module):
    """Source-conditioned DestHead with NoisyLinear scoring."""

    def __init__(self, feat_channels: int, global_dim: int, sigma0: float = 0.5):
        super().__init__()
        self.query_net = nn.Sequential(
            nn.Linear(global_dim + feat_channels, feat_channels),
            nn.ReLU(inplace=True),
        )
        mid = feat_channels // 2
        self.conv1 = nn.Conv3d(feat_channels, mid, 1)
        self.noisy = NoisyLinear(mid, 1, sigma0=sigma0)

    def forward(self, feat_map, global_feat, source_feat, dest_mask):
        B, C, R, Sd, T = feat_map.shape
        query = self.query_net(torch.cat([global_feat, source_feat], dim=-1))
        augmented = feat_map + query.view(B, -1, 1, 1, 1)
        x = F.relu(self.conv1(augmented), inplace=True)
        mid = x.shape[1]
        x = x.permute(0, 2, 3, 4, 1).reshape(-1, mid)
        q = self.noisy(x).view(B, R, Sd, T)
        q_flat = q.reshape(B, -1)
        mask_flat = dest_mask.reshape(B, -1)
        return q_flat.masked_fill(~mask_flat, float("-inf"))


# ── Agent ─────────────────────────────────────────────────────────────

class NoisyNetDQNAgent(BaseSpatialDQNAgent):
    """NoisyNet: parametric exploration via noisy scoring layers."""

    def _build_networks(self) -> Tuple[nn.Module, nn.Module]:
        sigma0 = self.cfg.training.noisy_sigma0
        Fc = self.cfg.cnn.feat_channels
        G = self.cfg.cnn.global_dim

        def _make():
            return QNetwork(
                self.cfg.unified, self.cfg.cnn, self.cfg.heads,
                source_head=NoisySourceHead(Fc, sigma0),
                dest_head=NoisyDestHead(Fc, G, sigma0),
            )

        return _make(), _make()

    def _get_epsilon(self) -> float:
        """No epsilon-greedy; noise provides exploration."""
        if self.epsilon_override is not None:
            return self.epsilon_override
        return 0.0

    # ── Noise scale annealing ─────────────────────────────────────

    def set_noise_scale(self, scale: float) -> None:
        """Set noise multiplier on all NoisyLinear layers in both networks."""
        for net in (self.q_net, self.target_net):
            for m in net.modules():
                if isinstance(m, NoisyLinear):
                    m.noise_scale = scale

    def set_tutorial_noise(self, epoch: int) -> None:
        """Linearly anneal noise from 1.0 → noisy_sigma_min over training."""
        cfg = self.cfg.training
        frac = min(1.0, epoch / max(cfg.noisy_decay_epochs, 1))
        scale = 1.0 - frac * (1.0 - cfg.noisy_sigma_min)
        self.set_noise_scale(scale)

    def _select_source_action(self, q_flat, valid_mask_flat, eps):
        if eps > 0 and random.random() < eps:
            valid = np.where(valid_mask_flat)[0]
            return int(random.choice(valid))
        return int(q_flat.argmax().item())

    def _select_dest_action(self, q_flat, valid_mask_flat, eps):
        if eps > 0 and random.random() < eps:
            valid = np.where(valid_mask_flat)[0]
            return int(random.choice(valid))
        return int(q_flat.argmax().item())

    def _pre_act_hook(self):
        """Re-sample noise before each action for fresh exploration."""
        for m in self.q_net.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def _post_optimize_hook(self):
        """Re-sample noise on both networks after each optimization step."""
        for m in self.q_net.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()
        for m in self.target_net.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def _compute_loss(
        self, transitions: List[Transition], weights: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._standard_td_loss(transitions, weights)
