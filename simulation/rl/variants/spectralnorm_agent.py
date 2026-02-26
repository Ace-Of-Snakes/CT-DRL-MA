# simulation/rl/variants/spectralnorm_agent.py
"""Spectral Norm DQN — backbone regularization for plasticity preservation.

Applies spectral normalization to all Conv3d layers in the backbone,
constraining the Lipschitz constant. Same heads, same loss, same exploration.
Uses BackboneFactory's spectral_norm flag instead of a separate backbone class.
"""
from typing import Optional, Tuple, List

import torch
import torch.nn as nn

from simulation.rl.base_agent import BaseSpatialDQNAgent
from simulation.rl.backbone_factory import build_backbone
from simulation.rl.multihead_dqn.networks import QNetwork
from simulation.rl.multihead_dqn.replay_buffer import Transition


class SpectralNormDQNAgent(BaseSpatialDQNAgent):
    """DQN with spectral-normalized backbone convolutions."""

    def _build_networks(self) -> Tuple[nn.Module, nn.Module]:
        backbone_q = build_backbone(self.cfg.cnn, spectral_norm=True)
        backbone_t = build_backbone(self.cfg.cnn, spectral_norm=True)
        q = QNetwork(self.cfg.unified, self.cfg.cnn, self.cfg.heads, backbone=backbone_q)
        t = QNetwork(self.cfg.unified, self.cfg.cnn, self.cfg.heads, backbone=backbone_t)
        return q, t

    def _compute_loss(
        self, transitions: List[Transition], weights: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._standard_td_loss(transitions, weights)
