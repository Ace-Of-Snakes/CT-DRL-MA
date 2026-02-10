# simulation/rl/variants/dueling_agent.py
"""Dueling DQN — V(s) + A(s,a) decomposition.

Separates state-value estimation from action-advantage estimation,
improving learning when many actions in a state have similar Q-values.
Uses standard UnifiedQNetwork with swapped Dueling heads.
"""
from typing import Optional, Tuple, List

import torch
import torch.nn as nn

from simulation.rl.base_agent import BaseSpatialDQNAgent
from simulation.rl.multihead_dqn.unified_networks import UnifiedQNetwork
from simulation.rl.multihead_dqn.unified_replay_buffer import UnifiedTransition
from simulation.rl.variants.networks.dueling_heads import (
    DuelingSourceHead, DuelingDestHead,
)


class DuelingDQNAgent(BaseSpatialDQNAgent):
    """Dueling DQN with V+A head decomposition."""

    def _build_networks(self) -> Tuple[nn.Module, nn.Module]:
        Fc = self.cfg.cnn.feat_channels
        G = self.cfg.cnn.global_dim

        def _make():
            return UnifiedQNetwork(
                self.cfg.unified, self.cfg.cnn, self.cfg.heads,
                source_head=DuelingSourceHead(Fc),
                dest_head=DuelingDestHead(Fc, G),
            )

        return _make(), _make()

    def _compute_loss(
        self, transitions: List[UnifiedTransition], weights: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._standard_td_loss(transitions, weights)
