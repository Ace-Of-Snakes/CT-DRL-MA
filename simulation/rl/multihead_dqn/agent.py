# multihead_dqn/agent.py
"""Spatial DQN Agent — baseline variant (Double DQN).

Inherits all shared infrastructure from BaseSpatialDQNAgent.
Only provides: network construction + loss (delegates to standard TD).

Backward compatible: same public interface as the original standalone class.
"""
from typing import Optional, Tuple, List

import torch

from simulation.rl.base_agent import (
    BaseSpatialDQNAgent,
    ActionResult,
    DestMaskFn,
    resolve_move_type,
)
from simulation.rl.multihead_dqn.config import MultiHeadDQNConfig
from simulation.rl.multihead_dqn.networks import QNetwork
from simulation.rl.multihead_dqn.replay_buffer import Transition


class DQNAgent(BaseSpatialDQNAgent):
    """Baseline 2-stage spatial DQN agent (standard Double DQN + aux dest loss)."""

    def _build_networks(self) -> Tuple[torch.nn.Module, torch.nn.Module]:
        q = QNetwork(self.cfg.unified, self.cfg.cnn, self.cfg.heads)
        t = QNetwork(self.cfg.unified, self.cfg.cnn, self.cfg.heads)
        return q, t

    def _compute_loss(
        self,
        transitions: List[Transition],
        weights: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._standard_td_loss(transitions, weights)
