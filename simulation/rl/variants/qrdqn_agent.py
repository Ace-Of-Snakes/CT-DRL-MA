# simulation/rl/variants/qrdqn_agent.py
"""QR-DQN — Quantile Regression DQN with fixed quantile midpoints.

Models the full return distribution using N fixed quantile atoms.
Uses quantile Huber loss instead of standard TD error.
Uses standard QNetwork with swapped Quantile heads.
"""
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from simulation.rl.base_agent import BaseSpatialDQNAgent
from simulation.rl.multihead_dqn.networks import QNetwork
from simulation.rl.multihead_dqn.replay_buffer import Transition
from simulation.rl.variants.networks.quantile_heads import (
    QuantileSourceHead, QuantileDestHead,
)


# ── Loss utilities ────────────────────────────────────────────────────

def _quantile_huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    taus: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Quantile Huber loss.

    Args:
        predictions: (B, N) predicted quantile values
        targets: (B, N) target quantile values
        taus: (N,) quantile midpoints
        kappa: Huber threshold (1.0 = standard)
    Returns:
        scalar loss
    """
    B, N = predictions.shape
    # Pairwise differences: (B, N_pred, N_target)
    delta = targets.unsqueeze(1) - predictions.unsqueeze(2)  # (B, N, N)

    # Huber loss
    huber = torch.where(
        delta.abs() <= kappa,
        0.5 * delta ** 2,
        kappa * (delta.abs() - 0.5 * kappa),
    )

    # Quantile weights: |tau - I(delta < 0)|
    tau_weights = (taus.view(1, N, 1) - (delta < 0).float()).abs()

    loss = (tau_weights * huber).sum(dim=2).mean(dim=1)  # (B,)
    return loss.mean()


def _quantile_huber_loss_per_sample(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    taus: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Per-sample quantile Huber loss (for PER weighting). Returns (B,)."""
    B, N = predictions.shape
    delta = targets.unsqueeze(1) - predictions.unsqueeze(2)
    huber = torch.where(
        delta.abs() <= kappa,
        0.5 * delta ** 2,
        kappa * (delta.abs() - 0.5 * kappa),
    )
    tau_weights = (taus.view(1, N, 1) - (delta < 0).float()).abs()
    return (tau_weights * huber).sum(dim=2).mean(dim=1)


# ── Agent ─────────────────────────────────────────────────────────────

class QRDQNAgent(BaseSpatialDQNAgent):
    """QR-DQN with fixed quantile midpoints."""

    def __init__(self, cfg):
        super().__init__(cfg)
        N = cfg.training.n_quantiles
        self._taus = (torch.arange(N, dtype=torch.float32) + 0.5) / N
        self._taus = self._taus.to(self.device)

    def _build_networks(self) -> Tuple[nn.Module, nn.Module]:
        N = self.cfg.training.n_quantiles
        Fc = self.cfg.cnn.feat_channels
        G = self.cfg.cnn.global_dim

        def _make():
            return QNetwork(
                self.cfg.unified, self.cfg.cnn, self.cfg.heads,
                source_head=QuantileSourceHead(Fc, N),
                dest_head=QuantileDestHead(Fc, G, N),
            )

        return _make(), _make()

    def _compute_loss(
        self,
        transitions: List[Transition],
        weights: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg.training
        gamma = cfg.gamma
        dest_aux_w = cfg.dest_aux_weight
        N = cfg.n_quantiles

        states = np.stack([t.state for t in transitions])
        next_states = np.stack([t.next_state for t in transitions])
        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        dones = np.array([t.done for t in transitions], dtype=np.float32)

        states_t = self._to_tensor(states)
        next_states_t = self._to_tensor(next_states)
        rewards_t = self._to_tensor(rewards)
        dones_t = self._to_tensor(dones)

        B = len(transitions)
        encoded = self.q_net.encode_state(states_t)
        src_mask_batch = self._batch_source_mask(states_t)

        # Current quantile predictions: (B, N_q, N_actions) — direct head access
        q_quantiles = self.q_net.source_head(encoded.feat_map, src_mask_batch)
        src_indices = torch.tensor(
            [self._flatten_down(t.source_pos_down) for t in transitions],
            dtype=torch.long, device=self.device,
        )
        # Gather taken action's quantiles: (B, N_q)
        theta = q_quantiles[:, :, :].gather(
            2, src_indices.view(B, 1, 1).expand(B, N, 1),
        ).squeeze(2)

        # Target quantiles
        with torch.no_grad():
            next_src_mask = self._batch_source_mask(next_states_t)

            # Action selection from mean Q-values (Double DQN)
            if cfg.double_dqn:
                online_next_enc = self.q_net.encode_state(next_states_t)
                q_mean_next = self.q_net.q_source(
                    online_next_enc.feat_map, next_src_mask,
                )
                best_actions = q_mean_next.argmax(dim=1)
            else:
                target_next_enc = self.target_net.encode_state(next_states_t)
                q_mean_next = self.target_net.q_source(
                    target_next_enc.feat_map, next_src_mask,
                )
                best_actions = q_mean_next.argmax(dim=1)

            target_enc = self.target_net.encode_state(next_states_t)
            # Direct head access for full quantile output
            target_quantiles = self.target_net.source_head(
                target_enc.feat_map, next_src_mask,
            )
            # Gather best action's target quantiles: (B, N_q)
            theta_target = target_quantiles.gather(
                2, best_actions.view(B, 1, 1).expand(B, N, 1),
            ).squeeze(2)

            # Handle all-masked states
            theta_target = torch.where(
                torch.isfinite(theta_target), theta_target,
                torch.zeros_like(theta_target),
            )

            gamma_n = gamma ** cfg.n_step
            T_theta = rewards_t.unsqueeze(1) + gamma_n * (1 - dones_t.unsqueeze(1)) * theta_target

        # Quantile Huber loss
        src_loss = _quantile_huber_loss(theta, T_theta, self._taus)

        if weights is not None:
            # Approximate per-sample weighting via mean quantile error
            with torch.no_grad():
                td_errors = (T_theta - theta).mean(dim=1)
            src_loss = (weights * _quantile_huber_loss_per_sample(
                theta, T_theta, self._taus,
            )).mean()
        else:
            with torch.no_grad():
                td_errors = (T_theta - theta).mean(dim=1)

        # Auxiliary dest loss (shared helper)
        total_loss = src_loss
        aux_loss = self._aux_dest_loss(encoded, transitions)
        if aux_loss is not None:
            total_loss = src_loss + dest_aux_w * aux_loss

        return total_loss, td_errors.detach()
