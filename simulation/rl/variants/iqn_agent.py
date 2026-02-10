# simulation/rl/variants/iqn_agent.py
"""IQN — Implicit Quantile Networks with sampled tau.

Samples quantile fractions tau ~ U([0,1]) at runtime and conditions
the Q-value prediction on the quantile level via cosine embedding.
Uses standard UnifiedQNetwork with swapped IQN heads.
"""
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from simulation.rl.base_agent import BaseSpatialDQNAgent
from simulation.rl.multihead_dqn.unified_networks import UnifiedQNetwork
from simulation.rl.multihead_dqn.unified_replay_buffer import UnifiedTransition
from simulation.rl.variants.networks.iqn_heads import IQNSourceHead, IQNDestHead


# ── Loss utility ──────────────────────────────────────────────────────

def _iqn_loss(
    theta: torch.Tensor,
    target_theta: torch.Tensor,
    tau: torch.Tensor,
    kappa: float = 1.0,
    weights: torch.Tensor = None,
) -> torch.Tensor:
    """IQN loss with quantile Huber.

    Args:
        theta: (B, N) online quantile predictions
        target_theta: (B, N') target quantile values
        tau: (B, N) sampled quantile fractions
        kappa: Huber threshold
        weights: (B,) optional PER importance weights
    Returns:
        scalar loss
    """
    B, N = theta.shape

    # Pairwise TD errors: (B, N, N')
    delta = target_theta.unsqueeze(1) - theta.unsqueeze(2)

    huber = torch.where(
        delta.abs() <= kappa,
        0.5 * delta ** 2,
        kappa * (delta.abs() - 0.5 * kappa),
    )

    # tau weights: |tau_i - I(delta < 0)|
    tau_weights = (tau.unsqueeze(2) - (delta < 0).float()).abs()

    per_sample = (tau_weights * huber).sum(dim=2).mean(dim=1)  # (B,)

    if weights is not None:
        return (weights * per_sample).mean()
    return per_sample.mean()


# ── Agent ─────────────────────────────────────────────────────────────

class IQNAgent(BaseSpatialDQNAgent):
    """IQN with sampled quantile fractions."""

    def _build_networks(self) -> Tuple[nn.Module, nn.Module]:
        emb_dim = self.cfg.training.iqn_embedding_dim
        Fc = self.cfg.cnn.feat_channels
        G = self.cfg.cnn.global_dim

        def _make():
            return UnifiedQNetwork(
                self.cfg.unified, self.cfg.cnn, self.cfg.heads,
                source_head=IQNSourceHead(Fc, n_cos=emb_dim),
                dest_head=IQNDestHead(Fc, G, n_cos=emb_dim),
            )

        return _make(), _make()

    def _compute_loss(
        self,
        transitions: List[UnifiedTransition],
        weights: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg.training
        gamma = cfg.gamma
        dest_aux_w = cfg.dest_aux_weight
        N_train = cfg.iqn_n_quantiles_train

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

        # Sample tau for online predictions — direct head access
        tau = torch.rand(B, N_train, device=self.device)
        q_quantiles, _ = self.q_net.source_head(
            encoded.feat_map, src_mask_batch, n_tau=N_train, tau=tau,
        )

        src_indices = torch.tensor(
            [self._flatten_down(t.source_pos_down) for t in transitions],
            dtype=torch.long, device=self.device,
        )
        # Gather taken action's quantiles: (B, N_train)
        theta = q_quantiles.gather(
            2, src_indices.view(B, 1, 1).expand(B, N_train, 1),
        ).squeeze(2)

        # Target quantiles (sample independent tau')
        with torch.no_grad():
            next_src_mask = self._batch_source_mask(next_states_t)

            # Action selection from mean Q-values (q_source uses q_mean via **kwargs)
            if cfg.double_dqn:
                online_next_enc = self.q_net.encode_state(next_states_t)
                q_mean_next = self.q_net.q_source(
                    online_next_enc.feat_map, next_src_mask, n_tau=N_train,
                )
                best_actions = q_mean_next.argmax(dim=1)
            else:
                target_enc = self.target_net.encode_state(next_states_t)
                q_mean_next = self.target_net.q_source(
                    target_enc.feat_map, next_src_mask, n_tau=N_train,
                )
                best_actions = q_mean_next.argmax(dim=1)

            tau_prime = torch.rand(B, N_train, device=self.device)
            target_enc = self.target_net.encode_state(next_states_t)
            # Direct head access for full quantile output
            target_quantiles, _ = self.target_net.source_head(
                target_enc.feat_map, next_src_mask,
                n_tau=N_train, tau=tau_prime,
            )

            theta_target = target_quantiles.gather(
                2, best_actions.view(B, 1, 1).expand(B, N_train, 1),
            ).squeeze(2)

            theta_target = torch.where(
                torch.isfinite(theta_target), theta_target,
                torch.zeros_like(theta_target),
            )

            T_theta = (
                rewards_t.unsqueeze(1)
                + gamma * (1 - dones_t.unsqueeze(1)) * theta_target
            )

        with torch.no_grad():
            td_errors = (T_theta - theta).mean(dim=1)

        src_loss = _iqn_loss(theta, T_theta, tau, weights=weights)

        # Auxiliary dest loss (scalar, uses q_mean via q_dest)
        total_loss = src_loss
        aux_preds, aux_targets = [], []
        for i, t in enumerate(transitions):
            if t.dest_pos_down is None:
                continue
            sp = t.source_pos_down
            pos = torch.tensor(
                [[sp[0], sp[1], sp[2]]], dtype=torch.long, device=self.device,
            )
            src_feat = self.q_net.extract_source_feat(
                encoded.feat_map[i:i + 1], pos,
            )
            dp = t.dest_pos_down
            dp_flat = self._flatten_down(dp)
            dest_mask = torch.zeros(
                1, self._R, self._s_down, self._T,
                dtype=torch.bool, device=self.device,
            )
            dest_mask[0, dp[0], dp[1], dp[2]] = True
            q_dst = self.q_net.q_dest(
                encoded.feat_map[i:i + 1], encoded.global_feat[i:i + 1],
                src_feat, dest_mask,
            )[0]
            aux_preds.append(q_dst[dp_flat])
            aux_targets.append(t.reward)

        if aux_preds:
            aux_preds_t = torch.stack(aux_preds)
            aux_targets_t = torch.tensor(
                aux_targets, dtype=torch.float32, device=self.device,
            )
            aux_loss = F.smooth_l1_loss(aux_preds_t, aux_targets_t)
            total_loss = src_loss + dest_aux_w * aux_loss

        return total_loss, td_errors.detach()
