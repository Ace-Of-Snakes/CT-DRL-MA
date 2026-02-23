# simulation/rl/variants/munchausen_agent.py
"""Munchausen DQN — reward augmentation via scaled log-policy.

Augments the TD target reward with alpha * tau * log_pi(a|s), where
pi is the softmax policy implied by the Q-values.  This provides an
implicit entropy bonus that encourages action diversity.
"""
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from simulation.rl.base_agent import BaseSpatialDQNAgent
from simulation.rl.multihead_dqn.unified_networks import UnifiedQNetwork
from simulation.rl.multihead_dqn.unified_replay_buffer import UnifiedTransition


class MunchausenDQNAgent(BaseSpatialDQNAgent):
    """M-DQN: adds tau * log pi(a|s) to the reward in TD targets."""

    def _build_networks(self) -> Tuple[nn.Module, nn.Module]:
        q = UnifiedQNetwork(self.cfg.unified, self.cfg.cnn, self.cfg.heads)
        t = UnifiedQNetwork(self.cfg.unified, self.cfg.cnn, self.cfg.heads)
        return q, t

    def _compute_loss(
        self,
        transitions: List[UnifiedTransition],
        weights: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg.training
        gamma = cfg.gamma
        dest_aux_w = cfg.dest_aux_weight
        m_alpha = cfg.munchausen_alpha
        m_tau = cfg.munchausen_tau
        m_clip = cfg.munchausen_clip

        states = np.stack([t.state for t in transitions])
        next_states = np.stack([t.next_state for t in transitions])
        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        dones = np.array([t.done for t in transitions], dtype=np.float32)

        states_t = self._to_tensor(states)
        next_states_t = self._to_tensor(next_states)
        rewards_t = self._to_tensor(rewards)
        dones_t = self._to_tensor(dones)

        # Encode current states
        encoded = self.q_net.encode_state(states_t)
        src_mask_batch = self._batch_source_mask(states_t)
        q_src_all = self.q_net.q_source(encoded.feat_map, src_mask_batch)

        B = len(transitions)
        src_indices = torch.tensor(
            [self._flatten_down(t.source_pos_down) for t in transitions],
            dtype=torch.long, device=self.device,
        )
        q_source_taken = q_src_all[torch.arange(B, device=self.device), src_indices]

        # ── Munchausen reward augmentation ──────────────────────────
        # Compute log-policy from current Q-values (softmax over valid actions)
        with torch.no_grad():
            # Replace -inf with very negative value for stable logsumexp
            q_for_policy = q_src_all.clone()
            mask_with_idle = self._source_mask_with_idle(
                src_mask_batch.reshape(B, -1),
            )
            q_for_policy[~mask_with_idle] = -1e8

            # log pi(a|s) = Q(s,a)/tau - logsumexp(Q(s,.)/tau)
            v_tau = m_tau * torch.logsumexp(q_for_policy / m_tau, dim=1)
            log_pi = (q_for_policy - v_tau.unsqueeze(1)) / m_tau
            log_pi_a = log_pi[torch.arange(B, device=self.device), src_indices]
            log_pi_a = log_pi_a.clamp(min=m_clip, max=0.0)

            munchausen_reward = rewards_t + m_alpha * m_tau * log_pi_a

        # ── Target computation (Double DQN with Munchausen reward) ──
        with torch.no_grad():
            next_src_mask = self._batch_source_mask(next_states_t)

            if cfg.double_dqn:
                online_enc = self.q_net.encode_state(next_states_t)
                q_online = self.q_net.q_source(online_enc.feat_map, next_src_mask)
                best_actions = q_online.argmax(dim=1, keepdim=True)
                target_enc = self.target_net.encode_state(next_states_t)
                q_target = self.target_net.q_source(
                    target_enc.feat_map, next_src_mask,
                )
                max_q_next = q_target.gather(1, best_actions).squeeze(1)
            else:
                target_enc = self.target_net.encode_state(next_states_t)
                q_target = self.target_net.q_source(
                    target_enc.feat_map, next_src_mask,
                )
                max_q_next = q_target.max(dim=1)[0]

            max_q_next = torch.where(
                torch.isfinite(max_q_next), max_q_next,
                torch.zeros_like(max_q_next),
            )
            targets = munchausen_reward + gamma * (1 - dones_t) * max_q_next

        td_errors = targets - q_source_taken

        if weights is not None:
            src_loss = (
                weights * F.smooth_l1_loss(q_source_taken, targets, reduction="none")
            ).mean()
        else:
            src_loss = F.smooth_l1_loss(q_source_taken, targets)

        # ── Auxiliary dest loss (unchanged — no Munchausen here) ────
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
