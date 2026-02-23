# simulation/rl/variants/kitchen_sink_agent.py
"""Kitchen Sink DQN — combines all independently successful components.

Following the Rainbow philosophy of Hessel et al. (2018), this agent
merges the four orthogonal improvements that survived the Phase 1
tutorial screening into a single architecture:

  1. **Deeper-Residual backbone** — 5-layer CNN with skip connections,
     providing a larger cross-region receptive field (RF_R=5) and
     stable gradient flow.  (From CNN screening: Deeper + Residual.)

  2. **Spectral normalisation** — applied to all Conv3d layers in the
     backbone, constraining the Lipschitz constant for stable off-policy
     learning.  (From DQN screening: SpectralNorm.)

  3. **Munchausen reward augmentation** — adds alpha * tau * log pi(a|s)
     to the TD target reward, providing implicit KL regularisation.
     (From DQN screening: Munchausen.)

  4. **NoisyNet exploration** — parametric noise in the scoring layers
     replaces epsilon-greedy, enabling state-dependent exploration.
     (From DQN screening: NoisyNet.)

The backbone + spectral norm are handled by the backbone factory
(variant="kitchen_sink", spectral_norm=True).  This agent class
combines Munchausen loss with NoisyNet heads.
"""
import random
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from simulation.rl.base_agent import BaseSpatialDQNAgent
from simulation.rl.backbone_factory import build_backbone
from simulation.rl.multihead_dqn.unified_networks import UnifiedQNetwork
from simulation.rl.multihead_dqn.unified_replay_buffer import UnifiedTransition
from simulation.rl.variants.noisynet_agent import NoisySourceHead, NoisyDestHead
from simulation.rl.variants.networks.noisy_linear import NoisyLinear


class KitchenSinkDQNAgent(BaseSpatialDQNAgent):
    """Combined agent: Deeper-Residual backbone + SpectralNorm + Munchausen + NoisyNet.

    - Backbone: KitchenSinkCNNBackbone with spectral normalisation
    - Heads: NoisySourceHead + NoisyDestHead (parametric exploration)
    - Loss: Munchausen-augmented TD (reward += alpha * tau * log_pi)
    - Exploration: NoisyNet (epsilon = 0, noise provides exploration)
    """

    def _build_networks(self) -> Tuple[nn.Module, nn.Module]:
        sigma0 = self.cfg.training.noisy_sigma0
        Fc = self.cfg.cnn.feat_channels
        G = self.cfg.cnn.global_dim

        def _make():
            backbone = build_backbone(
                self.cfg.cnn,
                variant="kitchen_sink",
                spectral_norm=True,
            )
            return UnifiedQNetwork(
                self.cfg.unified, self.cfg.cnn, self.cfg.heads,
                backbone=backbone,
                source_head=NoisySourceHead(Fc, sigma0),
                dest_head=NoisyDestHead(Fc, G, sigma0),
            )

        return _make(), _make()

    # ── NoisyNet exploration (no epsilon-greedy) ─────────────────────

    def _get_epsilon(self) -> float:
        """No epsilon-greedy; noise provides exploration."""
        if self.epsilon_override is not None:
            return self.epsilon_override
        return 0.0

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

    # ── Munchausen loss ──────────────────────────────────────────────

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
        with torch.no_grad():
            q_for_policy = q_src_all.clone()
            mask_with_idle = self._source_mask_with_idle(
                src_mask_batch.reshape(B, -1),
            )
            q_for_policy[~mask_with_idle] = -1e8

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
