# simulation/rl/base_agent.py
"""Abstract base for all 2-stage spatial DQN agent variants.

Extracts reusable infrastructure from UnifiedDQNAgent:
  - 2-stage act (source → destination, with ε-greedy hooks)
  - Replay buffer management
  - Optimize loop (sample, loss, backprop, target update)
  - Epsilon scheduling (main + tutorial override)
  - Mask helpers (downsample, unflatten, etc.)
  - Save / load

Subclasses implement only:
  _build_networks()  — create the Q-network pair
  _compute_loss()    — variant-specific loss function
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from simulation.rl.multihead_dqn.config import MultiHeadDQNConfig, UnifiedDims
from simulation.rl.multihead_dqn.unified_replay_buffer import (
    UnifiedReplayBuffer, UnifiedPrioritizedReplayBuffer, UnifiedTransition,
)

# Channel indices (must match unified_state_encoder.UnifiedChannelSpec)
_CH_OCCUPANCY = 0
_CH_CONTAINER_START = 1

# Type alias for dest mask builder callable
DestMaskFn = Callable[[int, int, int], np.ndarray]


# ══════════════════════════════════════════════════════════════════════════
# Move type resolution from spatial regions
# ══════════════════════════════════════════════════════════════════════════

_MOVE_TYPE_MAP: Dict[Tuple[str, str], str] = {
    ("QUEUE", "PARKING"):  "PARK_TRUCK",
    ("RAIL", "YARD"):      "TRAIN_TO_YARD",
    ("PARKING", "YARD"):   "TRUCK_TO_YARD",
    ("YARD", "YARD"):      "YARD_TO_YARD",
    ("YARD", "RAIL"):      "YARD_TO_TRAIN",
    ("YARD", "PARKING"):   "YARD_TO_TRUCK",
}


def resolve_move_type(source_region: str, dest_region: str) -> Optional[str]:
    return _MOVE_TYPE_MAP.get((source_region, dest_region))


# ══════════════════════════════════════════════════════════════════════════
# Action result
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class UnifiedActionResult:
    """Complete action from the unified agent."""
    source_pos: Optional[Tuple[int, int, int]] = None
    dest_pos: Optional[Tuple[int, int, int]] = None
    source_region: Optional[str] = None
    dest_region: Optional[str] = None
    move_type: Optional[str] = None
    source_pos_down: Optional[Tuple[int, int, int]] = None
    dest_pos_down: Optional[Tuple[int, int, int]] = None
    q_values: Optional[Dict[str, float]] = None


# ══════════════════════════════════════════════════════════════════════════
# Base Agent ABC
# ══════════════════════════════════════════════════════════════════════════

class BaseSpatialDQNAgent(ABC):
    """Abstract base for 2-stage spatial DQN agents.

    Public interface (required by tutorial runner + curriculum trainer):
        act(state, source_mask, dest_mask_fn, epsilon) -> UnifiedActionResult
        remember(transition)
        optimize() -> float
        save(path), load(path)
        set_tutorial_epsilon(epoch), clear_epsilon_override()
        _get_epsilon() -> float

    Attributes (accessed externally):
        q_net, target_net, optimizer, replay, step_count, losses, cfg
    """

    def __init__(self, cfg: MultiHeadDQNConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dims = cfg.unified
        self._s_stride = cfg.cnn.s_stride
        self._s_down = cfg.s_down
        self._R = cfg.unified.R_unified
        self._T = cfg.unified.n_tiers

        # Networks (variant-specific)
        self.q_net, self.target_net = self._build_networks()
        self.q_net = self.q_net.to(self.device)
        self.target_net = self.target_net.to(self.device)
        self._hard_update_target()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.training.lr)

        # Replay buffer
        if cfg.training.use_per:
            self.replay = UnifiedPrioritizedReplayBuffer(
                cfg.training.replay_size, cfg.training.per_alpha,
            )
        else:
            self.replay = UnifiedReplayBuffer(cfg.training.replay_size)

        self.step_count = 0
        self.losses: List[float] = []
        self.epsilon_override: Optional[float] = None

    # ── Abstract hooks ──────────────────────────────────────────────

    @abstractmethod
    def _build_networks(self) -> Tuple[nn.Module, nn.Module]:
        """Return (q_net, target_net). Both must expose:
            .encode_state(state) -> EncodedState
            .extract_source_feat(feat_map, pos) -> Tensor
            .q_source(feat_map, mask) -> (B, N_actions)
            .q_dest(feat_map, global_feat, src_feat, mask) -> (B, N_actions)
        """
        ...

    @abstractmethod
    def _compute_loss(
        self,
        transitions: List[UnifiedTransition],
        weights: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute loss and TD errors. Return (loss, td_errors_detached)."""
        ...

    # ── Overridable hooks (with defaults) ───────────────────────────

    def _select_source_action(
        self, q_flat: torch.Tensor, valid_mask_flat: np.ndarray, eps: float,
    ) -> int:
        """Default: ε-greedy. NoisyNet overrides to always use argmax."""
        if random.random() < eps:
            valid = np.where(valid_mask_flat)[0]
            return int(random.choice(valid))
        return int(q_flat.argmax().item())

    def _select_dest_action(
        self, q_flat: torch.Tensor, valid_mask_flat: np.ndarray, eps: float,
    ) -> int:
        if random.random() < eps:
            valid = np.where(valid_mask_flat)[0]
            return int(random.choice(valid))
        return int(q_flat.argmax().item())

    def _post_optimize_hook(self):
        """Called after each optimize() step. Override for noise reset etc."""
        pass

    # ── Target network ──────────────────────────────────────────────

    def _hard_update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def _soft_update_target(self):
        tau = self.cfg.training.target_tau
        for tp, op in zip(self.target_net.parameters(), self.q_net.parameters()):
            tp.data.mul_(1 - tau).add_(op.data, alpha=tau)

    # ── Epsilon ─────────────────────────────────────────────────────

    def _get_epsilon(self) -> float:
        if self.epsilon_override is not None:
            return self.epsilon_override
        cfg = self.cfg.training
        frac = min(1.0, self.step_count / max(cfg.epsilon_decay_steps, 1))
        return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)

    def set_tutorial_epsilon(self, epoch: int):
        cfg = self.cfg.training
        frac = min(1.0, epoch / max(cfg.tutorial_epsilon_epochs, 1))
        self.epsilon_override = (
            cfg.tutorial_epsilon_start
            + frac * (cfg.tutorial_epsilon_end - cfg.tutorial_epsilon_start)
        )

    def clear_epsilon_override(self):
        self.epsilon_override = None

    # ── Mask helpers ────────────────────────────────────────────────

    def _downsample_mask(self, mask: np.ndarray) -> np.ndarray:
        R, S, T = mask.shape
        S_down = S // self._s_stride
        trimmed = mask[:, :S_down * self._s_stride, :]
        reshaped = trimmed.reshape(R, S_down, self._s_stride, T)
        return reshaped.any(axis=2)

    def _find_start_in_window(
        self, row: int, s_down: int, tier: int,
        state: np.ndarray, channel: int = _CH_CONTAINER_START,
    ) -> int:
        s_start = s_down * self._s_stride
        s_end = min(s_start + self._s_stride, self.dims.n_splits)
        for s in range(s_start, s_end):
            if state[channel, row, s, tier] > 0.5:
                return s
        return s_start

    def _resolve_dest_split(
        self, row: int, s_down: int, tier: int, state: np.ndarray,
    ) -> int:
        region = self.dims.region_of(row)
        s_start = s_down * self._s_stride
        s_end = min(s_start + self._s_stride, self.dims.n_splits)
        if region == "YARD":
            for s in range(s_start, s_end):
                if state[_CH_OCCUPANCY, row, s, tier] < 0.5:
                    return s
            return s_start
        else:
            return self._find_start_in_window(row, s_down, tier, state)

    def _unflatten(self, flat_idx: int) -> Tuple[int, int, int]:
        T = self._T
        S_d = self._s_down
        tier = flat_idx % T
        s_down = (flat_idx // T) % S_d
        row = flat_idx // (S_d * T)
        return row, s_down, tier

    def _flatten_down(self, pos_down: Tuple[int, int, int]) -> int:
        row, s_d, tier = pos_down
        return (row * self._s_down + s_d) * self._T + tier

    def _batch_source_mask(self, states: torch.Tensor) -> torch.Tensor:
        starts = states[:, _CH_CONTAINER_START:_CH_CONTAINER_START + 1]
        stride = self._s_stride
        down = F.max_pool3d(starts, (1, stride, 1), stride=(1, stride, 1))
        return down.squeeze(1) > 0.5

    # ── Action selection ────────────────────────────────────────────

    @torch.no_grad()
    def act(
        self,
        state: np.ndarray,
        source_mask: np.ndarray,
        dest_mask_fn: DestMaskFn,
        epsilon: Optional[float] = None,
    ) -> UnifiedActionResult:
        self.step_count += 1
        eps = epsilon if epsilon is not None else self._get_epsilon()

        if not source_mask.any():
            return UnifiedActionResult()

        state_t = self._to_tensor(state).unsqueeze(0)
        self.q_net.eval()
        encoded = self.q_net.encode_state(state_t)

        # ── Stage 1: Source selection ───────────────────────────────
        src_down = self._downsample_mask(source_mask)
        if not src_down.any():
            return UnifiedActionResult()

        src_mask_t = self._to_bool_tensor(src_down).unsqueeze(0)
        q_src = self.q_net.q_source(encoded.feat_map, src_mask_t)[0]
        src_flat = self._select_source_action(q_src, src_down.flatten(), eps)

        src_row, src_s_down, src_tier = self._unflatten(src_flat)
        src_s_orig = self._find_start_in_window(src_row, src_s_down, src_tier, state)
        source_pos = (src_row, src_s_orig, src_tier)
        source_region = self.dims.region_of(src_row)

        pos_t = torch.tensor(
            [[src_row, src_s_down, src_tier]], dtype=torch.long, device=self.device,
        )
        source_feat = self.q_net.extract_source_feat(encoded.feat_map, pos_t)

        # ── Stage 2: Destination selection ──────────────────────────
        dest_mask_down = dest_mask_fn(src_row, src_s_orig, src_tier)
        if not dest_mask_down.any():
            return UnifiedActionResult(
                source_pos=source_pos,
                source_region=source_region,
                source_pos_down=(src_row, src_s_down, src_tier),
            )

        dst_mask_t = self._to_bool_tensor(dest_mask_down).unsqueeze(0)
        q_dst = self.q_net.q_dest(
            encoded.feat_map, encoded.global_feat, source_feat, dst_mask_t,
        )[0]
        dst_flat = self._select_dest_action(q_dst, dest_mask_down.flatten(), eps)

        dst_row, dst_s_down, dst_tier = self._unflatten(dst_flat)
        dst_s_orig = self._resolve_dest_split(dst_row, dst_s_down, dst_tier, state)
        dest_pos = (dst_row, dst_s_orig, dst_tier)
        dest_region = self.dims.region_of(dst_row)
        move_type = resolve_move_type(source_region, dest_region)

        return UnifiedActionResult(
            source_pos=source_pos,
            dest_pos=dest_pos,
            source_region=source_region,
            dest_region=dest_region,
            move_type=move_type,
            source_pos_down=(src_row, src_s_down, src_tier),
            dest_pos_down=(dst_row, dst_s_down, dst_tier),
        )

    # ── Replay ──────────────────────────────────────────────────────

    def remember(self, transition: UnifiedTransition):
        self.replay.push(transition)

    # ── Optimization ────────────────────────────────────────────────

    def optimize(self) -> float:
        if not self.replay.is_ready(self.cfg.training.batch_size):
            return 0.0

        self.q_net.train()

        if self.cfg.training.use_per:
            transitions, indices, weights = self.replay.sample(
                self.cfg.training.batch_size,
            )
            weights_t = self._to_tensor(weights)
        else:
            transitions = self.replay.sample(self.cfg.training.batch_size)
            weights_t = None
            indices = None

        loss, td_errors = self._compute_loss(transitions, weights_t)

        if self.cfg.training.use_per and indices is not None:
            self.replay.update_priorities(indices, td_errors.cpu().numpy())

        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.training.grad_clip > 0:
            nn.utils.clip_grad_norm_(
                self.q_net.parameters(), self.cfg.training.grad_clip,
            )
        self.optimizer.step()
        self._soft_update_target()
        self._post_optimize_hook()

        loss_val = loss.item()
        self.losses.append(loss_val)
        return loss_val

    # ── Standard TD loss (shared by most variants) ──────────────────

    def _standard_td_loss(
        self,
        transitions: List[UnifiedTransition],
        weights: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Double-DQN TD on source head + auxiliary reward prediction on dest."""
        gamma = self.cfg.training.gamma
        dest_aux_w = self.cfg.training.dest_aux_weight

        states = np.stack([t.state for t in transitions])
        next_states = np.stack([t.next_state for t in transitions])
        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        dones = np.array([t.done for t in transitions], dtype=np.float32)

        states_t = self._to_tensor(states)
        next_states_t = self._to_tensor(next_states)
        rewards_t = self._to_tensor(rewards)
        dones_t = self._to_tensor(dones)

        encoded = self.q_net.encode_state(states_t)
        src_mask_batch = self._batch_source_mask(states_t)
        q_src_all = self.q_net.q_source(encoded.feat_map, src_mask_batch)

        B = len(transitions)
        src_indices = torch.tensor(
            [self._flatten_down(t.source_pos_down) for t in transitions],
            dtype=torch.long, device=self.device,
        )
        # Source Q-values at taken positions
        q_source_taken = q_src_all[torch.arange(B, device=self.device), src_indices]

        with torch.no_grad():
            next_src_mask = self._batch_source_mask(next_states_t)
            if self.cfg.training.double_dqn:
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
            targets = rewards_t + gamma * (1 - dones_t) * max_q_next

        td_errors = targets - q_source_taken

        if weights is not None:
            src_loss = (
                weights * F.smooth_l1_loss(q_source_taken, targets, reduction="none")
            ).mean()
        else:
            src_loss = F.smooth_l1_loss(q_source_taken, targets)

        # Auxiliary dest loss: reward prediction
        total_loss = src_loss
        aux_preds = []
        aux_targets = []

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

    # ── Utilities ───────────────────────────────────────────────────

    def _to_tensor(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=torch.float32)
        return torch.as_tensor(
            np.asarray(x), dtype=torch.float32, device=self.device,
        )

    def _to_bool_tensor(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=torch.bool)
        return torch.as_tensor(
            np.asarray(x), dtype=torch.bool, device=self.device,
        )

    # ── Persistence ─────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "config": self.cfg,
        }, path)

    def load(self, path: str, map_location: str = None):
        ckpt = torch.load(path, map_location=map_location)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step_count = ckpt["step_count"]
