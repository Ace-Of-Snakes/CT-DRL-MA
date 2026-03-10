# simulation/rl/variants/gru_agent.py
"""GRU DQN — temporal context via recurrent global features.

Adds a GRU cell between OccupiedPooling and the action heads, providing
the agent with memory of previous decisions within an episode.  This
targets the multi-step planning failures observed in S9/S10 (reshuffle
loops) and S18 (concurrent import/export), where the memoryless baseline
agent cannot maintain plan persistence across steps.

Architecture (inference-only recurrence — Phase 1):
  - CNN backbone + OccupiedPooling produce global_feat (B, 128) as usual
  - GRUCell(128 → 64) + Linear(64 → 128) enrich global_feat with temporal
    context from the hidden state h_{t-1}
  - Hidden state h_t persists across act() calls within an episode
  - At episode/scenario boundaries, reset_hidden() zeros the state
  - During training, hidden state is None → pure feedforward (standard TD)

Uses standard baseline TD loss.  No NoisyNet, no Munchausen, no Dueling.
This isolates the GRU contribution from other algorithmic improvements,
enabling clean comparison in the thesis ablation grid.
"""
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn

from simulation.rl.base_agent import (
    BaseSpatialDQNAgent, ActionResult, DestMaskFn, resolve_move_type,
)
from simulation.rl.multihead_dqn.networks import RecurrentQNetwork
from simulation.rl.multihead_dqn.replay_buffer import Transition


class GRUDQNAgent(BaseSpatialDQNAgent):
    """DQN with GRU temporal context on global features.

    Inference-only recurrence: GRU enriches global_feat during act()
    but training uses standard TD loss with hidden=None (feedforward).
    Combinable with any CNN backbone via the standard agent registry.
    """

    def _build_networks(self) -> Tuple[nn.Module, nn.Module]:
        def _make():
            return RecurrentQNetwork(
                self.cfg.unified,
                self.cfg.cnn,
                self.cfg.heads,
                gru_hidden=64,
            )
        return _make(), _make()

    # ── Hidden state management ────────────────────────────────────────

    def reset_hidden(self) -> None:
        """Reset GRU hidden state — call at episode/scenario boundaries."""
        self.q_net.reset_hidden()
        self.target_net.reset_hidden()

    def _ensure_hidden(self) -> None:
        """Lazily initialize hidden state on first act() call."""
        if self.q_net._h is None:
            self.q_net.init_hidden(self.device)

    # ── Action selection (override to add GRU plumbing) ────────────────

    @torch.no_grad()
    def act(
        self,
        state: np.ndarray,
        source_mask: np.ndarray,
        dest_mask_fn: DestMaskFn,
        epsilon: Optional[float] = None,
    ) -> ActionResult:
        self.step_count += 1
        eps = epsilon if epsilon is not None else self._get_epsilon()
        self._pre_act_hook()

        if not source_mask.any():
            return ActionResult()

        # ── GRU: ensure hidden state is initialized ──────────────────
        self._ensure_hidden()

        state_t = self._to_tensor(state).unsqueeze(0)
        self.q_net.eval()
        encoded = self.q_net.encode_state(state_t)
        # encoded.global_feat is now GRU-augmented (temporal context)

        # ── Stage 1: Source selection ────────────────────────────────
        src_down = self._downsample_mask(source_mask)
        if not src_down.any():
            return ActionResult()

        src_mask_t = self._to_bool_tensor(src_down).unsqueeze(0)
        q_src = self.q_net.q_source(encoded.feat_map, src_mask_t)[0]
        valid_with_idle = np.append(src_down.flatten(), True)
        src_flat = self._select_source_action(q_src, valid_with_idle, eps)

        if src_flat == self._idle_source_index:
            return ActionResult(move_type="IDLE")

        src_row, src_s_down, src_tier = self._unflatten(src_flat)
        src_s_orig = self._find_start_in_window(
            src_row, src_s_down, src_tier, state,
        )
        source_pos = (src_row, src_s_orig, src_tier)
        source_region = self.dims.region_of(src_row)

        pos_t = torch.tensor(
            [[src_row, src_s_down, src_tier]],
            dtype=torch.long, device=self.device,
        )
        source_feat = self.q_net.extract_source_feat(encoded.feat_map, pos_t)

        # ── Stage 2: Destination selection ───────────────────────────
        dest_mask_down, src_n_splits = dest_mask_fn(
            src_row, src_s_orig, src_tier,
        )
        if not dest_mask_down.any():
            return ActionResult(
                source_pos=source_pos,
                source_region=source_region,
                source_pos_down=(src_row, src_s_down, src_tier),
            )

        dst_mask_t = self._to_bool_tensor(dest_mask_down).unsqueeze(0)
        q_dst = self.q_net.q_dest(
            encoded.feat_map, encoded.global_feat, source_feat, dst_mask_t,
        )[0]
        dst_flat = self._select_dest_action(
            q_dst, dest_mask_down.flatten(), eps,
        )

        dst_row, dst_s_down, dst_tier = self._unflatten(dst_flat)
        dst_s_orig = self._resolve_dest_split(
            dst_row, dst_s_down, dst_tier, state,
            n_splits=src_n_splits, source_region=source_region,
        )
        dest_pos = (dst_row, dst_s_orig, dst_tier)
        dest_region = self.dims.region_of(dst_row)
        move_type = resolve_move_type(source_region, dest_region)

        return ActionResult(
            source_pos=source_pos,
            dest_pos=dest_pos,
            source_region=source_region,
            dest_region=dest_region,
            move_type=move_type,
            source_pos_down=(src_row, src_s_down, src_tier),
            dest_pos_down=(dst_row, dst_s_down, dst_tier),
        )

    # ── Training ───────────────────────────────────────────────────────

    def _compute_loss(
        self, transitions: List[Transition], weights: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard TD loss — GRU hidden is None during training.

        Transitions are sampled i.i.d. from the replay buffer (not
        sequential), so the GRU cannot provide meaningful temporal
        context.  We explicitly clear hidden state so encode_state()
        falls back to the pure feedforward path.
        """
        self.q_net.reset_hidden()
        self.target_net.reset_hidden()
        return self._standard_td_loss(transitions, weights)
