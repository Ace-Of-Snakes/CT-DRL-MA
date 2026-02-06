# multihead_dqn/agent.py
"""Factored-CNN Multi-Head DQN Agent with Double DQN and tutorial epsilon."""
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from simulation.rl.multihead_dqn.config import (
    MultiHeadDQNConfig, ActionType, DestinationType, YardDims,
)
from simulation.rl.multihead_dqn.networks import FactoredCNNQNetwork, EncodedState
from simulation.rl.multihead_dqn.replay_buffer import (
    ReplayBuffer, PrioritizedReplayBuffer, Transition,
)

# State tensor channel indices (must match state_encoder.ChannelSpec)
_CH_CONTAINER_START = 1
_CH_DIRECTION = 10             # 0.0 = Import, 1.0 = Export

# Direction-based destination masking threshold
_DIRECTION_EXPORT_THRESHOLD = 0.5

# Vehicle feature indices (must match env._build_vehicle_features)
_VEH_FEAT_IS_TRAIN = 0        # 1.0 = train, 0.0 = truck


@dataclass
class ActionResult:
    """Complete action decision from agent."""
    action_type: ActionType
    container_pos: Optional[Tuple[int, int, int]] = None   # (row, split, tier)
    dest_type: Optional[DestinationType] = None
    placement_pos: Optional[Tuple[int, int, int]] = None    # (row, split, tier)
    vehicle_idx: int = -1
    parking_idx: int = -1
    q_values: Optional[Dict[str, float]] = None


class MultiHeadDQNAgent:
    """Factored-CNN Multi-Head DQN Agent.

    Architecture:
      (1,21,1) + (3,1,3) + (1,5,1) factored CNN backbone
      -> occupied-only pooling -> global_feat
      -> spatial container selection via 1x1 conv on feat_map
      -> per-container embedding via feat_map indexing
      -> factorized placement at bay resolution
    """

    def __init__(self, cfg: MultiHeadDQNConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.yard = cfg.yard
        self._s_stride = cfg.cnn.s_stride
        self._s_down = cfg.s_down
        self._n_rows = cfg.yard.n_rows            # yard-only rows (5)
        self._n_state_rows = cfg.yard.n_state_rows  # yard + parking (6)

        # Networks
        self.q_net = FactoredCNNQNetwork(cfg.yard, cfg.cnn, cfg.heads).to(self.device)
        self.target_net = FactoredCNNQNetwork(cfg.yard, cfg.cnn, cfg.heads).to(self.device)
        self._hard_update_target()

        # Proximity placement
        self._proximity_bays = cfg.heads.proximity_bays
        self._window_splits = (2 * self._proximity_bays + 1) * cfg.yard.split_factor

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.training.lr)

        # Replay buffer
        if cfg.training.use_per:
            self.replay = PrioritizedReplayBuffer(
                cfg.training.replay_size, cfg.training.per_alpha,
            )
        else:
            self.replay = ReplayBuffer(cfg.training.replay_size)

        self.step_count = 0
        self.losses: List[float] = []
        self.epsilon_override: Optional[float] = None

    # ----------------------------------------------------------------
    # Target network
    # ----------------------------------------------------------------

    def _hard_update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def _soft_update_target(self):
        tau = self.cfg.training.target_tau
        for tp, op in zip(self.target_net.parameters(), self.q_net.parameters()):
            tp.data.mul_(1 - tau).add_(op.data, alpha=tau)

    # ----------------------------------------------------------------
    # Epsilon
    # ----------------------------------------------------------------

    def _get_epsilon(self) -> float:
        if self.epsilon_override is not None:
            return self.epsilon_override
        cfg = self.cfg.training
        frac = min(1.0, self.step_count / max(cfg.epsilon_decay_steps, 1))
        return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)

    def set_tutorial_epsilon(self, epoch: int):
        """Set fast-decaying epsilon for tutorial phase."""
        cfg = self.cfg.training
        frac = min(1.0, epoch / max(cfg.tutorial_epsilon_epochs, 1))
        self.epsilon_override = (
            cfg.tutorial_epsilon_start
            + frac * (cfg.tutorial_epsilon_end - cfg.tutorial_epsilon_start)
        )

    def clear_epsilon_override(self):
        self.epsilon_override = None

    # ----------------------------------------------------------------
    # Mask helpers
    # ----------------------------------------------------------------

    def _downsample_container_mask(self, state: np.ndarray) -> np.ndarray:
        """CONTAINER_START mask at downsampled resolution.

        Max-pools the CONTAINER_START channel along S by s_stride.
        Result has exactly one True per container (no multi-split
        selection ambiguity).

        Args:
            state: (C, R, S, T) numpy
        Returns:
            (R, S_down, T) bool
        """
        starts = state[_CH_CONTAINER_START] > 0.5       # (R, S, T)
        R, S, T = starts.shape
        S_down = S // self._s_stride
        reshaped = starts[:, :S_down * self._s_stride, :].reshape(
            R, S_down, self._s_stride, T,
        )
        return reshaped.any(axis=2)

    def _find_container_start(self, r: int, s_down: int, t: int,
                              state: np.ndarray) -> int:
        """Map downsampled position -> original CONTAINER_START split.

        Searches the stride-window [s_down*stride, (s_down+1)*stride)
        for the actual container start marker.
        """
        s_start = s_down * self._s_stride
        s_end = min(s_start + self._s_stride, self.yard.n_splits)
        for s in range(s_start, s_end):
            if state[_CH_CONTAINER_START, r, s, t] > 0.5:
                return s
        return s_start  # fallback

    def _unflatten_down(self, flat_idx: int) -> Tuple[int, int, int]:
        """Unflatten index from (R_state, S_down, T) layout -> (row, s_down, tier)."""
        T = self.yard.n_tiers
        S_d = self._s_down
        tier = flat_idx % T
        s_down = (flat_idx // T) % S_d
        row = flat_idx // (S_d * T)
        return row, s_down, tier

    # ----------------------------------------------------------------
    # Action selection
    # ----------------------------------------------------------------

    @torch.no_grad()
    def act(
        self,
        state: np.ndarray,
        occupancy_mask: np.ndarray,
        validity_mask: np.ndarray,
        vehicle_feats: Optional[np.ndarray] = None,
        vehicle_mask: Optional[np.ndarray] = None,
        import_mask: Optional[np.ndarray] = None,
        parking_feats: Optional[np.ndarray] = None,
        parking_mask: Optional[np.ndarray] = None,
        epsilon: Optional[float] = None,
    ) -> ActionResult:
        """Hierarchical action selection with ÃƒÅ½Ã‚Âµ-greedy exploration."""
        self.step_count += 1
        eps = epsilon if epsilon is not None else self._get_epsilon()

        # Encode state through CNN backbone
        state_t = self._to_tensor(state).unsqueeze(0)       # (1, C, R, S, T)
        self.q_net.eval()
        encoded = self.q_net.encode_state(state_t)

        has_containers = occupancy_mask.any()
        has_parking = parking_mask is not None and parking_mask.any()
        has_imports = import_mask is not None and import_mask.any()

        if not has_containers and not has_parking and not has_imports:
            return ActionResult(action_type=ActionType.MOVE_CONTAINER)

        # Stage 1: Action type
        if random.random() < eps:
            choices = []
            if has_containers:
                choices.append(ActionType.MOVE_CONTAINER)
            if has_parking:
                choices.append(ActionType.SLOT_PARKING)
            if has_imports:
                choices.append(ActionType.IMPORT_VEHICLE)
            action_type = random.choice(choices) if choices else ActionType.MOVE_CONTAINER
        else:
            q_type = self.q_net.q_action_type(encoded.global_feat)[0]
            if not has_containers:
                q_type[ActionType.MOVE_CONTAINER] = float("-inf")
            if not has_parking:
                q_type[ActionType.SLOT_PARKING] = float("-inf")
            if not has_imports:
                q_type[ActionType.IMPORT_VEHICLE] = float("-inf")
            action_type = ActionType(q_type.argmax().item())

        if action_type == ActionType.SLOT_PARKING:
            return self._select_parking(encoded, parking_feats, parking_mask, eps)
        elif action_type == ActionType.IMPORT_VEHICLE:
            return self._select_import(encoded, vehicle_feats, import_mask, eps)
        else:
            return self._select_container_move(
                encoded, state, validity_mask, vehicle_feats, vehicle_mask, eps,
            )

    def _select_parking(self, encoded, parking_feats, parking_mask, eps):
        park_t = self._to_tensor(parking_feats).unsqueeze(0)
        mask_t = self._to_bool_tensor(parking_mask).unsqueeze(0)
        if random.random() < eps:
            valid = np.where(parking_mask)[0]
            idx = random.choice(valid)
        else:
            q = self.q_net.q_parking(encoded.global_feat, park_t, mask_t)[0]
            idx = q.argmax().item()
        return ActionResult(action_type=ActionType.SLOT_PARKING, parking_idx=idx)

    def _select_import(self, encoded, vehicle_feats, import_mask, eps):
        if vehicle_feats.shape[0] == 0:
            return ActionResult(action_type=ActionType.IMPORT_VEHICLE)
        veh_t = self._to_tensor(vehicle_feats).unsqueeze(0)
        mask_t = self._to_bool_tensor(import_mask).unsqueeze(0)
        # Dummy container feat (not applicable for imports)
        Fc = self.cfg.cnn.feat_channels
        dummy = torch.zeros(1, Fc, device=self.device)

        if random.random() < eps:
            valid = np.where(import_mask)[0]
            idx = random.choice(valid) if len(valid) > 0 else -1
        else:
            q = self.q_net.q_vehicle(encoded.global_feat, dummy, veh_t, mask_t)[0]
            idx = q.argmax().item()
        return ActionResult(action_type=ActionType.IMPORT_VEHICLE, vehicle_idx=idx)

    def _select_container_move(self, encoded, state, validity_mask,
                               vehicle_feats, vehicle_mask, eps):
        """Select container -> destination -> placement/vehicle."""
        # Build downsampled container-start mask
        cstart_down = self._downsample_container_mask(state)  # (R, S_down, T)
        cstart_t = self._to_bool_tensor(cstart_down).unsqueeze(0)  # (1, R, S_down, T)

        if not cstart_down.any():
            return ActionResult(action_type=ActionType.MOVE_CONTAINER)

        # Stage 2: Select container
        if random.random() < eps:
            valid_flat = np.where(cstart_down.flatten())[0]
            flat_idx = random.choice(valid_flat)
        else:
            q = self.q_net.q_container_selection(encoded.feat_map, cstart_t)[0]
            flat_idx = q.argmax().item()

        row, s_down, tier = self._unflatten_down(flat_idx)
        s_orig = self._find_container_start(row, s_down, tier, state)
        container_pos = (row, s_orig, tier)

        # Extract container feature from feat_map (64-dim CNN embedding)
        pos_t = torch.tensor([[row, s_down, tier]], dtype=torch.long, device=self.device)
        container_feat = self.q_net.extract_container_feat(encoded.feat_map, pos_t)

        # Stage 3: Destination type with direction masking
        # Import (ch10 Ã¢â€°Ë† 0) Ã¢â€ â€™ leaves on TRUCK; Export (ch10 Ã¢â€°Ë† 1) Ã¢â€ â€™ leaves on TRAIN
        has_vehicles = vehicle_mask is not None and len(vehicle_mask) > 0 and vehicle_mask.any()
        is_import = state[_CH_DIRECTION, row, s_orig, tier] < _DIRECTION_EXPORT_THRESHOLD
        can_train = has_vehicles and not is_import
        can_truck = has_vehicles and is_import

        # Separate higher epsilon floor for dest_type prevents catastrophic
        # forgetting of rare vehicle destinations in the aux-trained head.
        dest_eps = max(eps, self.cfg.training.dest_epsilon_floor)
        if random.random() < dest_eps:
            choices = [DestinationType.YARD]
            if can_train:
                choices.append(DestinationType.TRAIN)
            if can_truck:
                choices.append(DestinationType.TRUCK)
            dest_type = random.choice(choices)
        else:
            q = self.q_net.q_dest_type(encoded.global_feat, container_feat)[0]
            if not can_train:
                q[DestinationType.TRAIN] = float("-inf")
            if not can_truck:
                q[DestinationType.TRUCK] = float("-inf")
            dest_type = DestinationType(q.argmax().item())

        # Stage 4
        if dest_type == DestinationType.YARD:
            return self._select_yard_placement(
                encoded, container_feat, container_pos, validity_mask, eps,
            )
        else:
            return self._select_vehicle(
                encoded, container_feat, container_pos, dest_type,
                vehicle_feats, vehicle_mask, eps,
            )

    def _select_yard_placement(self, encoded, container_feat, container_pos,
                               validity_mask, eps):
        """Select placement at split resolution within proximity window.

        Validity mask has R_state rows but placement head uses n_rows (yard only).
        """
        R = self._n_rows  # yard-only rows (5)
        T = self.yard.n_tiers
        W = self._window_splits

        # Reference bay = container's current bay
        ref_bay = container_pos[1] // self.yard.split_factor
        start_split = self._window_start_split(ref_bay)

        # Slice to yard rows, then extract proximity window: (R, W, T)
        yard_validity = validity_mask[:R]
        window_mask = yard_validity[:, start_split:start_split + W, :]
        window_t = self._to_bool_tensor(window_mask).unsqueeze(0)  # (1, R, W, T)

        if random.random() < eps:
            valid_flat = np.where(window_mask.flatten())[0]
            if len(valid_flat) == 0:
                return ActionResult(
                    action_type=ActionType.MOVE_CONTAINER,
                    container_pos=container_pos,
                    dest_type=DestinationType.YARD,
                )
            flat_idx = random.choice(valid_flat)
        else:
            q = self.q_net.q_placement(
                encoded.global_feat, container_feat, window_t,
            )[0]
            flat_idx = q.argmax().item()

        # Unflatten (R, W, T) -> (row, window_split, tier)
        tier = flat_idx % T
        ws = (flat_idx // T) % W
        row = flat_idx // (W * T)

        # Map to absolute split coordinate
        abs_split = start_split + ws

        return ActionResult(
            action_type=ActionType.MOVE_CONTAINER,
            container_pos=container_pos,
            dest_type=DestinationType.YARD,
            placement_pos=(row, abs_split, tier),
        )

    def _window_start_split(self, ref_bay: int) -> int:
        """Compute proximity window start split, clamped to yard bounds."""
        P = self._proximity_bays
        center = max(P, min(self.yard.n_bays - 1 - P, ref_bay))
        return (center - P) * self.yard.split_factor

    def _select_vehicle(self, encoded, container_feat, container_pos,
                        dest_type, vehicle_feats, vehicle_mask, eps):
        """Select target vehicle, filtered by type matching dest_type."""
        if vehicle_feats.shape[0] == 0:
            return ActionResult(
                action_type=ActionType.MOVE_CONTAINER,
                container_pos=container_pos,
                dest_type=DestinationType.YARD,
            )

        # Filter mask by vehicle type: feat[0] = is_train (1.0=train, 0.0=truck)
        is_train = vehicle_feats[:, _VEH_FEAT_IS_TRAIN] > 0.5
        if dest_type == DestinationType.TRAIN:
            type_mask = vehicle_mask & is_train
        elif dest_type == DestinationType.TRUCK:
            type_mask = vehicle_mask & ~is_train
        else:
            type_mask = vehicle_mask

        if not type_mask.any():
            return ActionResult(
                action_type=ActionType.MOVE_CONTAINER,
                container_pos=container_pos,
                dest_type=DestinationType.YARD,
            )

        veh_t = self._to_tensor(vehicle_feats).unsqueeze(0)
        mask_t = self._to_bool_tensor(type_mask).unsqueeze(0)
        if random.random() < eps:
            valid = np.where(type_mask)[0]
            idx = random.choice(valid) if len(valid) > 0 else -1
        else:
            q = self.q_net.q_vehicle(encoded.global_feat, container_feat, veh_t, mask_t)[0]
            idx = q.argmax().item()
        return ActionResult(
            action_type=ActionType.MOVE_CONTAINER,
            container_pos=container_pos,
            dest_type=dest_type,
            vehicle_idx=idx,
        )

    # ----------------------------------------------------------------
    # Replay
    # ----------------------------------------------------------------

    def remember(self, transition: Transition):
        self.replay.push(transition)

    # ----------------------------------------------------------------
    # Optimization
    # ----------------------------------------------------------------

    def optimize(self) -> float:
        """One optimization step with Double DQN."""
        if not self.replay.is_ready(self.cfg.training.batch_size):
            return 0.0

        self.q_net.train()

        if self.cfg.training.use_per:
            transitions, indices, weights = self.replay.sample(self.cfg.training.batch_size)
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
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg.training.grad_clip)
        self.optimizer.step()
        self._soft_update_target()

        loss_val = loss.item()
        self.losses.append(loss_val)
        return loss_val

    def _compute_loss(self, transitions: List[Transition],
                      weights: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """TD loss on action_type + auxiliary reward-prediction for dest_type.

        The main TD target bootstraps only from action_type Q-values,
        ensuring consistency (target and current use the same heads).
        The dest_type head is trained separately as a reward predictor
        so it learns to discriminate TRUCK vs YARD from immediate rewards.
        """
        gamma = self.cfg.training.gamma
        dest_aux_weight = self.cfg.training.dest_aux_weight

        states = np.stack([t.state for t in transitions])
        next_states = np.stack([t.next_state for t in transitions])
        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        dones = np.array([t.done for t in transitions], dtype=np.float32)

        states_t = self._to_tensor(states)
        next_states_t = self._to_tensor(next_states)
        rewards_t = self._to_tensor(rewards)
        dones_t = self._to_tensor(dones)

        # Encode full batch through backbone (single forward pass)
        encoded = self.q_net.encode_state(states_t)

        # -- Main Q: action_type only (consistent with target) --
        q_values = []
        # -- Auxiliary: collect dest_type predictions + rewards --
        aux_preds = []
        aux_targets = []

        for i, t in enumerate(transitions):
            # Stage 1: action type Q-value (this is the TD-trained head)
            q_type = self.q_net.q_action_type(encoded.global_feat[i:i + 1])[0]
            q_values.append(q_type[t.action_type])

            # Auxiliary dest_type: train as reward predictor for MOVE transitions
            if (t.action_type == ActionType.MOVE_CONTAINER
                    and t.container_pos is not None
                    and t.dest_type is not None
                    and t.dest_type >= 0):
                r, s_orig, tier = t.container_pos
                s_down = s_orig // self._s_stride
                pos_down = torch.tensor(
                    [[r, s_down, tier]], dtype=torch.long, device=self.device,
                )
                cont_feat = self.q_net.extract_container_feat(
                    encoded.feat_map[i:i + 1], pos_down,
                )
                q_dest = self.q_net.q_dest_type(
                    encoded.global_feat[i:i + 1], cont_feat,
                )[0]
                aux_preds.append(q_dest[t.dest_type])
                aux_targets.append(t.reward)

        q_values_t = torch.stack(q_values)

        # -- Target Q (Double DQN, action_type only) --
        with torch.no_grad():
            if self.cfg.training.double_dqn:
                online_enc = self.q_net.encode_state(next_states_t)
                q_online = self.q_net.q_action_type(online_enc.global_feat)
                best_actions = q_online.argmax(dim=1, keepdim=True)

                target_enc = self.target_net.encode_state(next_states_t)
                q_target = self.target_net.q_action_type(target_enc.global_feat)
                max_q_next = q_target.gather(1, best_actions).squeeze(1)
            else:
                target_enc = self.target_net.encode_state(next_states_t)
                q_target = self.target_net.q_action_type(target_enc.global_feat)
                max_q_next = q_target.max(dim=1)[0]

            targets = rewards_t + gamma * (1 - dones_t) * max_q_next

        td_errors = targets - q_values_t

        if weights is not None:
            td_loss = (weights * F.smooth_l1_loss(q_values_t, targets, reduction="none")).mean()
        else:
            td_loss = F.smooth_l1_loss(q_values_t, targets)

        # -- Auxiliary dest_type loss: reward prediction --
        total_loss = td_loss
        if aux_preds:
            aux_preds_t = torch.stack(aux_preds)
            aux_targets_t = torch.tensor(aux_targets, dtype=torch.float32, device=self.device)
            aux_loss = F.smooth_l1_loss(aux_preds_t, aux_targets_t)
            total_loss = td_loss + dest_aux_weight * aux_loss

        return total_loss, td_errors.detach()

    # ----------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------

    def _to_tensor(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=torch.float32)
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)

    def _to_bool_tensor(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=torch.bool)
        return torch.as_tensor(np.asarray(x), dtype=torch.bool, device=self.device)

    # ----------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------

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