# multihead_dqn/agent.py
"""Multi-Head DQN Agent implementation."""
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from simulation.rl.multihead_dqn.config import MultiHeadDQNConfig, ActionType, DestinationType, YardDims
from simulation.rl.multihead_dqn.networks import MultiHeadQNetwork
from simulation.rl.multihead_dqn.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, Transition


@dataclass
class ActionResult:
    """Complete action decision from agent."""
    action_type: ActionType
    
    # Container move details (if MOVE_CONTAINER)
    container_pos: Optional[Tuple[int, int, int]] = None  # (row, split, tier)
    dest_type: Optional[DestinationType] = None
    placement_pos: Optional[Tuple[int, int, int]] = None  # if YARD
    vehicle_idx: int = -1  # if TRAIN/TRUCK
    
    # Parking details (if SLOT_PARKING)
    parking_idx: int = -1
    
    # For debugging
    q_values: Optional[Dict[str, float]] = None


class MultiHeadDQNAgent:
    """
    Multi-Head DQN Agent for container terminal operations.
    
    Hierarchical decision structure:
    1. Action type: MOVE_CONTAINER or SLOT_PARKING
    2. If MOVE: select container -> destination type -> specific dest
    3. If PARK: select parking action
    """
    
    def __init__(self, cfg: MultiHeadDQNConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.yard = cfg.yard
        
        # Networks
        self.q_net = MultiHeadQNetwork(
            cfg.yard, cfg.backbone, cfg.heads
        ).to(self.device)
        
        self.target_net = MultiHeadQNetwork(
            cfg.yard, cfg.backbone, cfg.heads
        ).to(self.device)
        self._hard_update_target()
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_net.parameters(),
            lr=cfg.training.lr
        )
        
        # Replay buffer
        if cfg.training.use_per:
            self.replay = PrioritizedReplayBuffer(
                cfg.training.replay_size,
                cfg.training.per_alpha,
                cfg.training.per_beta_start,
                cfg.training.per_beta_frames
            )
        else:
            self.replay = ReplayBuffer(cfg.training.replay_size)
        
        # Epsilon scheduling
        self.step_count = 0
        self.epsilon = cfg.training.epsilon_start
        
        # Training stats
        self.losses: List[float] = []
    
    def _hard_update_target(self):
        """Copy Q-network weights to target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())
    
    def _soft_update_target(self):
        """Soft update target network."""
        tau = self.cfg.training.target_tau
        for tp, sp in zip(self.target_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(tp.data * (1 - tau) + sp.data * tau)
    
    def _get_epsilon(self) -> float:
        """Get current epsilon with linear decay."""
        cfg = self.cfg.training
        progress = min(1.0, self.step_count / cfg.epsilon_decay_steps)
        return cfg.epsilon_start + progress * (cfg.epsilon_end - cfg.epsilon_start)
    
    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """Convert numpy array to tensor on device."""
        return torch.from_numpy(arr).float().to(self.device)
    
    def _to_bool_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """Convert boolean numpy array to tensor."""
        return torch.from_numpy(arr).bool().to(self.device)
    
    @torch.no_grad()
    def act(
        self,
        state: np.ndarray,
        occupancy_mask: np.ndarray,
        validity_mask: np.ndarray,
        vehicle_feats: Optional[np.ndarray] = None,
        vehicle_mask: Optional[np.ndarray] = None,
        parking_feats: Optional[np.ndarray] = None,
        parking_mask: Optional[np.ndarray] = None,
        epsilon: Optional[float] = None
    ) -> ActionResult:
        """
        Select action through hierarchical decision process.
        
        Args:
            state: (C, R, S, T) yard state tensor
            occupancy_mask: (R, S, T) bool - where containers exist
            validity_mask: (R, S, T) bool - where placement is valid
            vehicle_feats: (V, Fv) features for trains/trucks
            vehicle_mask: (V,) bool - which vehicles available
            parking_feats: (P, Fp) features for parking actions
            parking_mask: (P,) bool - which parking actions valid
            epsilon: Override epsilon value
            
        Returns:
            ActionResult with complete decision
        """
        self.step_count += 1
        eps = epsilon if epsilon is not None else self._get_epsilon()
        
        # Convert to tensors and add batch dim
        state_t = self._to_tensor(state).unsqueeze(0)  # (1, C, R, S, T)
        occ_t = self._to_bool_tensor(occupancy_mask).unsqueeze(0)  # (1, R, S, T)
        val_t = self._to_bool_tensor(validity_mask).unsqueeze(0)
        
        # Encode state
        self.q_net.eval()
        feat_map, global_feat = self.q_net.encode_state(state_t)
        
        # Check what actions are available
        has_containers = occ_t.any()
        has_parking = parking_mask is not None and parking_mask.any()
        
        if not has_containers and not has_parking:
            # No valid actions
            return ActionResult(action_type=ActionType.MOVE_CONTAINER)
        
        # Stage 1: Decide action type
        if random.random() < eps:
            # Random action type (weighted by availability)
            if has_containers and has_parking:
                action_type = random.choice([ActionType.MOVE_CONTAINER, ActionType.SLOT_PARKING])
            elif has_containers:
                action_type = ActionType.MOVE_CONTAINER
            else:
                action_type = ActionType.SLOT_PARKING
        else:
            # Greedy: compare best Q from each branch
            q_type = self.q_net.q_action_type(global_feat)[0]  # (2,)
            
            # Mask unavailable types
            if not has_containers:
                q_type[ActionType.MOVE_CONTAINER] = float('-inf')
            if not has_parking:
                q_type[ActionType.SLOT_PARKING] = float('-inf')
            
            action_type = ActionType(q_type.argmax().item())
        
        # Branch based on action type
        if action_type == ActionType.SLOT_PARKING:
            return self._select_parking(
                global_feat, parking_feats, parking_mask, eps
            )
        else:
            return self._select_container_move(
                feat_map, global_feat, occ_t, val_t,
                vehicle_feats, vehicle_mask, eps
            )
    
    def _select_parking(
        self,
        global_feat: torch.Tensor,
        parking_feats: np.ndarray,
        parking_mask: np.ndarray,
        eps: float
    ) -> ActionResult:
        """Select parking action."""
        park_t = self._to_tensor(parking_feats).unsqueeze(0)  # (1, P, Fp)
        mask_t = self._to_bool_tensor(parking_mask).unsqueeze(0)  # (1, P)
        
        if random.random() < eps:
            # Random valid parking
            valid_indices = np.where(parking_mask)[0]
            idx = random.choice(valid_indices)
        else:
            q_park = self.q_net.q_parking(global_feat, park_t, mask_t)[0]  # (P,)
            idx = q_park.argmax().item()
        
        return ActionResult(
            action_type=ActionType.SLOT_PARKING,
            parking_idx=idx
        )
    
    def _select_container_move(
        self,
        feat_map: torch.Tensor,
        global_feat: torch.Tensor,
        occ_t: torch.Tensor,
        val_t: torch.Tensor,
        vehicle_feats: Optional[np.ndarray],
        vehicle_mask: Optional[np.ndarray],
        eps: float
    ) -> ActionResult:
        """Select container and destination."""
        R, S, T = self.yard.n_rows, self.yard.n_splits, self.yard.n_tiers
        
        # Stage 2: Select container
        if random.random() < eps:
            # Random container
            valid_indices = torch.where(occ_t[0].flatten())[0]
            flat_idx = valid_indices[random.randrange(len(valid_indices))].item()
        else:
            q_cont = self.q_net.q_container_selection(feat_map, occ_t)[0]  # (R*S*T,)
            flat_idx = q_cont.argmax().item()
        
        # Convert flat index to (row, split, tier)
        container_pos = self._unflatten_index(flat_idx, R, S, T)
        pos_tensor = torch.tensor([container_pos], device=self.device)
        
        # Extract selected container features
        container_feat = self.q_net.extract_container_features(feat_map, pos_tensor)  # (1, F)
        
        # Stage 3: Select destination type
        has_vehicles = vehicle_mask is not None and len(vehicle_mask) > 0 and vehicle_mask.any()
        if random.random() < eps:
            choices = [DestinationType.YARD]
            if has_vehicles:
                choices.extend([DestinationType.TRAIN, DestinationType.TRUCK])
            dest_type = random.choice(choices)
        else:
            q_dest = self.q_net.q_dest_type(global_feat, container_feat)[0]  # (3,)

            if not has_vehicles:
                q_dest[DestinationType.TRAIN] = float('-inf')
                q_dest[DestinationType.TRUCK] = float('-inf')

            dest_type = DestinationType(q_dest.argmax().item())
        
        # Stage 4: Select specific destination
        if dest_type == DestinationType.YARD:
            return self._select_yard_placement(
                feat_map, global_feat, pos_tensor, val_t, container_pos, eps
            )
        else:
            return self._select_vehicle(
                global_feat, container_feat, container_pos, dest_type,
                vehicle_feats, vehicle_mask, eps
            )
    
    def _select_yard_placement(
        self,
        feat_map: torch.Tensor,
        global_feat: torch.Tensor,
        source_pos: torch.Tensor,
        val_t: torch.Tensor,
        container_pos: Tuple[int, int, int],
        eps: float
    ) -> ActionResult:
        """Select yard placement position."""
        R, S, T = self.yard.n_rows, self.yard.n_splits, self.yard.n_tiers
        
        if random.random() < eps:
            # Random valid placement
            valid_indices = torch.where(val_t[0].flatten())[0]
            if len(valid_indices) == 0:
                return ActionResult(
                    action_type=ActionType.MOVE_CONTAINER,
                    container_pos=container_pos,
                    dest_type=DestinationType.YARD
                )
            flat_idx = valid_indices[random.randrange(len(valid_indices))].item()
        else:
            q_place = self.q_net.q_placement(feat_map, global_feat, source_pos, val_t)[0]
            flat_idx = q_place.argmax().item()
        
        placement_pos = self._unflatten_index(flat_idx, R, S, T)
        
        return ActionResult(
            action_type=ActionType.MOVE_CONTAINER,
            container_pos=container_pos,
            dest_type=DestinationType.YARD,
            placement_pos=placement_pos
        )
    
    def _select_vehicle(
        self,
        global_feat: torch.Tensor,
        container_feat: torch.Tensor,
        container_pos: Tuple[int, int, int],
        dest_type: DestinationType,
        vehicle_feats: np.ndarray,
        vehicle_mask: np.ndarray,
        eps: float
    ) -> ActionResult:
        """Select train or truck."""
        # Guard: no vehicles available -> fall back to YARD with no placement
        if vehicle_feats.shape[0] == 0:
            return ActionResult(
                action_type=ActionType.MOVE_CONTAINER,
                container_pos=container_pos,
                dest_type=DestinationType.YARD,
            )
        veh_t = self._to_tensor(vehicle_feats).unsqueeze(0)  # (1, V, Fv)
        mask_t = self._to_bool_tensor(vehicle_mask).unsqueeze(0)  # (1, V)
        
        if random.random() < eps:
            valid_indices = np.where(vehicle_mask)[0]
            if len(valid_indices) == 0:
                idx = -1
            else:
                idx = random.choice(valid_indices)
        else:
            q_veh = self.q_net.q_vehicle(global_feat, container_feat, veh_t, mask_t)[0]
            idx = q_veh.argmax().item()
        
        return ActionResult(
            action_type=ActionType.MOVE_CONTAINER,
            container_pos=container_pos,
            dest_type=dest_type,
            vehicle_idx=idx
        )
    
    def _unflatten_index(self, flat_idx: int, R: int, S: int, T: int) -> Tuple[int, int, int]:
        """Convert flat index to (row, split, tier)."""
        tier = flat_idx % T
        split = (flat_idx // T) % S
        row = flat_idx // (S * T)
        return (row, split, tier)
    
    def _flatten_index(self, row: int, split: int, tier: int) -> int:
        """Convert (row, split, tier) to flat index."""
        S, T = self.yard.n_splits, self.yard.n_tiers
        return row * S * T + split * T + tier
    
    def remember(self, transition: Transition):
        """Store transition in replay buffer."""
        self.replay.push(transition)
    
    def optimize(self) -> float:
        """Perform one optimization step. Returns loss."""
        if not self.replay.is_ready(self.cfg.training.batch_size):
            return 0.0
        
        self.q_net.train()
        
        # Sample batch
        if self.cfg.training.use_per:
            transitions, indices, weights = self.replay.sample(self.cfg.training.batch_size)
            weights_t = self._to_tensor(weights)
        else:
            transitions = self.replay.sample(self.cfg.training.batch_size)
            weights_t = None
            indices = None
        
        # Compute loss
        loss, td_errors = self._compute_loss(transitions, weights_t)
        
        # Update priorities for PER
        if self.cfg.training.use_per and indices is not None:
            self.replay.update_priorities(indices, td_errors.cpu().numpy())
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.cfg.training.grad_clip > 0:
            nn.utils.clip_grad_norm_(
                self.q_net.parameters(),
                self.cfg.training.grad_clip
            )
        
        self.optimizer.step()
        
        # Soft update target
        self._soft_update_target()
        
        loss_val = loss.item()
        self.losses.append(loss_val)
        return loss_val
    
    def _compute_loss(
        self,
        transitions: List[Transition],
        weights: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute TD loss for batch of transitions."""
        batch_size = len(transitions)
        gamma = self.cfg.training.gamma
        
        # Batch states
        states = np.stack([t.state for t in transitions])  # (B, C, R, S, T)
        next_states = np.stack([t.next_state for t in transitions])
        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        dones = np.array([t.done for t in transitions], dtype=np.float32)
        
        states_t = self._to_tensor(states)
        next_states_t = self._to_tensor(next_states)
        rewards_t = self._to_tensor(rewards)
        dones_t = self._to_tensor(dones)
        
        # Encode states
        feat_map, global_feat = self.q_net.encode_state(states_t)
        
        # Compute Q-values for taken actions (sum across stages)
        q_values = []
        
        for i, t in enumerate(transitions):
            q_total = torch.tensor(0.0, device=self.device)
            
            # Action type Q-value
            q_type = self.q_net.q_action_type(global_feat[i:i+1])[0]
            q_total = q_total + q_type[t.action_type]
            
            if t.action_type == ActionType.MOVE_CONTAINER and t.container_pos is not None:
                # Container selection Q-value
                flat_idx = self._flatten_index(*t.container_pos)
                # Note: Would need occupancy mask reconstruction for proper Q
                # For simplicity, using feat-based scoring
                
                # Destination type Q-value
                if t.dest_type >= 0:
                    pos_t = torch.tensor([t.container_pos], device=self.device)
                    cont_feat = self.q_net.extract_container_features(
                        feat_map[i:i+1], pos_t
                    )
                    q_dest = self.q_net.q_dest_type(global_feat[i:i+1], cont_feat)[0]
                    q_total = q_total + q_dest[t.dest_type]
            
            q_values.append(q_total)
        
        q_values_t = torch.stack(q_values)
        
        # Compute target Q-values (simplified: use reward + gamma * 0 for terminal)
        with torch.no_grad():
            # For non-terminal states, estimate max future Q
            # This is simplified - full implementation would need next action masks
            next_feat_map, next_global = self.target_net.encode_state(next_states_t)
            q_next_type = self.target_net.q_action_type(next_global)
            max_q_next = q_next_type.max(dim=1)[0]
            
            targets = rewards_t + gamma * (1 - dones_t) * max_q_next
        
        # TD error
        td_errors = targets - q_values_t
        
        # Loss (Huber)
        if weights is not None:
            loss = (weights * F.smooth_l1_loss(q_values_t, targets, reduction='none')).mean()
        else:
            loss = F.smooth_l1_loss(q_values_t, targets)
        
        return loss, td_errors.detach()
    
    def save(self, path: str):
        """Save agent state."""
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'config': self.cfg
        }, path)
    
    def load(self, path: str, map_location: str = None):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=map_location)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step_count = checkpoint['step_count']