# simulation/rl/agents/hierarchical_dqn_agent.py
"""Two-stage hierarchical DQN agent for container terminal operations."""
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from simulation.config.curriculum_config import HierarchicalDQNConfig
from simulation.rl.policy.hierarchical_networks import HierarchicalQNetwork
from simulation.rl.features.featurizers import (
    MoveableContainer, Destination, ParkingAction,
    ContainerFeaturizer, DestinationFeaturizer, ParkingFeaturizer,
    SourceType, DestinationType
)


@dataclass
class HierarchicalTransition:
    """Complete transition for replay buffer."""
    state: np.ndarray                    # [R, B, T, C]
    container_feats: np.ndarray          # [K_cont, 16]
    container_idx: int                   # Selected container index
    container_feat: np.ndarray           # [16] - the selected container's features
    destination_feats: np.ndarray        # [K_dest, 12]
    destination_idx: int                 # Selected destination index
    reward: float
    next_state: np.ndarray
    done: bool
    # Metadata for debugging
    was_parking: bool = False
    parking_feats: Optional[np.ndarray] = None
    parking_idx: int = -1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class HierarchicalReplayBuffer:
    """Replay buffer for hierarchical transitions."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[HierarchicalTransition] = []
        self.ptr = 0
    
    def push(self, transition: HierarchicalTransition):
        """Add transition to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.ptr] = transition
        self.ptr = (self.ptr + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[HierarchicalTransition]:
        """Sample random batch."""
        n = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, n)
    
    def __len__(self) -> int:
        return len(self.buffer)


@dataclass
class ActionPool:
    """Pool of available actions at current timestep."""
    containers: List[MoveableContainer]
    parkings: List[ParkingAction]
    
    # Featurized versions (computed lazily)
    container_feats: Optional[torch.Tensor] = None
    parking_feats: Optional[torch.Tensor] = None
    
    def is_empty(self) -> bool:
        return len(self.containers) == 0 and len(self.parkings) == 0
    
    def total_size(self) -> int:
        return len(self.containers) + len(self.parkings)


@dataclass
class Stage1Selection:
    """Result of Stage 1 selection."""
    is_parking: bool
    index: int  # Index in containers or parkings list
    
    # If container
    container: Optional[MoveableContainer] = None
    container_feat: Optional[np.ndarray] = None
    
    # If parking
    parking: Optional[ParkingAction] = None


class HierarchicalDQNAgent:
    """
    Two-stage hierarchical DQN agent.
    
    Stage 1: Select container or parking action
    Stage 2: Select destination (only if container selected)
    """
    
    def __init__(
        self,
        yard_dims: Tuple[int, int, int, int],
        cfg: HierarchicalDQNConfig = None
    ):
        """
        Initialize agent.
        
        Args:
            yard_dims: (n_rows, n_bays, n_tiers, split_factor)
            cfg: Configuration object
        """
        self.R, self.B, self.T, self.SF = yard_dims
        self.cfg = cfg or HierarchicalDQNConfig()
        
        # Networks
        self.q_net = HierarchicalQNetwork(self.cfg).to(self.cfg.device)
        self.target_net = HierarchicalQNetwork(self.cfg).to(self.cfg.device)
        self._hard_update_target()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.cfg.lr)
        
        # Replay buffer
        self.replay = HierarchicalReplayBuffer(self.cfg.replay_size)
        
        # Featurizers
        self.cont_featurizer = ContainerFeaturizer(self.R, self.B, self.T, self.SF)
        self.dest_featurizer = DestinationFeaturizer(self.R, self.B, self.T, self.SF)
        self.park_featurizer = ParkingFeaturizer(self.B)
        
        # Epsilon scheduling
        self.step_count = 0
        self.epsilon = self.cfg.epsilon_start
        
        # Cached state embedding (computed once per timestep)
        self._cached_state_emb: Optional[torch.Tensor] = None
        self._cached_state_np: Optional[np.ndarray] = None
    
    def _hard_update_target(self):
        """Copy weights from Q-network to target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())
    
    def _soft_update_target(self):
        """Soft update target network."""
        tau = self.cfg.target_tau
        for tp, sp in zip(self.target_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(tp.data * (1 - tau) + sp.data * tau)
    
    def _get_epsilon(self) -> float:
        """Get current epsilon value with linear decay."""
        progress = min(1.0, self.step_count / self.cfg.epsilon_decay_steps)
        return self.cfg.epsilon_end + (self.cfg.epsilon_start - self.cfg.epsilon_end) * (1 - progress)
    
    def reset_epsilon(self):
        """Reset epsilon schedule (for curriculum stage transitions)."""
        self.step_count = 0
    
    def _to_tensor_state(self, state_np: np.ndarray) -> torch.Tensor:
        """Convert state array to tensor [1, C, R, B, T]."""
        # state_np: [R, B, T, C] -> [1, C, R, B, T]
        x = torch.from_numpy(np.transpose(state_np, (3, 0, 1, 2))).float().unsqueeze(0)
        return x.to(self.cfg.device)
    
    def encode_state(self, state_np: np.ndarray, force: bool = False) -> torch.Tensor:
        """
        Encode state, using cache if same state.
        
        Args:
            state_np: State array [R, B, T, C]
            force: Force recomputation even if cached
            
        Returns:
            State embedding [1, H]
        """
        # Check cache
        if not force and self._cached_state_np is not None:
            if np.array_equal(state_np, self._cached_state_np):
                return self._cached_state_emb
        
        # Compute new embedding
        with torch.no_grad():
            state_t = self._to_tensor_state(state_np)
            self._cached_state_emb = self.q_net.encode_state(state_t)
            self._cached_state_np = state_np.copy()
        
        return self._cached_state_emb
    
    def clear_state_cache(self):
        """Clear cached state embedding."""
        self._cached_state_emb = None
        self._cached_state_np = None
    
    def select_stage1(
        self,
        state_np: np.ndarray,
        pool: ActionPool,
        epsilon: Optional[float] = None
    ) -> Stage1Selection:
        """
        Stage 1: Select container or parking action.
        
        Args:
            state_np: Current state [R, B, T, C]
            pool: Available actions
            epsilon: Override epsilon (None = use schedule)
            
        Returns:
            Stage1Selection with chosen action
        """
        self.step_count += 1
        eps = epsilon if epsilon is not None else self._get_epsilon()
        
        n_containers = len(pool.containers)
        n_parkings = len(pool.parkings)
        total = n_containers + n_parkings
        
        if total == 0:
            # No actions available
            return Stage1Selection(is_parking=False, index=-1)
        
        # Epsilon-greedy
        if random.random() < eps:
            # Random selection
            idx = random.randrange(total)
            if idx < n_containers:
                cont = pool.containers[idx]
                feat = self.cont_featurizer.featurize_single(cont)
                return Stage1Selection(
                    is_parking=False,
                    index=idx,
                    container=cont,
                    container_feat=feat
                )
            else:
                park_idx = idx - n_containers
                return Stage1Selection(
                    is_parking=True,
                    index=park_idx,
                    parking=pool.parkings[park_idx]
                )
        
        # Greedy selection
        with torch.no_grad():
            state_emb = self.encode_state(state_np)
            
            q_values = []
            
            # Score containers
            if n_containers > 0:
                if pool.container_feats is None:
                    pool.container_feats = self.cont_featurizer.featurize_batch(
                        pool.containers
                    ).to(self.cfg.device)
                q_cont = self.q_net.score_containers(state_emb, pool.container_feats)
                q_values.append(q_cont)
            
            # Score parking
            if n_parkings > 0:
                if pool.parking_feats is None:
                    pool.parking_feats = self.park_featurizer.featurize_batch(
                        pool.parkings
                    ).to(self.cfg.device)
                q_park = self.q_net.score_parking(state_emb, pool.parking_feats)
                q_values.append(q_park)
            
            # Concatenate and find max
            all_q = torch.cat(q_values, dim=0)  # [total]
            best_idx = int(torch.argmax(all_q).item())
            
            if best_idx < n_containers:
                cont = pool.containers[best_idx]
                feat = self.cont_featurizer.featurize_single(cont)
                return Stage1Selection(
                    is_parking=False,
                    index=best_idx,
                    container=cont,
                    container_feat=feat
                )
            else:
                park_idx = best_idx - n_containers
                return Stage1Selection(
                    is_parking=True,
                    index=park_idx,
                    parking=pool.parkings[park_idx]
                )
    
    def select_stage2(
        self,
        state_np: np.ndarray,
        container_feat: np.ndarray,
        destinations: List[Destination],
        source_bay: int,
        source_tier: int,
        epsilon: Optional[float] = None
    ) -> int:
        """
        Stage 2: Select destination for container.
        
        Args:
            state_np: Current state
            container_feat: Selected container features [16]
            destinations: Available destinations
            source_bay: Container's current bay
            source_tier: Container's current tier
            epsilon: Override epsilon
            
        Returns:
            Index of selected destination, or -1 if none available
        """
        n_dests = len(destinations)
        if n_dests == 0:
            return -1
        
        eps = epsilon if epsilon is not None else self._get_epsilon()
        
        # Epsilon-greedy
        if random.random() < eps:
            return random.randrange(n_dests)
        
        # Greedy
        with torch.no_grad():
            state_emb = self.encode_state(state_np)
            
            dest_feats = self.dest_featurizer.featurize_batch(
                destinations, source_bay, source_tier
            ).to(self.cfg.device)
            
            cont_feat_t = torch.from_numpy(container_feat).float().to(self.cfg.device)
            
            q_dests = self.q_net.score_destinations(state_emb, cont_feat_t, dest_feats)
            return int(torch.argmax(q_dests).item())
    
    def remember(self, transition: HierarchicalTransition):
        """Store transition in replay buffer."""
        self.replay.push(transition)
    
    def optimize(self) -> float:
        """
        Perform one optimization step.
        
        Returns:
            Loss value
        """
        if len(self.replay) < self.cfg.batch_size:
            return 0.0
        
        batch = self.replay.sample(self.cfg.batch_size)
        
        total_loss = 0.0
        valid_count = 0
        
        self.optimizer.zero_grad(set_to_none=True)
        
        for trans in batch:
            if trans.was_parking:
                # Parking transition - simpler update
                loss = self._compute_parking_loss(trans)
            else:
                # Container move - two-stage update
                loss = self._compute_hierarchical_loss(trans)
            
            if loss is not None:
                loss.backward()
                total_loss += loss.item()
                valid_count += 1
        
        if valid_count > 0:
            # Gradient clipping
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            self._soft_update_target()
        
        return total_loss / max(1, valid_count)
    
    def _compute_parking_loss(self, trans: HierarchicalTransition) -> Optional[torch.Tensor]:
        """Compute loss for parking transition."""
        if trans.parking_feats is None or trans.parking_idx < 0:
            return None
        
        state_t = self._to_tensor_state(trans.state)
        state_emb = self.q_net.encode_state(state_t)
        
        parking_feats = torch.from_numpy(trans.parking_feats).float().to(self.cfg.device)
        q_park = self.q_net.score_parking(state_emb, parking_feats)
        
        if trans.parking_idx >= q_park.size(0):
            return None
        
        q_sa = q_park[trans.parking_idx]
        
        # Target (parking is typically terminal for that action)
        with torch.no_grad():
            if trans.done:
                target = trans.reward
            else:
                next_state_t = self._to_tensor_state(trans.next_state)
                next_emb = self.target_net.encode_state(next_state_t)
                # For next state, we'd need the next pool - approximate with 0
                target = trans.reward
        
        target_t = torch.tensor(target, dtype=torch.float32, device=self.cfg.device)
        return nn.SmoothL1Loss()(q_sa, target_t)
    
    def _compute_hierarchical_loss(self, trans: HierarchicalTransition) -> Optional[torch.Tensor]:
        """Compute loss for container move transition."""
        if trans.container_feats.shape[0] == 0 or trans.destination_feats.shape[0] == 0:
            return None
        if trans.container_idx < 0 or trans.destination_idx < 0:
            return None
        
        state_t = self._to_tensor_state(trans.state)
        state_emb = self.q_net.encode_state(state_t)
        
        # Stage 1: Container selection Q-value
        cont_feats = torch.from_numpy(trans.container_feats).float().to(self.cfg.device)
        
        if trans.container_idx >= cont_feats.size(0):
            return None
        
        q_containers = self.q_net.score_containers(state_emb, cont_feats)
        q_stage1 = q_containers[trans.container_idx]
        
        # Stage 2: Destination selection Q-value
        cont_feat = torch.from_numpy(trans.container_feat).float().to(self.cfg.device)
        dest_feats = torch.from_numpy(trans.destination_feats).float().to(self.cfg.device)
        
        if trans.destination_idx >= dest_feats.size(0):
            return None
        
        q_dests = self.q_net.score_destinations(state_emb, cont_feat, dest_feats)
        q_stage2 = q_dests[trans.destination_idx]
        
        # Combined Q-value (average or sum)
        q_combined = (q_stage1 + q_stage2) / 2.0
        
        # Target
        with torch.no_grad():
            if trans.done:
                target = trans.reward
            else:
                # Bootstrap from next state
                next_state_t = self._to_tensor_state(trans.next_state)
                next_emb = self.target_net.encode_state(next_state_t)
                
                # Approximate max Q for next state
                # We don't have next pool, so use simplified estimate
                target = trans.reward + self.cfg.gamma * 0.0  # Conservative
        
        target_t = torch.tensor(target, dtype=torch.float32, device=self.cfg.device)
        return nn.SmoothL1Loss()(q_combined, target_t)
    
    def save(self, path: str):
        """Save agent state."""
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "cfg": self.cfg.__dict__,
        }, path)
    
    def load(self, path: str, map_location: Optional[str] = None):
        """Load agent state."""
        loc = map_location or self.cfg.device
        ckpt = torch.load(path, map_location=loc)
        
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step_count = ckpt.get("step_count", 0)
    
    def eval_mode(self):
        """Set to evaluation mode."""
        self.q_net.eval()
        self.target_net.eval()
    
    def train_mode(self):
        """Set to training mode."""
        self.q_net.train()
        self.target_net.train()
