import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from typing import Dict, Tuple, List, Optional, Any
import copy


Experience = namedtuple('Experience', 
    ['state', 'action', 'reward', 'next_state', 'done', 'info', 'td_error', 'day'])


class PrioritizedReplayBuffer:
    """Experience replay with prioritization and day-based soft reset."""
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = 0.00001
        
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.position = 0
        
        # Day tracking for soft reset
        self.current_day = 0
        self.day_weights = {}  # Maps day -> weight multiplier
        
    def add(self, experience: Experience):
        """Add experience with initial high priority."""
        # Use reward magnitude to influence initial priority
        reward_influence = abs(experience.reward) + 1.0
        
        if self.priorities:
            max_priority = max(self.priorities)
        else:
            max_priority = 1.0
            
        # Higher priority for experiences with significant rewards
        initial_priority = max_priority * reward_influence
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            
        self.priorities.append(initial_priority)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size, current_day=0):
        """Sample a batch with prioritization and day-awareness."""
        if len(self.buffer) < batch_size:
            return None
            
        # Calculate priorities with overflow protection
        priorities = []
        for exp in self.buffer:
            # Base priority from TD error (clipped to prevent overflow)
            td_priority = np.clip(abs(exp.td_error), 0.01, 100.0) ** self.alpha
            
            # Day recency factor (also clipped)
            days_old = max(0, current_day - exp.day)
            recency_factor = np.exp(-days_old * 0.1)  # Exponential decay
            
            # Reward influence (clipped to prevent overflow)
            reward_factor = 1.0 + np.clip(exp.reward, -10, 10) * 0.1
            
            # Combined priority (with safety checks)
            priority = td_priority * recency_factor * reward_factor
            priority = np.clip(priority, 1e-8, 1e8)  # Prevent extreme values
            priorities.append(priority)
        
        # Convert to probabilities (with safety checks)
        priorities = np.array(priorities)
        if np.any(np.isnan(priorities)) or np.all(priorities == 0):
            # Fallback to uniform sampling
            probabilities = np.ones(len(self.buffer)) / len(self.buffer)
        else:
            probabilities = priorities / (priorities.sum() + 1e-8)  # Add epsilon to prevent division by zero
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Extract batch data
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        infos = []
        
        for idx in indices:
            exp = self.buffer[idx]
            states.append(exp.state)
            actions.append(exp.action)
            rewards.append(exp.reward)
            next_states.append(exp.next_state)
            dones.append(exp.done)
            infos.append(exp.info)
        
        # Convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        return states, actions, rewards, next_states, dones, infos, indices
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities based on combined TD errors and rewards."""
        for idx, priority in zip(indices, priorities):
            # Ensure priority is positive and not too small
            priority = max(abs(priority), 1e-6) ** self.alpha
            self.priorities[idx] = priority
    
    def soft_reset(self, current_day: int):
        """Soft reset: reduce influence of old negative experiences."""
        self.current_day = current_day
        # This is handled in the sample method with day-based weighting

    def __len__(self):
        return len(self.buffer)  # or whatever the internal storage is called

class ContinuousTerminalAgent(nn.Module):
    """
    DRL agent optimized for continuous terminal operations.
    Features:
    - Dueling DQN architecture
    - Prioritized experience replay
    - Day-based soft reset mechanism
    - Action masking for valid moves
    - Adaptive exploration
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [512, 512, 256],
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.device = device
        self.gamma = gamma
        self.tau = tau
        
        # Build network architecture
        self._build_network(hidden_dims)
        
        # Move to device
        self.to(device)
        
        # Create target network as a deep copy
        self.target_net = copy.deepcopy(self)
        self.target_net.eval()
        
        # Optimizer with gradient clipping
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=0.0001)
        
        # Experience replay
        self.replay_buffer = PrioritizedReplayBuffer(capacity=20000)  # Reduced from 100k
        
        # Training parameters
        self.batch_size = 64
        self.update_every = 4
        self.step_count = 0
        
        # Exploration parameters with day-based adaptation
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.current_day = 0
        
        # Performance tracking
        self.training_stats = {
            'losses': deque(maxlen=1000),
            'td_errors': deque(maxlen=1000),
            'q_values': deque(maxlen=1000),
            'rewards': deque(maxlen=1000)
        }
        
    def _build_network(self, hidden_dims: List[int]):
        """Build dueling DQN architecture."""
        # Shared feature extractor
        layers = []
        input_dim = self.state_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_dim = hidden_dim
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # Value stream
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Advantage stream
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 256),
            nn.ReLU(),
            nn.Linear(256, 1000)  # Max possible actions
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
    def forward(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with dueling architecture and action masking."""
        features = self.feature_extractor(state)
        
        # Compute value and advantages
        value = self.value_head(features)
        advantages = self.advantage_head(features)
        
        # Combine into Q-values (dueling formula)
        q_values = value + advantages - advantages.mean(dim=-1, keepdim=True)
        
        # Apply action mask if provided
        if action_mask is not None:
            # Set invalid actions to large negative value
            q_values = q_values.masked_fill(~action_mask, -1e9)
            
        return q_values
    
    def select_action(
        self, 
        state: np.ndarray, 
        available_actions: List[int],
        epsilon: Optional[float] = None,
        training: bool = True
    ) -> int:
        """Select action using epsilon-greedy with action masking."""
        if epsilon is None:
            epsilon = self.epsilon if training else 0.0
            
        # Exploration
        if training and random.random() < epsilon:
            return random.choice(available_actions) if available_actions else 0
            
        # Exploitation
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Create action mask
            action_mask = torch.zeros(1000, dtype=torch.bool, device=self.device)
            action_mask[available_actions] = True
            action_mask = action_mask.unsqueeze(0)
            
            # Get Q-values
            q_values = self.forward(state_tensor, action_mask)
            
            # Select best available action
            if available_actions:
                valid_q_values = q_values[0, available_actions]
                best_idx = valid_q_values.argmax().item()
                return available_actions[best_idx]
            else:
                return 0
    
    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Dict[str, Any]
    ):
        """Store experience with initial TD error estimate."""
        # Calculate initial TD error for prioritization
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            
            # Current Q-value
            current_q = self.forward(state_tensor)[0, action].item()
            
            # Next state value (double DQN)
            move_list = info.get('move_list', [])
            if move_list and not done:
                # Convert move list to integer indices
                next_actions = list(range(len(move_list)))  # <-- FIX: Convert to indices
                
                # Create mask for next state
                next_mask = torch.zeros(1000, dtype=torch.bool, device=self.device)
                next_mask[next_actions] = True
                next_mask = next_mask.unsqueeze(0)
                
                # Select action using main network
                next_q_values = self.forward(next_state_tensor, next_mask)
                next_action = next_q_values.argmax(dim=1).item()
                
                # Evaluate using target network
                target_q_values = self.target_net.forward(next_state_tensor)
                next_value = target_q_values[0, next_action].item()
            else:
                next_value = 0.0
                
            # TD error
            td_error = reward + self.gamma * next_value * (1 - done) - current_q
            
        # Create experience
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info,
            td_error=td_error,
            day=info.get('day', 0)
        )
        
        # Add to buffer
        self.replay_buffer.add(experience)
        
        # Track reward
        self.training_stats['rewards'].append(reward)
        
    def update(self, current_day=0):
        """Update Q-network from sampled experiences."""
        # Only update after sufficient experiences
        if len(self.replay_buffer.buffer) < self.batch_size:
            return None
            
        # Sample batch - returns a tuple
        batch_data = self.replay_buffer.sample(self.batch_size, current_day)
        
        # Unpack the tuple
        states, actions, rewards, next_states, dones, infos, indices = batch_data
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.forward(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values with double DQN
        with torch.no_grad():
            # Create masks for valid next actions
            batch_size = states.size(0)
            next_masks = torch.zeros(batch_size, 1000, dtype=torch.bool, device=self.device)
            
            for i in range(batch_size):
                if not dones[i]:
                    # FIX: Convert move_list strings to integer indices
                    move_list = infos[i].get('move_list', [])
                    if move_list:
                        next_actions = list(range(len(move_list)))
                        next_masks[i, next_actions] = True
            
            # Select actions using main network
            next_q_values = self.forward(next_states, next_masks)
            next_actions_selected = next_q_values.argmax(dim=1)
            
            # Evaluate using target network
            target_next_q_values = self.target_net.forward(next_states)
            next_q_values_selected = target_next_q_values.gather(1, next_actions_selected.unsqueeze(1))
            
            # Compute targets
            targets = rewards.unsqueeze(1) + self.gamma * next_q_values_selected * (1 - dones.unsqueeze(1))
        
        # Compute loss
        loss = F.mse_loss(current_q_values, targets)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        
        # Update priorities in replay buffer
        with torch.no_grad():
            td_errors = (targets - current_q_values).squeeze().cpu().numpy()
            self.replay_buffer.update_priorities(indices, np.abs(td_errors))
        
        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.update_every == 0:
            self._soft_update_target()
        
        # Track statistics
        self.training_stats['losses'].append(loss.item())
        self.training_stats['q_values'].append(current_q_values.mean().item())
        self.training_stats['td_errors'].append(np.mean(np.abs(td_errors)))
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
        
    def _soft_update_target(self):
        """Soft update target network parameters."""
        for target_param, param in zip(self.target_net.parameters(), self.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
    def day_end_update(self, day: int, daily_reward: float):
        """Special update at end of day with soft reset."""
        self.current_day = day
        
        # Soft reset in replay buffer
        self.replay_buffer.soft_reset(day)
        
        # Adaptive exploration based on daily performance
        if daily_reward > 0:
            # Good day - reduce exploration faster
            self.epsilon *= 0.98
        else:
            # Bad day - maintain exploration
            self.epsilon = min(self.epsilon * 1.02, 0.3)
            
        # Extra training at day end
        for _ in range(10):
            self.update()
            
    def save(self, filepath: str):
        """Save agent state."""
        torch.save({
            'model_state': self.state_dict(),
            'target_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'current_day': self.current_day,
            'training_stats': dict(self.training_stats)
        }, filepath)
        
    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state'])
        self.target_net.load_state_dict(checkpoint['target_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint['epsilon']
        self.current_day = checkpoint['current_day']
        
        # Restore training stats
        for key, value in checkpoint['training_stats'].items():
            self.training_stats[key] = deque(value, maxlen=1000)