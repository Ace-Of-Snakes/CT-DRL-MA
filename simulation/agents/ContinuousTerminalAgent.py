import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from typing import Dict, Tuple, List, Optional, Any


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
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            
        self.priorities.append(max_priority)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int, current_day: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample batch with prioritization and day-based weighting."""
        if len(self.buffer) < batch_size:
            return None, None, None
            
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Apply day-based soft reset weights
        experiences = []
        for idx in indices:
            exp = self.buffer[idx]
            experiences.append(exp)
            
            # Reduce weight for older experiences (soft reset)
            day_diff = current_day - exp.day
            if day_diff > 0:
                # Exponential decay: experiences lose 10% importance per day
                day_weight = 0.9 ** day_diff
                weights[len(experiences)-1] *= day_weight
        
        return experiences, weights, indices
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
    
    def soft_reset(self, current_day: int):
        """Soft reset: reduce influence of old negative experiences."""
        self.current_day = current_day
        # This is handled in the sample method with day-based weighting


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
        
        # Target network
        self.target_net = ContinuousTerminalAgent(
            state_dim, hidden_dims, learning_rate, gamma, tau, device='cpu'
        )
        self.target_net.load_state_dict(self.state_dict())
        self.target_net.to(device)
        self.target_net.eval()
        
        # Optimizer with gradient clipping
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=0.0001)
        
        # Experience replay
        self.replay_buffer = PrioritizedReplayBuffer(capacity=100000)
        
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
            next_actions = info.get('move_list', [])
            if next_actions and not done:
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
        
    def update(self) -> Optional[float]:
        """Update network from replay buffer."""
        self.step_count += 1
        
        # Only update periodically
        if self.step_count % self.update_every != 0:
            return None
            
        # Sample batch
        experiences, weights, indices = self.replay_buffer.sample(
            self.batch_size, 
            self.current_day
        )
        
        if experiences is None:
            return None
            
        # Convert to tensors
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.FloatTensor([e.done for e in experiences]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q-values
        current_q_values = self.forward(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q-values (double DQN with action masking)
        with torch.no_grad():
            next_q_values = torch.zeros(self.batch_size, device=self.device)
            
            for i, exp in enumerate(experiences):
                if not exp.done:
                    next_actions = exp.info.get('move_list', [])
                    if next_actions:
                        # Create mask
                        mask = torch.zeros(1000, dtype=torch.bool, device=self.device)
                        mask[next_actions] = True
                        
                        # Select action with main network
                        next_q = self.forward(next_states[i:i+1], mask.unsqueeze(0))
                        next_action = next_q.argmax(dim=1).item()
                        
                        # Evaluate with target network
                        target_q = self.target_net.forward(next_states[i:i+1])
                        next_q_values[i] = target_q[0, next_action]
                        
        # Compute targets
        targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss with importance sampling weights
        td_errors = targets - current_q_values
        loss = (weights * td_errors.pow(2)).mean()
        
        # Add L2 regularization
        l2_reg = sum(p.pow(2).sum() for p in self.parameters())
        loss = loss + 0.0001 * l2_reg
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        
        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        # Soft update target network
        self._soft_update_target()
        
        # Update exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Track statistics
        self.training_stats['losses'].append(loss.item())
        self.training_stats['td_errors'].append(td_errors.abs().mean().item())
        self.training_stats['q_values'].append(current_q_values.mean().item())
        
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