# agents/continuous_terminal_agent.py - Complete version
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import copy
from typing import Dict, List, Tuple, Optional, Any
from collections import deque, namedtuple
from simulation.agents.base_agent import BaseTransferableAgent, MoveEmbedder

# Define Experience tuple
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
        reward_influence = min(abs(experience.reward), 100.0) + 1.0
        
        if self.priorities:
            max_priority = max(self.priorities)
        else:
            max_priority = 1.0
            
        # Higher priority for experiences with significant rewards
        initial_priority = min(max_priority * reward_influence, 1e6)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            
        self.priorities.append(initial_priority)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size, current_day=0):
        """Sample with enhanced priority for recent high-reward experiences."""
        if len(self.buffer) < batch_size:
            return None
        
        # Calculate priorities with stronger recency bias
        priorities = []
        for i, exp in enumerate(self.buffer):
            # Base priority from TD error
            td_priority = np.clip(abs(exp.td_error), 0.01, 100.0) ** self.alpha
            
            # Stronger recency factor for current day
            days_old = max(0, current_day - exp.day)
            if days_old == 0:  # Current day
                recency_factor = 2.0
            else:
                recency_factor = np.exp(-days_old * 0.2)  # Faster decay
            
            # Reward influence - prioritize high rewards
            if exp.reward > 10.0:
                reward_factor = 3.0
            elif exp.reward > 5.0:
                reward_factor = 2.0
            elif exp.reward > 0:
                reward_factor = 1.5
            else:
                reward_factor = 0.8
            
            # Combined priority
            priority = td_priority * recency_factor * reward_factor
            priority = np.clip(priority, 1e-8, 1e8)
            priorities.append(priority)
        
        # Convert to probabilities
        priorities = np.array(priorities)
        probabilities = priorities / (priorities.sum() + 1e-8)
        
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
        
        # Convert to arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        return states, actions, rewards, next_states, dones, infos, indices
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, priority in zip(indices, priorities):
            # Ensure priority is positive and not too small
            priority = max(abs(priority), 1e-6) ** self.alpha
            self.priorities[idx] = priority
    
    def soft_reset(self, current_day: int):
        """Soft reset: reduce influence of old negative experiences."""
        self.current_day = current_day
    
    def __len__(self):
        return len(self.buffer)


class ContinuousTerminalAgent(BaseTransferableAgent):
    """Adapted ContinuousTerminalAgent for the transferable training pipeline."""
    
    def __init__(self, state_dim: int, 
                 learning_rate: float = 0.0001,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 device: str = 'cuda'):
        super().__init__(state_dim, device)
        
        self.gamma = gamma
        self.tau = tau
        
        # Build dueling network architecture
        self._build_network([512, 512, 256])
        
        # Create target network
        self.target_net = copy.deepcopy(self)
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=0.0001)
        
        # Replace memory with prioritized replay buffer
        self.memory = PrioritizedReplayBuffer(capacity=20000)
        
        # Training parameters
        self.batch_size = 64
        self.update_every = 4
        self.step_count = 0
        self.current_day = 0
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        
        # Performance tracking
        self.training_stats = {
            'losses': deque(maxlen=1000),
            'td_errors': deque(maxlen=1000),
            'q_values': deque(maxlen=1000),
            'rewards': deque(maxlen=1000)
        }
        
    def _build_network(self, hidden_dims: List[int]):
        """Build dueling DQN architecture."""
        layers = []
        input_dim = self.state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # Value stream
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Advantage stream - dynamic size based on moves
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
        """Forward pass with dueling architecture."""
        features = self.feature_extractor(state)
        
        value = self.value_head(features)
        advantages = self.advantage_head(features)
        
        # Dueling formula
        q_values = value + advantages - advantages.mean(dim=-1, keepdim=True)
        
        # Apply action mask
        if action_mask is not None:
            q_values = q_values.masked_fill(~action_mask, -1e9)
            
        return q_values
    
    def select_action(self, state: torch.Tensor,
                     available_moves: Dict[str, Dict],
                     move_embeddings: torch.Tensor,
                     epsilon: float = 0.0,
                     ranked_actions: List[int] = None) -> Tuple[int, Dict]:
        """Select action compatible with new pipeline."""
        # Convert state to numpy for compatibility
        if isinstance(state, torch.Tensor):
            state_np = state.cpu().numpy()
        else:
            state_np = state
            
        # Get available action indices
        available_actions = list(range(len(available_moves)))
        
        if not available_actions:
            return 0, {'move_id': 'wait', 'move_data': {'move_type': 'wait'}}
        
        # Use epsilon if not provided
        if epsilon == 0.0 and self.training:
            epsilon = self.epsilon
        
        # Exploration
        if self.training and np.random.random() < epsilon:
            if ranked_actions and np.random.random() < 0.7:
                top_k = max(1, len(ranked_actions) // 4)
                action = random.choice(ranked_actions[:top_k])
            else:
                action = random.choice(available_actions)
        else:
            # Exploitation
            with torch.no_grad():
                if not isinstance(state, torch.Tensor):
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                else:
                    state_tensor = state.unsqueeze(0) if state.dim() == 1 else state
                
                # Create action mask
                action_mask = torch.zeros(1000, dtype=torch.bool, device=self.device)
                action_mask[available_actions] = True
                action_mask = action_mask.unsqueeze(0)
                
                # Get Q-values
                q_values = self.forward(state_tensor, action_mask)
                
                # Apply urgency bonus if provided
                if ranked_actions and self.training:
                    valid_q_values = q_values[0, available_actions].clone()
                    for i, act in enumerate(available_actions):
                        if act in ranked_actions[:5]:
                            rank = ranked_actions.index(act)
                            valid_q_values[i] += (5 - rank) * 0.1
                    best_idx = valid_q_values.argmax().item()
                    action = available_actions[best_idx]
                else:
                    action = available_actions[q_values[0, available_actions].argmax().item()]
        
        # Get move info
        move_list = list(available_moves.items())
        move_id, move_data = move_list[action]
        
        return action, {'move_id': move_id, 'move_data': move_data}
    
    def store_transition(self, state, action, reward, next_state, done, info):
        """Store experience with prioritization."""
        # Ensure info contains move_list
        if 'available_moves' in info and 'move_list' not in info:
            info['move_list'] = list(info['available_moves'].keys())
        
        # Calculate TD error for prioritization
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            
            # Current Q-value
            current_q = self.forward(state_tensor)[0, action].item()
            
            # Next state value
            next_moves = info.get('available_moves', {})
            if next_moves and not done:
                next_actions = list(range(len(next_moves)))
                next_mask = torch.zeros(1000, dtype=torch.bool, device=self.device)
                next_mask[next_actions] = True
                next_mask = next_mask.unsqueeze(0)
                
                next_q_values = self.forward(next_state_tensor, next_mask)
                next_action = next_q_values.argmax(dim=1).item()
                
                target_q_values = self.target_net.forward(next_state_tensor)
                next_value = target_q_values[0, next_action].item()
            else:
                next_value = 0.0
            
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
            day=info.get('day', self.current_day)
        )
        
        self.memory.add(experience)
        self.training_stats['rewards'].append(reward)
    
    def update(self, batch_size: int = 32) -> Dict[str, float]:
        """Perform training update compatible with pipeline."""
        if len(self.memory) < batch_size:
            return {}
        
        # Use the actual batch size from the agent
        batch_size = self.batch_size
        
        # Sample from prioritized replay buffer
        batch_data = self.memory.sample(batch_size, self.current_day)
        if batch_data is None:
            return {}
            
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
                    move_list = infos[i].get('available_moves', {})
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
        
        # Update priorities
        with torch.no_grad():
            td_errors = (targets - current_q_values).squeeze().cpu().numpy()
            self.memory.update_priorities(indices, np.abs(td_errors))
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.update_every == 0:
            self._soft_update_target()
        
        # Track statistics
        self.training_stats['losses'].append(loss.item())
        self.training_stats['q_values'].append(current_q_values.mean().item())
        self.training_stats['td_errors'].append(np.mean(np.abs(td_errors)))
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return {
            'loss': loss.item(),
            'epsilon': self.epsilon,
            'q_values': current_q_values.mean().item(),
            'td_error': np.mean(np.abs(td_errors))
        }
    
    def _soft_update_target(self):
        """Soft update target network."""
        for target_param, param in zip(self.target_net.parameters(), self.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def day_end_update(self, day: int, daily_reward: float):
        """Enhanced day-end update with performance-based exploration adjustment."""
        self.current_day = day
        
        # Soft reset in replay buffer
        self.memory.soft_reset(day)
        
        # Adjust exploration based on performance
        if daily_reward > 100:  # Excellent day
            self.epsilon *= 0.95
        elif daily_reward > 50:  # Good day
            self.epsilon *= 0.97
        elif daily_reward > 0:  # Positive day
            self.epsilon *= 0.99
        else:  # Bad day
            self.epsilon = min(self.epsilon * 1.05, 0.4)
        
        # Ensure minimum exploration
        self.epsilon = max(self.epsilon, self.epsilon_min)
        
        # Extra training at day end
        for _ in range(20):
            self.update()
    
    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'model_state': self.state_dict(),
            'target_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'current_day': self.current_day,
            'steps': self.steps,
            'episodes': self.episodes,
            'training_stats': dict(self.training_stats)
        }, path)
    
    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state'])
        self.target_net.load_state_dict(checkpoint['target_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint.get('epsilon', 0.1)
        self.current_day = checkpoint.get('current_day', 0)
        self.steps = checkpoint.get('steps', 0)
        self.episodes = checkpoint.get('episodes', 0)
        
        # Restore training stats if available
        if 'training_stats' in checkpoint:
            for key, value in checkpoint['training_stats'].items():
                self.training_stats[key] = deque(value, maxlen=1000)