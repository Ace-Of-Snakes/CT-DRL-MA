import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import time
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Define experience tuple structure
Experience = namedtuple('Experience', ('state', 'role', 'action', 'reward', 'next_state', 'done', 'action_masks'))


class ReplayBuffer:
    """Experience replay buffer for multi-agent learning"""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, role, action, reward, next_state, done, action_masks):
        """Save an experience"""
        self.buffer.append(Experience(state, role, action, reward, next_state, done, action_masks))
    
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        experiences = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        # Convert batch of experiences to experience of batches
        states = torch.FloatTensor([e.state for e in experiences])
        roles = [e.role for e in experiences]
        actions = torch.LongTensor([e.action for e in experiences]).unsqueeze(1)
        rewards = torch.FloatTensor([e.reward for e in experiences]).unsqueeze(1)
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences]).unsqueeze(1)
        
        # Handle action masks
        action_masks = {}
        next_action_masks = {}
        
        # Get all unique roles
        all_roles = set(roles)
        
        # Gather masks for each role
        for role in all_roles:
            role_indices = [i for i, r in enumerate(roles) if r == role]
            role_masks = [experiences[i].action_masks[role] for i in role_indices]
            if role_masks:
                action_masks[role] = torch.stack([m for m in role_masks])
                next_action_masks[role] = torch.stack([m for m in role_masks])  # Same masks for simplicity
        
        return (states, roles, actions, rewards, next_states, dones, action_masks, next_action_masks)
    
    def __len__(self):
        return len(self.buffer)


class SharedKnowledgeRepository(nn.Module):
    """Central knowledge repository shared by all agents"""
    
    def __init__(self, state_dim, hidden_dim=256, agent_roles=None):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.agent_roles = agent_roles or ['pre_marshalling', 'direct_transfer', 'truck_parking', 'train_loading']
        
        # State encoder - transforms terminal state into knowledge representation
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Global knowledge representation
        self.global_knowledge = nn.Parameter(torch.zeros(hidden_dim))
        
        # Hypernetwork for QMIX - generates mixing network weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(self.agent_roles))
        )
        
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.hyper_b1 = nn.Linear(state_dim, 1)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def encode_state(self, state):
        """Encode global state into knowledge representation"""
        encoded = self.state_encoder(state)
        # Add global knowledge embedding
        knowledge_state = encoded + self.global_knowledge
        return knowledge_state
    
    def mix_q_values(self, agent_q_values, state):
        """
        QMIX mixing network that combines individual agent Q-values
        into a global Q-value in a way that respects joint action selection
        
        Args:
            agent_q_values: Dictionary mapping roles to Q-values
            state: Global terminal state
        """
        # Ensure Q-values are in consistent order
        agent_qs = torch.stack([agent_q_values[role] for role in self.agent_roles if role in agent_q_values])
        
        # If we're missing some roles, pad with zeros
        if len(agent_qs) < len(self.agent_roles):
            padding = torch.zeros(len(self.agent_roles) - len(agent_qs), device=agent_qs.device)
            agent_qs = torch.cat([agent_qs, padding])
        
        # Generate mixing network weights
        w1 = torch.abs(self.hyper_w1(state))  # Ensure positive weights
        b1 = self.hyper_b1(state)
        
        # First layer
        hidden = F.elu(torch.matmul(w1, agent_qs) + b1)
        
        # Second layer
        w2 = torch.abs(self.hyper_w2(state))  # Ensure positive weights
        b2 = self.hyper_b2(state)
        
        # Output global Q-value
        global_q = torch.matmul(w2, hidden) + b2
        
        return global_q


class RoleSpecificAgent(nn.Module):
    """Agent specialized for a specific role but using shared knowledge"""
    
    def __init__(self, role, action_dim, knowledge_dim=256, hidden_dim=128):
        super().__init__()
        self.role = role
        self.action_dim = action_dim
        
        # Knowledge interface - how this agent interprets the shared knowledge
        self.knowledge_interface = nn.Sequential(
            nn.Linear(knowledge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Advantage stream for dueling architecture
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value stream for dueling architecture
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, knowledge_state, action_mask=None):
        """
        Forward pass using shared knowledge
        
        Args:
            knowledge_state: Encoded knowledge from repository
            action_mask: Boolean mask for valid actions
        """
        # Process shared knowledge through this agent's interface
        features = self.knowledge_interface(knowledge_state)
        
        # Dueling architecture
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Apply action mask if provided
        if action_mask is not None:
            # Fill invalid actions with very negative values
            advantage = advantage.masked_fill(~action_mask, -1e9)
        
        # Combine value and advantage (dueling architecture)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class TerminalAgentSystem:
    """
    Complete system of agents with shared knowledge repository
    for managing container terminal operations
    """
    
    def __init__(self, state_dim, action_dims, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.state_dim = state_dim
        self.action_dims = action_dims
        self.roles = list(action_dims.keys())
        
        # Create shared knowledge repository
        self.knowledge_repo = SharedKnowledgeRepository(state_dim, agent_roles=self.roles).to(device)
        
        # Create target knowledge repository for stable learning
        self.target_knowledge_repo = SharedKnowledgeRepository(state_dim, agent_roles=self.roles).to(device)
        self.target_knowledge_repo.load_state_dict(self.knowledge_repo.state_dict())
        
        # Create role-specific agents
        self.agents = {
            role: RoleSpecificAgent(role, dim).to(device)
            for role, dim in action_dims.items()
        }
        
        # Create target networks for stable learning
        self.target_agents = {
            role: RoleSpecificAgent(role, dim).to(device)
            for role, dim in action_dims.items()
        }
        
        # Initialize target networks
        for role in self.roles:
            self.target_agents[role].load_state_dict(self.agents[role].state_dict())
        
        # Optimizer for all components
        self.optimizer = torch.optim.Adam(
            list(self.knowledge_repo.parameters()) + 
            [p for agent in self.agents.values() for p in agent.parameters()],
            lr=0.0005
        )
        
        # For epsilon-greedy exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Discount factor
        self.gamma = 0.99
        
        # Target network update rate
        self.tau = 0.005  # Soft updates
        
        # Experience replay
        self.memory = ReplayBuffer(capacity=100000)
        self.batch_size = 64
        
        # Training stats
        self.losses = []
        self.rewards = []
        self.epsilons = []
    
    def get_state_tensor(self, state):
        """Convert environment state to tensor with FIXED dimensions"""
        # Create a state vector with fixed dimensions
        state_dim = 1662  # This should match what the network expects
        
        # Create a zero-filled array with the expected size
        fixed_dim_state = np.zeros(state_dim, dtype=np.float32)
        
        # Fill in the components we can extract
        offset = 0
        
        # Crane positions: 4 values (2 cranes x 2 coordinates)
        if 'crane_positions' in state and state['crane_positions'] is not None:
            positions_flat = state['crane_positions'].flatten()
            end_idx = min(offset + len(positions_flat), state_dim)
            fixed_dim_state[offset:end_idx] = positions_flat[:end_idx-offset]
            offset = end_idx
        
        # Crane available times: 2 values (2 cranes)
        if 'crane_available_times' in state and state['crane_available_times'] is not None:
            times_flat = state['crane_available_times'].flatten()
            end_idx = min(offset + len(times_flat), state_dim)
            fixed_dim_state[offset:end_idx] = times_flat[:end_idx-offset]
            offset = end_idx
        
        # Current time: 1 value
        if 'current_time' in state and state['current_time'] is not None:
            time_flat = state['current_time'].flatten()
            end_idx = min(offset + len(time_flat), state_dim)
            fixed_dim_state[offset:end_idx] = time_flat[:end_idx-offset]
            offset = end_idx
        
        # Yard state: dimensions depend on yard size
        if 'yard_state' in state and state['yard_state'] is not None:
            # Reduce dimensionality using a summing approach
            yard_state = state['yard_state']
            # Sum across tiers to get occupancy
            yard_flat = np.sum(yard_state, axis=2).flatten()
            end_idx = min(offset + len(yard_flat), state_dim)
            fixed_dim_state[offset:end_idx] = yard_flat[:end_idx-offset]
            offset = end_idx
        
        # Parking status
        if 'parking_status' in state and state['parking_status'] is not None:
            parking_flat = state['parking_status'].flatten()
            end_idx = min(offset + len(parking_flat), state_dim)
            fixed_dim_state[offset:end_idx] = parking_flat[:end_idx-offset]
            offset = end_idx
        
        # Rail status
        if 'rail_status' in state and state['rail_status'] is not None:
            rail_flat = state['rail_status'].flatten()
            end_idx = min(offset + len(rail_flat), state_dim)
            fixed_dim_state[offset:end_idx] = rail_flat[:end_idx-offset]
            offset = end_idx
        
        # Queue sizes
        if 'queue_sizes' in state and state['queue_sizes'] is not None:
            queue_flat = state['queue_sizes'].flatten()
            end_idx = min(offset + len(queue_flat), state_dim)
            fixed_dim_state[offset:end_idx] = queue_flat[:end_idx-offset]
            offset = end_idx
        
        # Convert to tensor
        return torch.FloatTensor(fixed_dim_state).to(self.device)
    
    def get_action_masks(self, state):
        """Extract action masks for each role from environment state"""
        action_masks = {}
        
        # Extract masks from observation
        for role, mask_tensor in state['action_mask'].items():
            # Check format - we might need to flatten or otherwise process the mask
            if role == 'crane_movement':
                # This is a 3D tensor (crane, source, dest) - we need to process it
                # For each crane, create a separate role
                num_cranes = mask_tensor.shape[0]
                for crane_idx in range(num_cranes):
                    # Flatten the 2D mask for this crane into a 1D action space
                    crane_role = f"crane_{crane_idx}"
                    # Create a flattened mask
                    flat_mask = mask_tensor[crane_idx].flatten()
                    action_masks[crane_role] = torch.BoolTensor(flat_mask).to(self.device)
            elif role == 'truck_parking':
                # This is already 2D, just convert to tensor
                action_masks['truck_parking'] = torch.BoolTensor(mask_tensor).to(self.device)
        
        return action_masks
    
    def select_action(self, state, evaluation=False):
        """
        Select the best action across all agents, properly handling roles with no valid actions
        """
        # Convert state to knowledge representation
        state_tensor = self.get_state_tensor(state)
        knowledge_state = self.knowledge_repo.encode_state(state_tensor)
        
        # Get action masks
        action_masks = self.get_action_masks(state)
        
        # Check if we have any valid actions and filter out roles with zero valid actions
        valid_roles = [role for role, mask in action_masks.items() if mask.any()]
        if not valid_roles:
            # No valid actions available, return a dummy action
            return "dummy_role", 0, {'action_type': 0, 'crane_movement': [0, 0, 0]}
        
        # Exploration: randomly select a role and action
        epsilon = 0.0 if evaluation else self.epsilon
        if random.random() < epsilon:
            # Choose a random valid role
            random_role = random.choice(valid_roles)
            # Get valid actions for this role
            valid_actions = action_masks[random_role].nonzero(as_tuple=True)[0]
            # Choose a random valid action
            random_action_idx = random.randint(0, len(valid_actions) - 1)
            random_action = valid_actions[random_action_idx].item()
            
            # Convert to environment action format
            if random_role.startswith('crane_'):
                # This is a crane action
                crane_idx = int(random_role.split('_')[1])
                
                # Get the original mask from the state 
                crane_mask = state['action_mask']['crane_movement'][crane_idx]
                
                # Get dimensions to unflatten the action
                source_dim = crane_mask.shape[0]
                dest_dim = crane_mask.shape[1]
                
                # Unflatten the action index
                source_idx = random_action // dest_dim
                dest_idx = random_action % dest_dim
                
                action_data = {
                    'action_type': 0,  # Crane movement
                    'crane_movement': [crane_idx, source_idx, dest_idx]
                }
            elif random_role == 'truck_parking':
                # This is a truck parking action
                truck_mask = state['action_mask']['truck_parking']
                truck_dim = truck_mask.shape[0]
                spot_dim = truck_mask.shape[1]
                
                truck_idx = random_action // spot_dim
                spot_idx = random_action % spot_dim
                
                action_data = {
                    'action_type': 1,  # Truck parking
                    'truck_parking': [truck_idx, spot_idx]
                }
            
            return random_role, random_action, action_data
        
        # Greedy selection: get Q-values only from roles with valid actions
        q_values = {}
        for role in valid_roles:  # Only iterate over roles with valid actions
            # Get Q-values for this role
            role_q = self.agents[role](knowledge_state.unsqueeze(0), action_masks[role].unsqueeze(0)).squeeze(0)
            
            # Get maximum Q-value and corresponding action
            max_q, max_action = role_q.max(0)
            q_values[role] = (max_q.item(), max_action.item())
        
        # Now we're guaranteed to have at least one role in q_values
        best_role = max(q_values.keys(), key=lambda r: q_values[r][0])
        best_action = q_values[best_role][1]
        
        # Convert to environment action format
        if best_role.startswith('crane_'):
            # This is a crane action
            crane_idx = int(best_role.split('_')[1])
            
            # Get the original mask from the state 
            crane_mask = state['action_mask']['crane_movement'][crane_idx]
            
            # Get dimensions to unflatten the action
            source_dim = crane_mask.shape[0]
            dest_dim = crane_mask.shape[1]
            
            # Unflatten the action index
            source_idx = best_action // dest_dim
            dest_idx = best_action % dest_dim
            
            action_data = {
                'action_type': 0,  # Crane movement
                'crane_movement': [crane_idx, source_idx, dest_idx]
            }
        elif best_role == 'truck_parking':
            # This is a truck parking action
            truck_mask = state['action_mask']['truck_parking']
            truck_dim = truck_mask.shape[0]
            spot_dim = truck_mask.shape[1]
            
            truck_idx = best_action // spot_dim
            spot_idx = best_action % spot_dim
            
            action_data = {
                'action_type': 1,  # Truck parking
                'truck_parking': [truck_idx, spot_idx]
            }
        
        return best_role, best_action, action_data
    def get_action_masks(self, state):
        """
        Extract action masks for each role from environment state, ensuring proper dimensionality
        """
        action_masks = {}
        
        # Process crane movement masks
        if 'crane_movement' in state['action_mask']:
            crane_mask = state['action_mask']['crane_movement']
            num_cranes = crane_mask.shape[0]
            
            for crane_idx in range(num_cranes):
                # Check if this crane has any valid actions before creating a role
                if crane_mask[crane_idx].any():
                    crane_role = f"crane_{crane_idx}"
                    # Flatten the 2D mask (source, dest) into 1D
                    flat_mask = crane_mask[crane_idx].reshape(-1)
                    action_masks[crane_role] = torch.BoolTensor(flat_mask).to(self.device)
        
        # Process truck parking mask
        if 'truck_parking' in state['action_mask']:
            truck_mask = state['action_mask']['truck_parking']
            # Only include if there are valid actions
            if truck_mask.any():
                # Flatten the 2D mask (truck, spot) into 1D
                flat_mask = truck_mask.reshape(-1)
                action_masks['truck_parking'] = torch.BoolTensor(flat_mask).to(self.device)
        
        return action_masks
    def get_state_tensor(self, state):
        """Convert environment state to tensor, handling missing fields"""
        features = []
        
        # Handle each state component carefully
        if 'crane_positions' in state:
            features.append(state['crane_positions'].flatten())
        
        if 'crane_available_times' in state:
            features.append(state['crane_available_times'])
        
        if 'current_time' in state:
            features.append(state['current_time'])
        
        if 'yard_state' in state:
            # Use a reduced representation to keep tensor size manageable
            yard_state = state['yard_state']
            # Sum across tiers to get a 2D representation
            yard_occupancy = np.sum(yard_state, axis=2) > 0
            features.append(yard_occupancy.flatten().astype(np.float32))
        
        if 'parking_status' in state:
            features.append(state['parking_status'])
        
        if 'rail_status' in state:
            features.append(state['rail_status'].flatten())
        
        if 'queue_sizes' in state:
            features.append(state['queue_sizes'])
        
        # Concatenate all features and convert to tensor
        if features:
            state_tensor = np.concatenate([f.flatten() for f in features])
            return torch.FloatTensor(state_tensor).to(self.device)
        else:
            # Return a zero tensor if no features are available
            return torch.zeros(1).to(self.device)
    def update(self, batch_size=None):
        """Update the agent system based on experiences"""
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.memory) < batch_size:
            return 0.0  # Not enough samples
        
        # Sample batch from memory
        states, roles, actions, rewards, next_states, dones, action_masks, next_action_masks = self.memory.sample(batch_size)
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Encode states into knowledge representation
        encoded_states = self.knowledge_repo.encode_state(states)
        encoded_next_states = self.target_knowledge_repo.encode_state(next_states)
        
        # Calculate current Q-values
        current_q_values = {}
        for role in set(roles):
            # Get indices of samples with this role
            role_indices = [i for i, r in enumerate(roles) if r == role]
            if not role_indices:
                continue
                
            # Get state features for this role
            role_states = encoded_states[role_indices]
            role_actions = actions[role_indices]
            
            # Forward pass through agent to get Q-values
            agent = self.agents[role]
            q_values = agent(role_states)
            
            # Get Q-values for actions taken
            role_q_values = q_values.gather(1, role_actions)
            current_q_values[role] = role_q_values
        
        # Calculate target Q-values using Double DQN
        with torch.no_grad():
            target_q_values = torch.zeros_like(rewards)
            
            for role in set(roles):
                # Get indices of samples with this role
                role_indices = [i for i, r in enumerate(roles) if r == role]
                if not role_indices:
                    continue
                    
                # Get state features for this role
                role_next_states = encoded_next_states[role_indices]
                role_rewards = rewards[role_indices]
                role_dones = dones[role_indices]
                role_masks = next_action_masks.get(role)
                
                # Get best actions from current network
                agent = self.agents[role]
                q_values = agent(role_next_states, role_masks)
                best_actions = q_values.argmax(1).unsqueeze(1)
                
                # Get Q-values from target network
                target_agent = self.target_agents[role]
                target_q_values_role = target_agent(role_next_states)
                
                # Get Q-values for best actions
                target_q = target_q_values_role.gather(1, best_actions)
                
                # Calculate target using Bellman equation
                target = role_rewards + (1 - role_dones) * self.gamma * target_q
                
                # Assign to the right indices
                for i, idx in enumerate(role_indices):
                    target_q_values[idx] = target[i]
        
        # Combine Q-values from all roles
        agent_q_values = torch.cat([qv for qv in current_q_values.values()])
        
        # Calculate loss (MSE)
        loss = F.mse_loss(agent_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            list(self.knowledge_repo.parameters()) + 
            [p for agent in self.agents.values() for p in agent.parameters()],
            max_norm=10.0
        )
        self.optimizer.step()
        
        # Soft update target networks
        self._update_target_networks()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def _update_target_networks(self):
        """Soft update target networks"""
        # Update knowledge repository
        for target_param, param in zip(self.target_knowledge_repo.parameters(), self.knowledge_repo.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
        
        # Update agents
        for role in self.roles:
            for target_param, param in zip(self.target_agents[role].parameters(), self.agents[role].parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.tau) + param.data * self.tau
                )
    
    def save(self, path="qmix_model"):
        """Save model checkpoint"""
        os.makedirs(path, exist_ok=True)
        
        # Save knowledge repository
        torch.save(self.knowledge_repo.state_dict(), os.path.join(path, "knowledge_repo.pt"))
        
        # Save agents
        for role, agent in self.agents.items():
            torch.save(agent.state_dict(), os.path.join(path, f"agent_{role}.pt"))
        
        # Save optimizer
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
        
        # Save training stats
        training_stats = {
            'epsilon': self.epsilon,
            'losses': self.losses,
            'rewards': self.rewards,
            'epsilons': self.epsilons
        }
        torch.save(training_stats, os.path.join(path, "training_stats.pt"))
    
    def load(self, path="qmix_model"):
        """Load model checkpoint"""
        # Load knowledge repository
        self.knowledge_repo.load_state_dict(torch.load(os.path.join(path, "knowledge_repo.pt")))
        self.target_knowledge_repo.load_state_dict(self.knowledge_repo.state_dict())
        
        # Load agents
        for role in self.roles:
            agent_path = os.path.join(path, f"agent_{role}.pt")
            if os.path.exists(agent_path):
                self.agents[role].load_state_dict(torch.load(agent_path))
                self.target_agents[role].load_state_dict(self.agents[role].state_dict())
        
        # Load optimizer
        self.optimizer.load_state_dict(torch.load(os.path.join(path, "optimizer.pt")))
        
        # Load training stats
        training_stats = torch.load(os.path.join(path, "training_stats.pt"))
        self.epsilon = training_stats['epsilon']
        self.losses = training_stats['losses']
        self.rewards = training_stats['rewards']
        self.epsilons = training_stats['epsilons']
def calculate_state_dim(env):
    """Calculate the correct state dimension from a sample state"""
    # Get a sample state
    state, _ = env.reset()
    
    # Process it the same way we do during runtime
    features = []
    
    # Add each component
    if 'crane_positions' in state:
        features.append(state['crane_positions'].flatten())
    
    if 'crane_available_times' in state:
        features.append(state['crane_available_times'])
    
    if 'current_time' in state:
        features.append(state['current_time'])
    
    if 'yard_state' in state:
        # Use same dimensionality reduction as in runtime
        yard_state = state['yard_state']
        yard_occupancy = np.sum(yard_state, axis=2) > 0
        features.append(yard_occupancy.flatten().astype(np.float32))
    
    if 'parking_status' in state:
        features.append(state['parking_status'])
    
    if 'rail_status' in state:
        features.append(state['rail_status'].flatten())
    
    if 'queue_sizes' in state:
        features.append(state['queue_sizes'])
    
    # Concatenate and get dimension
    state_tensor = np.concatenate([f.flatten() for f in features])
    return state_tensor.shape[0]
def train_qmix_agent(env, agent_system, num_episodes=1000, max_steps=10000, 
                     log_dir="agent_logs"):
    """
    Improved training loop with better logging to CSV
    """
    # Set up logging
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.csv")
    
    with open(log_file, 'w') as f:
        f.write("episode,step,role,action_type,reward,valid_actions,total_reward\n")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Get action masks and count valid actions per role
            action_masks = agent_system.get_action_masks(state)
            valid_actions_by_role = {
                role: mask.sum().item() for role, mask in action_masks.items()
            }
            
            # Log the valid action counts
            valid_actions_str = ", ".join([f"{role}: {count}" 
                                        for role, count in valid_actions_by_role.items()])
            print(f"Step {step}, Valid actions: {valid_actions_str}")
            
            # Select action
            role, action_idx, action_data = agent_system.select_action(state)
            
            # Execute action
            next_state, reward, done, truncated, info = env.step(action_data)
            
            # Log action details
            action_type = action_data.get('action_type', -1)
            with open(log_file, 'a') as f:
                f.write(f"{episode},{step},{role},{action_type},{reward:.2f},")
                f.write(f"\"{valid_actions_str}\",{episode_reward + reward:.2f}\n")
            
            # Store experience
            if role != "dummy_role":
                try:
                    agent_system.memory.push(
                        agent_system.get_state_tensor(state).cpu().numpy(),
                        role,
                        action_idx,
                        reward,
                        agent_system.get_state_tensor(next_state).cpu().numpy(),
                        done or truncated,
                        action_masks
                    )
                except Exception as e:
                    print(f"Error storing experience: {str(e)}")
            
            # Update agent
            if len(agent_system.memory) >= agent_system.batch_size:
                agent_system.update()
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            if done or truncated:
                break
        
        print(f"Episode {episode} complete, Total reward: {episode_reward:.2f}")
        
        # Save model periodically
        if episode % 10 == 0:
            agent_system.save(f"qmix_model_ep{episode}")
    
    # Save final model
    agent_system.save("qmix_model_final")

class PerformanceLogger:
    """Logs performance metrics to CSV files"""
    
    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        
        # Create timestamped run folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(log_dir, f"run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Initialize log files
        self.episode_log_path = os.path.join(self.run_dir, "episode_metrics.csv")
        self.action_log_path = os.path.join(self.run_dir, "action_metrics.csv")
        self.role_log_path = os.path.join(self.run_dir, "role_metrics.csv")
        
        # Create headers
        with open(self.episode_log_path, 'w') as f:
            f.write("episode,total_reward,mean_loss,epsilon,memory_size,steps\n")
            
        with open(self.action_log_path, 'w') as f:
            f.write("episode,step,role,action_type,source,destination,reward,penalty,bonus,total\n")
            
        with open(self.role_log_path, 'w') as f:
            f.write("episode,role,actions_taken,total_reward,avg_reward\n")
    
    def log_episode(self, episode, total_reward, mean_loss, epsilon, memory_size, steps):
        """Log episode-level metrics"""
        with open(self.episode_log_path, 'a') as f:
            f.write(f"{episode},{total_reward:.2f},{mean_loss:.6f},{epsilon:.4f},{memory_size},{steps}\n")
    
    def log_action(self, episode, step, role, action_type, source, destination, reward_components):
        """Log detailed action metrics"""
        with open(self.action_log_path, 'a') as f:
            f.write(f"{episode},{step},{role},{action_type},{source},{destination}," +
                  f"{reward_components.get('reward', 0):.2f}," +
                  f"{reward_components.get('penalty', 0):.2f}," +
                  f"{reward_components.get('bonus', 0):.2f}," +
                  f"{reward_components.get('total', 0):.2f}\n")
    
    def log_role_performance(self, episode, role_metrics):
        """Log role-specific performance"""
        for role, metrics in role_metrics.items():
            with open(self.role_log_path, 'a') as f:
                f.write(f"{episode},{role},{metrics['count']},{metrics['total_reward']:.2f}," +
                      f"{metrics['avg_reward']:.2f}\n")
                
def evaluate_agent(env, agent_system, num_episodes=10):
    """Evaluate the agent without exploration"""
    eval_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action without exploration
            _, _, action_data = agent_system.select_action(state, evaluation=True)
            
            # Execute action
            next_state, reward, done, truncated, _ = env.step(action_data)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            if done or truncated:
                break
        
        eval_rewards.append(episode_reward)
    
    return np.mean(eval_rewards)


def plot_training_progress(agent_system, filename="training_progress.png"):
    """Plot training progress"""
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot rewards
    ax[0].plot(agent_system.rewards)
    ax[0].set_title("Episode Rewards")
    ax[0].set_xlabel("Episode")
    ax[0].set_ylabel("Reward")
    ax[0].grid(True)
    
    # Plot losses
    ax[1].plot(agent_system.losses)
    ax[1].set_title("Training Loss")
    ax[1].set_xlabel("Update Step")
    ax[1].set_ylabel("Loss")
    ax[1].grid(True)
    
    # Plot epsilon
    ax[2].plot(agent_system.epsilons)
    ax[2].set_title("Exploration Rate (Epsilon)")
    ax[2].set_xlabel("Episode")
    ax[2].set_ylabel("Epsilon")
    ax[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def train_qmix_agent_with_curriculum(env, agent_system, num_episodes=1000, max_steps=10000,
                                     checkpoint_interval=100, render_interval=100, eval_interval=100):
    """
    Training loop for the QMIX agent system with curriculum learning and detailed logging
    """
    # Initialize logger
    logger = PerformanceLogger()
    
    # Define curriculum
    curriculum = curriculum_learning_schedule(num_episodes)
    current_curriculum_level = 0
    
    # Apply initial curriculum level
    max_trucks, max_trains = curriculum[0][1], curriculum[0][2]
    setup_curriculum_learning(env, max_trucks, max_trains)
    print(f"Starting curriculum level 1: max_trucks={max_trucks}, max_trains={max_trains}")
    
    for episode in range(num_episodes):
        # Check if we need to update the curriculum
        for level, (level_start, level_trucks, level_trains) in enumerate(curriculum):
            if episode >= level_start and level > current_curriculum_level:
                current_curriculum_level = level
                max_trucks, max_trains = level_trucks, level_trains
                setup_curriculum_learning(env, max_trucks, max_trains)
                print(f"Advancing to curriculum level {level+1}: max_trucks={max_trucks}, max_trains={max_trains}")
                break
        
        state, _ = env.reset()
        episode_reward = 0
        losses = []
        steps = 0
        
        # Track role performance
        role_metrics = {}
        
        for step in range(max_steps):
            # Select action
            try:
                role, action_idx, action_data = agent_system.select_action(state)
            except Exception as e:
                print(f"Error in select_action: {e}")
                # Try a basic action as fallback
                role = "dummy_role"
                action_idx = 0
                action_data = {'action_type': 0, 'crane_movement': [0, 0, 0]}
            
            # Track action details for logging
            action_type = action_data.get('action_type', -1)
            source = "unknown"
            destination = "unknown"
            
            if action_type == 0 and 'crane_movement' in action_data:
                crane_idx, source_idx, dest_idx = action_data['crane_movement']
                # Convert indices to positions if possible
                source = env.idx_to_position.get(source_idx, f"source_{source_idx}")
                destination = env.idx_to_position.get(dest_idx, f"dest_{dest_idx}")
            elif action_type == 1 and 'truck_parking' in action_data:
                truck_idx, spot_idx = action_data['truck_parking']
                source = f"truck_{truck_idx}"
                destination = f"spot_{spot_idx}"
            
            # Execute action
            next_state, reward, done, truncated, info = env.step(action_data)
            
            # Log action details
            reward_components = {
                'total': reward,
                'reward': info.get('move_type_reward', 0),
                'penalty': info.get('distance_time_penalty', 0) + info.get('deadline_penalty', 0),
                'bonus': info.get('priority_bonus', 0) + info.get('deadline_bonus', 0)
            }
            logger.log_action(episode, step, role, action_type, source, destination, reward_components)
            
            # Update role metrics
            if role not in role_metrics:
                role_metrics[role] = {'count': 0, 'total_reward': 0, 'rewards': []}
            role_metrics[role]['count'] += 1
            role_metrics[role]['total_reward'] += reward
            role_metrics[role]['rewards'].append(reward)
            
            # Store experience
            if role != "dummy_role":
                action_masks = agent_system.get_action_masks(state)
                try:
                    agent_system.memory.push(
                        agent_system.get_state_tensor(state).cpu().numpy(),
                        role,
                        action_idx,
                        reward,
                        agent_system.get_state_tensor(next_state).cpu().numpy(),
                        done or truncated,
                        action_masks
                    )
                except Exception as e:
                    print(f"Error storing experience: {e}")
            
            # Update agent
            if len(agent_system.memory) >= agent_system.batch_size:
                try:
                    loss = agent_system.update()
                    losses.append(loss)
                except Exception as e:
                    print(f"Error updating agent: {e}")
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            steps += 1
            
            # Render occasionally
            if episode % render_interval == 0 and step % 100 == 0:
                env.render()
            
            if done or truncated:
                break
        
        # Calculate average reward per role
        for role in role_metrics:
            if role_metrics[role]['count'] > 0:
                role_metrics[role]['avg_reward'] = role_metrics[role]['total_reward'] / role_metrics[role]['count']
            else:
                role_metrics[role]['avg_reward'] = 0
        
        # Log episode metrics
        mean_loss = np.mean(losses) if losses else 0
        logger.log_episode(episode, episode_reward, mean_loss, agent_system.epsilon, 
                          len(agent_system.memory), steps)
        logger.log_role_performance(episode, role_metrics)
        
        # Print episode stats
        print(f"Episode {episode}: Reward = {episode_reward:.2f}, "
              f"Epsilon = {agent_system.epsilon:.4f}, "
              f"Loss = {mean_loss:.6f}, "
              f"Memory = {len(agent_system.memory)}, "
              f"Curriculum = {current_curriculum_level+1}/{len(curriculum)}")
        
        # Save checkpoint occasionally
        if episode % checkpoint_interval == 0:
            agent_system.save(f"qmix_model_ep{episode}")
            plot_training_progress(agent_system, f"training_progress_ep{episode}.png")
    
    # Final save
    agent_system.save("qmix_model_final")
    plot_training_progress(agent_system, "training_progress_final.png")

# Main function to set up and run training
def run_qmix_training():
    """Main function to set up and train the QMIX agent system"""
    from terminal_env import TerminalEnvironment
    from config import TerminalConfig
    
    # Create terminal config and environment
    config = TerminalConfig()
    env = TerminalEnvironment(terminal_config=config)
    
    # Calculate the CORRECT state dimension
    state_dim = calculate_state_dim(env)
    print(f"Calculated state dimension: {state_dim}")
    
    # Extract action dimensions
    action_dims = extract_action_dims(env)
    
    # Create agent system with the correct state dim
    agent_system = TerminalAgentSystem(state_dim, action_dims)
    
    # Train agent
    episode_rewards = train_qmix_agent(env, agent_system, num_episodes=2000)
    
    return agent_system


def calculate_state_dim(env):
    """Calculate the dimension of the flattened state vector"""
    obs, _ = env.reset()
    
    # Extract dimensions from observation
    state_dim = 0
    
    # Crane positions (num_cranes x 2)
    state_dim += obs['crane_positions'].size
    
    # Crane available times (num_cranes)
    state_dim += obs['crane_available_times'].size
    
    # Current time (1)
    state_dim += obs['current_time'].size
    
    # Yard state (rows x slots x tiers)
    state_dim += obs['yard_state'].size
    
    # Parking status (num_parking_spots)
    state_dim += obs['parking_status'].size
    
    # Rail status (num_tracks x num_slots)
    state_dim += obs['rail_status'].size
    
    # Queue sizes (2)
    state_dim += obs['queue_sizes'].size
    
    return state_dim


def extract_action_dims(env):
    """Extract action dimensions for each operation type"""
    obs, _ = env.reset()
    
    action_dims = {}
    
    # Crane movements (one role per crane)
    num_cranes = obs['action_mask']['crane_movement'].shape[0]
    for crane_idx in range(num_cranes):
        # Flatten source x destination into a single dimension
        crane_mask = obs['action_mask']['crane_movement'][crane_idx]
        action_dims[f"crane_{crane_idx}"] = crane_mask.size
    
    # Truck parking
    truck_mask = obs['action_mask']['truck_parking']
    action_dims['truck_parking'] = truck_mask.size
    
    return action_dims

def setup_curriculum_learning(env, max_trucks=None, max_trains=None):
    """
    Configure the environment for curriculum learning with limited vehicles
    
    Args:
        env: The terminal environment
        max_trucks: Maximum number of trucks allowed per day (None for unlimited)
        max_trains: Maximum number of trains allowed per day (None for unlimited)
    """
    # Store original methods
    original_schedule_trucks = env._schedule_trucks_for_existing_containers
    original_process_vehicle_arrivals = env._process_vehicle_arrivals
    
    # Track daily vehicle counts
    env.daily_truck_count = 0
    env.daily_train_count = 0
    env.last_sim_day = 0
    
    # Override truck scheduling
    def limited_schedule_trucks(*args, **kwargs):
        if max_trucks is None:
            return original_schedule_trucks(*args, **kwargs)
        
        # Count scheduled trucks
        truck_count = 0
        result = original_schedule_trucks(*args, **kwargs)
        
        # Limit by removing excess trucks from the queue
        if max_trucks > 0:
            # Get all scheduled trucks
            trucks = list(env.truck_queue.scheduled_arrivals)
            if len(trucks) > max_trucks:
                # Keep only max_trucks
                env.truck_queue.scheduled_arrivals = trucks[:max_trucks]
        
        return result
    
    # Override vehicle arrival processing
    def limited_process_vehicle_arrivals(time_advanced):
        # Reset counts when a new day starts
        sim_day = int(env.current_simulation_time / 86400)  # 86400 seconds in a day
        if sim_day > env.last_sim_day:
            env.daily_truck_count = 0
            env.daily_train_count = 0
            env.last_sim_day = sim_day
        
        # Process train arrivals with limit
        if max_trains is not None:
            # Only allow more trains if under the limit
            can_add_trains = env.daily_train_count < max_trains
            
            # Count and limit trains
            train_arrivals_before = len(env.train_queue.scheduled_arrivals)
            original_process_vehicle_arrivals(time_advanced)
            train_arrivals_after = len(env.train_queue.scheduled_arrivals)
            
            new_trains = train_arrivals_after - train_arrivals_before
            if new_trains > 0:
                env.daily_train_count += new_trains
                
                # Remove excess trains if over limit
                if env.daily_train_count > max_trains:
                    excess = env.daily_train_count - max_trains
                    for _ in range(excess):
                        if env.train_queue.scheduled_arrivals:
                            env.train_queue.scheduled_arrivals.pop()
                    env.daily_train_count = max_trains
        else:
            # No limit on trains
            original_process_vehicle_arrivals(time_advanced)
        
        # Limit truck arrivals
        if max_trucks is not None:
            # Remove any trucks that would exceed the limit
            for i, (time, truck) in reversed(list(enumerate(env.truck_queue.scheduled_arrivals))):
                if env.daily_truck_count >= max_trucks:
                    env.truck_queue.scheduled_arrivals.pop(i)
                else:
                    env.daily_truck_count += 1
    
    # Apply the overrides
    env._schedule_trucks_for_existing_containers = limited_schedule_trucks
    env._process_vehicle_arrivals = limited_process_vehicle_arrivals

# Define the curriculum
def curriculum_learning_schedule(num_episodes):
    """
    Create a curriculum learning schedule that gradually increases difficulty
    
    Args:
        num_episodes: Total number of episodes in training
        
    Returns:
        List of (episode_start, max_trucks, max_trains) tuples defining the curriculum
    """
    # Start with easy scenarios and gradually increase difficulty
    curriculum = [
        (0, 5, 1),         # Starting level: 5 trucks, 1 train
        (num_episodes//6, 10, 2),     # Level 2: 10 trucks, 2 trains
        (num_episodes//3, 15, 3),     # Level 3: 15 trucks, 3 trains
        (num_episodes//2, 20, 4),     # Level 4: 20 trucks, 4 trains
        (num_episodes*2//3, 30, 5),   # Level 5: 30 trucks, 5 trains
        (num_episodes*5//6, None, None)  # Final level: Unlimited
    ]
    return curriculum
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Train the QMIX agent system
    agent_system = run_qmix_training()