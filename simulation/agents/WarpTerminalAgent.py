import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import copy
import warp as wp
from typing import Dict, Tuple, List, Optional, Any, Union

# Define experience tuple structure for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'action_type', 'reward', 'next_state', 'done', 'action_mask'])

class ReplayBuffer:
    """GPU-optimized experience replay buffer for storing and sampling transitions."""
    
    def __init__(self, capacity=100000, device='cuda'):
        """Initialize replay buffer with specified capacity and device."""
        self.buffer = deque(maxlen=capacity)
        self.state_dim = None
        self.device = device
    
    def add(self, state, action, action_type, reward, next_state, done, action_mask):
        """Add experience to the buffer."""
        # Only store experiences with consistent state dimensions
        if self.state_dim is None:
            self.state_dim = len(state)
        elif len(state) != self.state_dim:
            return  # Skip adding experiences with inconsistent dimensions
            
        experience = Experience(state, action, action_type, reward, next_state, done, action_mask)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Sample a batch of experiences randomly."""
        if len(self.buffer) < batch_size:
            return None
        
        batch = random.sample(self.buffer, batch_size)
        
        # Unpack the batch
        states = np.array([exp.state for exp in batch])
        actions = np.array([exp.action for exp in batch])
        action_types = np.array([exp.action_type for exp in batch])
        rewards = np.array([exp.reward for exp in batch])
        next_states = np.array([exp.next_state for exp in batch if exp.next_state is not None])
        dones = np.array([exp.done for exp in batch])
        action_masks = [exp.action_mask for exp in batch]
        
        return states, actions, action_types, rewards, next_states, dones, action_masks
    
    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer and reset state dimension."""
        self.buffer.clear()
        self.state_dim = None


class WarpTerminalAgent(nn.Module):
    """
    Warp-accelerated agent for container terminal operations with specialized heads.
    Incorporates fast parallel operations for decision making and learning.
    """
    
    def __init__(self, 
                 state_dim, 
                 action_dims, 
                 hidden_dims=[256, 256],  # List of hidden layer dimensions for backbone
                 head_dims=[128],         # List of hidden layer dimensions for each head
                 dropout_rate=0.0,        # Dropout probability (0 to disable)
                 use_batch_norm=False,    # Whether to use batch normalization
                 learning_rate=0.0005,    # Learning rate for optimizer
                 device='cuda'):
        """
        Initialize the Warp-accelerated agent with configurable architecture.
        
        Args:
            state_dim: Dimension of flattened state
            action_dims: Dictionary of action dimensions
            hidden_dims: List of hidden layer dimensions for backbone
            head_dims: List of hidden layer dimensions for each head
            dropout_rate: Dropout probability (0 to disable)
            use_batch_norm: Whether to use batch normalization
            learning_rate: Learning rate for optimizer
            device: Computation device (CPU/GPU)
        """
        super(WarpTerminalAgent, self).__init__()
        
        self.device = device
        self.action_dims = action_dims
        self.state_dim = state_dim
        
        # Build dynamic backbone network
        backbone_layers = []
        input_dim = state_dim
        
        for i, dim in enumerate(hidden_dims):
            backbone_layers.append(nn.Linear(input_dim, dim))
            
            if use_batch_norm:
                backbone_layers.append(nn.BatchNorm1d(dim))
                
            backbone_layers.append(nn.ReLU())
            
            if dropout_rate > 0:
                backbone_layers.append(nn.Dropout(dropout_rate))
                
            input_dim = dim
        
        self.backbone = nn.Sequential(*backbone_layers)
        
        # Output dimension from backbone is the last hidden dimension
        backbone_output_dim = hidden_dims[-1]
        
        # Build transfer operations head
        transfer_layers = []
        input_dim = backbone_output_dim
        
        for dim in head_dims:
            transfer_layers.append(nn.Linear(input_dim, dim))
            
            if use_batch_norm:
                transfer_layers.append(nn.BatchNorm1d(dim))
                
            transfer_layers.append(nn.ReLU())
            
            if dropout_rate > 0:
                transfer_layers.append(nn.Dropout(dropout_rate))
                
            input_dim = dim
        
        # Add final output layer for transfer head
        transfer_layers.append(nn.Linear(input_dim, 1))
        self.transfer_head = nn.Sequential(*transfer_layers)
        
        # Build yard optimization head
        yard_layers = []
        input_dim = backbone_output_dim
        
        for dim in head_dims:
            yard_layers.append(nn.Linear(input_dim, dim))
            
            if use_batch_norm:
                yard_layers.append(nn.BatchNorm1d(dim))
                
            yard_layers.append(nn.ReLU())
            
            if dropout_rate > 0:
                yard_layers.append(nn.Dropout(dropout_rate))
                
            input_dim = dim
        
        # Add final output layer for yard head
        yard_layers.append(nn.Linear(input_dim, 1))
        self.yard_head = nn.Sequential(*yard_layers)
        
        # Move networks to device
        self.to(device)
        
        # Create target network
        self.target_network = copy.deepcopy(self)
        self.target_network.eval()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Define loss function
        self.criterion = nn.MSELoss()
        
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(capacity=100000, device=device)
        
        # Training parameters
        self.batch_size = 256
        self.gamma = 0.99          # Discount factor
        self.tau = 0.001           # Soft update parameter
        self.epsilon = 1.0         # Start with full exploration
        self.epsilon_decay = 0.998 # Slower decay
        self.epsilon_min = 0.1     # Higher minimum exploration
        self.update_every = 4      # Update target network every N steps
        self.step_count = 0
        
        # Register Warp kernels
        self._register_kernels()
        
        # Position type detection
        self.storage_position_threshold = int(0.6 * state_dim)
        
        # Log the architecture
        self._log_architecture()

    def _log_architecture(self):
        """Log the network architecture for debugging."""
        print(f"Initialized WarpTerminalAgent with:")
        print(f"- State dimension: {self.state_dim}")
        print(f"- Backbone: {self.backbone}")
        print(f"- Transfer head: {self.transfer_head}")
        print(f"- Yard head: {self.yard_head}")
        print(f"- Device: {self.device}")
    
    def _register_kernels(self):
        """Register Warp kernels for agent operations."""
        # Action selection kernels
        wp.register_kernel(self._kernel_filter_valid_actions)
        wp.register_kernel(self._kernel_calculate_action_scores)
    
    @staticmethod
    @wp.kernel
    def _kernel_filter_valid_actions(action_mask: wp.array,
                                  valid_actions: wp.array,
                                  valid_count: wp.array,
                                  action_type: int):
        """Kernel to filter valid actions from a mask."""
        idx = wp.tid()
        
        # Get appropriate dimensions based on action type
        dim1, dim2, dim3 = 0, 0, 0
        if action_type == 0:  # Crane Movement
            dim1, dim2, dim3 = action_mask.shape[0], action_mask.shape[1], action_mask.shape[2]
        elif action_type == 1:  # Truck Parking
            dim1, dim2 = action_mask.shape[0], action_mask.shape[1]
            dim3 = 1  # Add a dimension for consistency
        elif action_type == 2:  # Terminal Truck
            dim1, dim2, dim3 = action_mask.shape[0], action_mask.shape[1], action_mask.shape[2]
        
        # Extract 3D index from flattened index
        if action_type == 1:  # 2D case
            if idx >= dim1 * dim2:
                return
                
            i = idx // dim2
            j = idx % dim2
            k = 0
        else:  # 3D case
            if idx >= dim1 * dim2 * dim3:
                return
                
            i = idx // (dim2 * dim3)
            j = (idx % (dim2 * dim3)) // dim3
            k = idx % dim3
        
        # Check if action is valid
        if action_mask[i, j, k if action_type != 1 else 0] == 1:
            # Atomically increment the count and get the position
            pos = wp.atomic_add(valid_count, 0, 1)
            
            # Store the flattened index
            valid_actions[pos] = idx
    
    @staticmethod
    @wp.kernel
    def _kernel_calculate_action_scores(q_values: wp.array,
                                     source_positions: wp.array,
                                     dest_positions: wp.array,
                                     stack_heights: wp.array,
                                     action_scores: wp.array,
                                     valid_actions: wp.array,
                                     valid_count: int,
                                     action_type: int):
        """Kernel to calculate scores for valid actions."""
        idx = wp.tid()
        
        if idx >= valid_count:
            return
            
        # Get the flattened action index
        action_idx = valid_actions[idx]
        
        # Calculate action score based on Q-value and heuristics
        score = q_values[action_idx]
        
        # Apply heuristics based on action type
        if action_type == 0:  # Crane Movement
            # Extract source and destination positions
            # (This is a simplification - in a real implementation, 
            # we would decompose the flattened index to get i,j,k)
            src_idx = (action_idx // 100) % 1000  # Simplified extraction
            dst_idx = action_idx % 100            # Simplified extraction
            
            # Get source and destination coordinates
            src_x, src_y = source_positions[src_idx, 0], source_positions[src_idx, 1]
            dst_x, dst_y = dest_positions[dst_idx, 0], dest_positions[dst_idx, 1]
            
            # Calculate distance-based adjustment (closer is better)
            dist = wp.sqrt((dst_x - src_x)**2 + (dst_y - src_y)**2)
            
            # Adjust score based on distance (small penalty for longer distances)
            score -= 0.01 * dist
            
            # Adjust score based on destination stack height (prefer shorter stacks)
            dest_stack_height = stack_heights[dst_idx]
            score -= 0.1 * dest_stack_height
        
        # Store the calculated score
        action_scores[idx] = score
    
    def select_action(self, state, action_masks, env, epsilon=None):
        """
        Select an action using epsilon-greedy policy with Warp acceleration.
        
        Args:
            state: Current state observation
            action_masks: Dictionary of valid action masks for each action type
            env: Terminal environment (for additional state evaluation)
            epsilon: Exploration rate (uses self.epsilon if None)
            
        Returns:
            Tuple of (action, action_type, flattened_state)
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        # Extract masks for different movement types
        transfer_mask = self._extract_transfer_mask(action_masks)
        
        # Check if pre-marshalling is needed based on yard state
        need_premarshalling = self._evaluate_need_for_premarshalling(env)
        if need_premarshalling:
            yard_mask = self._extract_yard_mask(action_masks)
        else:
            yard_mask = None
        
        # Determine which action types are available
        available_types = []
        
        if transfer_mask is not None and transfer_mask.sum() > 0:
            available_types.append(0)  # Transfer movements (crane)
            
        if yard_mask is not None and yard_mask.sum() > 0:
            available_types.append(1)  # Yard optimization (pre-marshalling)
        
        # If no actions available, return None to signal waiting
        if not available_types:
            return None, None, None
        
        # With probability epsilon, choose a random action
        if random.random() < epsilon:
            # Try to find a valid action
            while available_types:
                # Choose random action type
                action_type_idx = random.randint(0, len(available_types) - 1)
                action_type = available_types[action_type_idx]
                
                # Get appropriate mask for this type
                if action_type == 0:  # Transfer
                    mask = transfer_mask
                    action_key = 'crane_movement'
                else:  # Yard optimization
                    mask = yard_mask
                    action_key = 'crane_movement'  # Both use crane movement
                
                # Find valid actions for this type
                valid_actions = np.argwhere(mask == 1)
                if valid_actions.size > 0:
                    # Randomly select among valid actions
                    action_idx = random.randint(0, len(valid_actions) - 1)
                    action = tuple(valid_actions[action_idx])
                    
                    # Create action dictionary
                    action_dict = {
                        'action_type': 0,  # All types use crane movement in env
                        'crane_movement': np.array(action, dtype=np.int64),
                        'truck_parking': np.zeros(2, dtype=np.int64),  # Placeholder
                        'terminal_truck': np.zeros(3, dtype=np.int64)  # Placeholder
                    }
                    
                    break
                else:
                    # No valid actions for this type, remove it and try another
                    available_types.pop(action_type_idx)
                    continue
            
            # If we exhausted all types without finding a valid action
            if 'action_dict' not in locals():
                return None, None, None
        
        else:
            # Greedy action selection
            try:
                # Convert state to tensor for network
                flat_state = self._flatten_state(state)
                state_tensor = torch.FloatTensor(flat_state).unsqueeze(0).to(self.device)
                
                # Initialize selection variables
                best_action_type = None
                best_action = None
                best_value = float('-inf')
                
                # Extract features from backbone
                with torch.no_grad():
                    features = self.backbone(state_tensor)
                
                # Evaluate value for each available action type
                for action_type in available_types:
                    # Get appropriate mask and head for this action type
                    if action_type == 0:  # Transfer movements
                        mask = transfer_mask
                        head = self.transfer_head
                        action_key = 'crane_movement'
                    else:  # Yard optimization
                        mask = yard_mask
                        head = self.yard_head
                        action_key = 'crane_movement'  # Both use crane movement
                    
                    # Get valid actions for this type
                    valid_indices = np.argwhere(mask == 1)
                    
                    if len(valid_indices) == 0:
                        continue
                    
                    # Calculate Q-values for all valid actions
                    for idx in valid_indices:
                        # For each valid action, get Q-value prediction
                        with torch.no_grad():
                            # Use the head for this action type
                            q_value = head(features).item()
                            
                            # For pre-marshalling, apply additional evaluation
                            if action_type == 1:  # Yard optimization
                                # Evaluate if this specific pre-marshalling action improves yard
                                source_pos = self._idx_to_position(idx[1], env)
                                dest_pos = self._idx_to_position(idx[2], env)
                                
                                # Get yard improvement estimate
                                improvement = self._estimate_move_improvement(env, source_pos, dest_pos)
                                
                                # Boost or penalize based on estimated improvement
                                q_value = q_value * (1.0 + improvement * 0.5)
                        
                        # Update best action if this one has higher value
                        if q_value > best_value:
                            best_value = q_value
                            best_action_type = action_type
                            best_action = tuple(idx)
                
                # If we found a valid best action, use it
                if best_action is not None:
                    action_type = best_action_type
                    
                    # Create action dictionary
                    action_dict = {
                        'action_type': 0,  # All types use crane movement in env
                        'crane_movement': np.array(best_action, dtype=np.int64),
                        'truck_parking': np.zeros(2, dtype=np.int64),  # Placeholder
                        'terminal_truck': np.zeros(3, dtype=np.int64)  # Placeholder
                    }
                else:
                    # Fallback to random selection if greedy failed
                    print("WARNING: Greedy selection failed, falling back to random")
                    action_type = available_types[0]
                    
                    if action_type == 0:  # Transfer
                        mask = transfer_mask
                    else:  # Yard optimization
                        mask = yard_mask
                    
                    valid_actions = np.argwhere(mask == 1)
                    if len(valid_actions) > 0:
                        action_idx = random.randint(0, len(valid_actions) - 1)
                        action = tuple(valid_actions[action_idx])
                        
                        # Create action dictionary
                        action_dict = {
                            'action_type': 0,  # All types use crane movement in env
                            'crane_movement': np.array(action, dtype=np.int64),
                            'truck_parking': np.zeros(2, dtype=np.int64),  # Placeholder
                            'terminal_truck': np.zeros(3, dtype=np.int64)  # Placeholder
                        }
                    else:
                        return None, None, None
            
            except Exception as e:
                print(f"Error in greedy action selection: {e}")
                import traceback
                traceback.print_exc()
                
                # Fallback to random selection
                action_type = available_types[0]
                
                if action_type == 0:  # Transfer
                    mask = transfer_mask
                else:  # Yard optimization
                    mask = yard_mask
                
                valid_actions = np.argwhere(mask == 1)
                if len(valid_actions) > 0:
                    action_idx = random.randint(0, len(valid_actions) - 1)
                    action = tuple(valid_actions[action_idx])
                    
                    # Create action dictionary
                    action_dict = {
                        'action_type': 0,  # All types use crane movement in env
                        'crane_movement': np.array(action, dtype=np.int64),
                        'truck_parking': np.zeros(2, dtype=np.int64),  # Placeholder
                        'terminal_truck': np.zeros(3, dtype=np.int64)  # Placeholder
                    }
                else:
                    return None, None, None
        
        # Return action and flattened state
        flat_state = self._flatten_state(state)
        return action_dict, action_type, flat_state
    
    def store_experience(self, state, action, action_type, reward, next_state, done, action_mask):
        """Store experience in replay buffer."""
        # Clip extreme rewards to prevent unstable learning
        reward = np.clip(reward, -10, 10)
        
        # Only store if state dimensions match
        if state is not None and next_state is not None and len(state) == len(next_state):
            self.replay_buffer.add(state, action, action_type, reward, next_state, done, action_mask)
    
    def update(self):
        """Update the Q-networks using experiences from the replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return None
            
        # Sample a batch of experiences
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return None
            
        # Unpack the batch
        states, actions, action_types, rewards, next_states, dones, action_masks = batch
        
        # Convert batch data to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        action_types = torch.LongTensor(action_types).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Process non-terminal next states
        non_terminal_mask = ~dones.bool()
        non_terminal_next_states = None
        if non_terminal_mask.any():
            non_terminal_indices = torch.where(non_terminal_mask)[0]
            non_terminal_next_states_np = np.array([next_states[i] for i in non_terminal_indices.cpu().numpy()])
            non_terminal_next_states = torch.FloatTensor(non_terminal_next_states_np).to(self.device)
        
        # Extract features from backbone
        features = self.backbone(states)
        
        # Process by action type (transfer or yard)
        transfer_mask = (action_types == 0)
        yard_mask = (action_types == 1)
        
        # Initialize loss
        loss = 0
        
        # Process transfer operations
        if transfer_mask.any():
            # Get transfer indices and features
            transfer_indices = torch.where(transfer_mask)[0]
            transfer_states = states[transfer_indices]
            transfer_features = self.backbone(transfer_states)
            transfer_q = self.transfer_head(transfer_features).squeeze(-1)
            
            # Get rewards for transfer operations
            transfer_rewards = rewards[transfer_indices]
            
            # Compute transfer targets
            transfer_targets = transfer_rewards.clone()
            
            # Add future discounted rewards for non-terminal states
            if non_terminal_mask.any() and transfer_mask.any():
                # Find indices of transfer operations in non-terminal states
                transfer_next_mask = torch.zeros_like(non_terminal_mask)
                for i, idx in enumerate(transfer_indices):
                    if non_terminal_mask[idx]:
                        transfer_next_mask[idx] = True
                
                if transfer_next_mask.any():
                    # Get next states for transfer operations
                    transfer_next_indices = torch.where(transfer_next_mask)[0]
                    transfer_next_states_idx = torch.tensor([torch.where(non_terminal_indices == idx)[0][0] 
                                                         for idx in transfer_next_indices if idx in non_terminal_indices])
                    
                    if len(transfer_next_states_idx) > 0:
                        transfer_next_states = non_terminal_next_states[transfer_next_states_idx]
                        
                        with torch.no_grad():
                            # Get features from target network
                            next_features = self.target_network.backbone(transfer_next_states)
                            
                            # Get Q-values from both heads
                            next_transfer_q = self.target_network.transfer_head(next_features).squeeze(-1)
                            next_yard_q = self.target_network.yard_head(next_features).squeeze(-1)
                            
                            # Take maximum Q-value
                            next_q = torch.max(next_transfer_q, next_yard_q)
                            
                            # Add future rewards
                            for i, idx in enumerate(transfer_next_indices):
                                if idx in non_terminal_indices:
                                    idx_in_transfer = (transfer_indices == idx).nonzero(as_tuple=True)[0]
                                    transfer_targets[idx_in_transfer] += self.gamma * next_q[i]
            
            # Compute transfer loss
            transfer_loss = self.criterion(transfer_q, transfer_targets)
            loss += transfer_loss
        
        # Process yard operations
        if yard_mask.any():
            # Get yard indices and features
            yard_indices = torch.where(yard_mask)[0]
            yard_states = states[yard_indices]
            yard_features = self.backbone(yard_states)
            yard_q = self.yard_head(yard_features).squeeze(-1)
            
            # Get rewards for yard operations
            yard_rewards = rewards[yard_indices]
            
            # Compute yard targets
            yard_targets = yard_rewards.clone()
            
            # Add future discounted rewards for non-terminal states
            if non_terminal_mask.any() and yard_mask.any():
                # Find indices of yard operations in non-terminal states
                yard_next_mask = torch.zeros_like(non_terminal_mask)
                for i, idx in enumerate(yard_indices):
                    if non_terminal_mask[idx]:
                        yard_next_mask[idx] = True
                
                if yard_next_mask.any():
                    # Get next states for yard operations
                    yard_next_indices = torch.where(yard_next_mask)[0]
                    yard_next_states_idx = torch.tensor([torch.where(non_terminal_indices == idx)[0][0] 
                                                     for idx in yard_next_indices if idx in non_terminal_indices])
                    
                    if len(yard_next_states_idx) > 0:
                        yard_next_states = non_terminal_next_states[yard_next_states_idx]
                        
                        with torch.no_grad():
                            # Get features from target network
                            next_features = self.target_network.backbone(yard_next_states)
                            
                            # Get Q-values from both heads
                            next_transfer_q = self.target_network.transfer_head(next_features).squeeze(-1)
                            next_yard_q = self.target_network.yard_head(next_features).squeeze(-1)
                            
                            # Take maximum Q-value
                            next_q = torch.max(next_transfer_q, next_yard_q)
                            
                            # Add future rewards
                            for i, idx in enumerate(yard_next_indices):
                                if idx in non_terminal_indices:
                                    idx_in_yard = (yard_indices == idx).nonzero(as_tuple=True)[0]
                                    yard_targets[idx_in_yard] += self.gamma * next_q[i]
            
            # Compute yard loss
            yard_loss = self.criterion(yard_q, yard_targets)
            loss += yard_loss
        
        # If we computed no loss, return None
        if loss == 0:
            return None
        
        # Optimize the networks
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Perform soft update of target network
        self._soft_update()
        
        # Increment step counter
        self.step_count += 1
        
        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def _soft_update(self):
        """Soft update of target network parameters."""
        for target_param, param in zip(self.target_network.parameters(), self.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def _flatten_state(self, state):
        """Flatten state dictionary to vector."""
        # Handle tensors properly
        def process_component(component):
            if isinstance(component, torch.Tensor):
                return component.cpu().numpy().flatten()
            elif isinstance(component, np.ndarray):
                return component.flatten()
            else:
                try:
                    return np.array(component).flatten()
                except:
                    return np.array([])
        
        # Extract relevant features and flatten
        try:
            crane_positions = process_component(state['crane_positions'])
            crane_available_times = process_component(state['crane_available_times'])
            terminal_truck_available_times = process_component(state.get('terminal_truck_available_times', [0]))
            current_time = process_component(state['current_time'])
            yard_state = process_component(state['yard_state'])
            parking_status = process_component(state['parking_status'])
            rail_status = process_component(state['rail_status'])
            queue_sizes = process_component(state['queue_sizes'])
            
            # Concatenate all features
            flat_state = np.concatenate([
                crane_positions, 
                crane_available_times,
                terminal_truck_available_times,
                current_time, 
                yard_state, 
                parking_status, 
                rail_status, 
                queue_sizes
            ])
            
            return flat_state
        except Exception as e:
            print(f"Error flattening state: {e}")
            # Return a zero vector as fallback
            return np.zeros(self.state_dim)
    
    def _extract_transfer_mask(self, action_masks):
        """Extract the crane transfer mask from the action masks."""
        # Get the crane movement action mask
        crane_mask = action_masks['crane_movement']
        
        # If it's a torch tensor, convert to numpy safely
        if isinstance(crane_mask, torch.Tensor):
            crane_mask = crane_mask.detach().cpu().numpy()
        
        # Create a tensor of the same shape but filled with zeros
        transfer_mask = np.zeros_like(crane_mask)
        
        # The transfer operations are from any position to a rail or truck position
        # Iterate through each crane, source and destination
        num_cranes, num_sources, num_destinations = crane_mask.shape
        
        for crane_idx in range(num_cranes):
            for src_idx in range(num_sources):
                for dst_idx in range(num_destinations):
                    # If this is a valid action and destination is a pickup zone
                    if crane_mask[crane_idx, src_idx, dst_idx] > 0:
                        transfer_mask[crane_idx, src_idx, dst_idx] = 1
        
        return transfer_mask
    
    def _extract_yard_mask(self, action_masks):
        """Extract the yard operations mask from the action masks."""
        # Get the crane movement action mask
        crane_mask = action_masks['crane_movement']
        
        # If it's a torch tensor, convert to numpy safely
        if isinstance(crane_mask, torch.Tensor):
            crane_mask = crane_mask.detach().cpu().numpy()
        
        # Create a tensor of the same shape but filled with zeros
        yard_mask = np.zeros_like(crane_mask)
        
        # The yard operations are from storage to storage
        # Iterate through each crane, source and destination
        num_cranes, num_sources, num_destinations = crane_mask.shape
        
        for crane_idx in range(num_cranes):
            for src_idx in range(num_sources):
                for dst_idx in range(num_destinations):
                    # If this is a valid action and it's a storage-to-storage move
                    if crane_mask[crane_idx, src_idx, dst_idx] > 0:
                        yard_mask[crane_idx, src_idx, dst_idx] = 1
        
        return yard_mask
    
    def _evaluate_need_for_premarshalling(self, env):
        """
        Determine if pre-marshalling is needed based on yard state.
        
        Args:
            env: Terminal environment
            
        Returns:
            Boolean indicating if pre-marshalling is needed
        """
        # Get all stacks in yard
        yard = env.storage_yard
        
        # Count problematic stacks (higher priority below lower)
        problem_stacks = 0
        total_stacks = 0
        
        if not hasattr(yard, 'row_names'):
            return False
        
        for row in yard.row_names:
            for bay in range(1, yard.num_bays + 1):
                position = f"{row}{bay}"
                containers = yard.get_containers_at_position(position)
                
                if isinstance(containers, dict) and len(containers) > 1:
                    total_stacks += 1
                    # Check if priorities are ordered correctly (higher on top)
                    tiers = sorted(containers.keys())
                    
                    # Try to get priorities, but be defensive
                    try:
                        priorities = []
                        for tier in tiers:
                            container = containers[tier]
                            if hasattr(container, 'priority'):
                                priorities.append(container.priority)
                            elif isinstance(container, (int, np.integer)):
                                # If it's just a container index, get info from the registry
                                container_idx = container
                                if hasattr(env, 'terminal_state') and hasattr(env.terminal_state, 'container_properties'):
                                    priority = env.terminal_state.container_properties[container_idx, 2].numpy()
                                    priorities.append(priority)
                                else:
                                    priorities.append(0)  # Default
                            else:
                                priorities.append(0)  # Default
                        
                        # Check if priorities are in descending order
                        if len(priorities) > 1:
                            if not all(priorities[i] >= priorities[i+1] for i in range(len(priorities)-1)):
                                problem_stacks += 1
                    except Exception as e:
                        # We hit an error trying to analyze this stack - consider it not problematic
                        print(f"Error analyzing stack at {position}: {e}")
                        continue
        
        # Only allow pre-marshalling if enough problematic stacks exist
        needs_premarshalling = problem_stacks > 0 and problem_stacks / max(1, total_stacks) > 0.2
        
        return needs_premarshalling
    
    def _estimate_move_improvement(self, env, source_position, dest_position):
        """
        Estimate how much a pre-marshalling move will improve yard state.
        
        Args:
            env: Terminal environment
            source_position: Source position string
            dest_position: Destination position string
            
        Returns:
            Estimated improvement (positive is better)
        """
        # Defensive coding - ensure yard exists
        if not hasattr(env, 'storage_yard'):
            return 0.0
            
        yard = env.storage_yard
        
        # Get containers at source and destination
        try:
            source_containers = yard.get_containers_at_position(source_position)
            dest_containers = yard.get_containers_at_position(dest_position)
        except Exception as e:
            print(f"Error getting containers: {e}")
            return 0.0
        
        # Defensively handle different return types
        if isinstance(source_containers, (int, np.integer)) or (isinstance(source_containers, dict) and not source_containers):
            return -5.0  # No container to move
        
        # Get top container at source
        if isinstance(source_containers, dict):
            if not source_containers:
                return -5.0  # No container to move
                
            source_tier = max(source_containers.keys())
            top_container = source_containers[source_tier]
        else:
            # If it's a list, array, etc., just take the last one
            top_container = source_containers[-1] if source_containers else None
            
        if top_container is None:
            return -5.0  # No container
        
        # Estimate improvement based on simple heuristics
        improvement = 0.0
        
        # Check destination stack height
        dest_height = len(dest_containers) if isinstance(dest_containers, dict) else len(dest_containers) if isinstance(dest_containers, (list, tuple, np.ndarray)) else 0
        if dest_height >= 4:
            improvement -= 3.0  # Penalty for creating tall stacks
        
        # Check if destination has a container
        if dest_containers and isinstance(dest_containers, dict) and dest_containers:
            dest_tier = max(dest_containers.keys())
            bottom_container = dest_containers[dest_tier]
            
            # Compare priorities if possible
            try:
                top_priority = getattr(top_container, 'priority', 0)
                if isinstance(top_container, (int, np.integer)):
                    # It's a container index - get priority from registry
                    if hasattr(env, 'terminal_state') and hasattr(env.terminal_state, 'container_properties'):
                        top_priority = env.terminal_state.container_properties[top_container, 2].numpy()
                
                bottom_priority = getattr(bottom_container, 'priority', 0)
                if isinstance(bottom_container, (int, np.integer)):
                    # It's a container index - get priority from registry
                    if hasattr(env, 'terminal_state') and hasattr(env.terminal_state, 'container_properties'):
                        bottom_priority = env.terminal_state.container_properties[bottom_container, 2].numpy()
                
                # Better to have lower priority containers below higher priority ones
                if top_priority >= bottom_priority:
                    improvement += 2.0
                else:
                    improvement -= 2.0
            except Exception as e:
                # Failed to compare priorities
                print(f"Error comparing priorities: {e}")
        
        # Check if freeing up a problematic stack at source
        if isinstance(source_containers, dict) and len(source_containers) > 1:
            try:
                # Get container below the top one
                source_below_tier = max([t for t in source_containers.keys() if t < source_tier])
                below_container = source_containers[source_below_tier]
                
                # Compare priorities if possible
                top_priority = getattr(top_container, 'priority', 0)
                if isinstance(top_container, (int, np.integer)):
                    # It's a container index - get priority from registry
                    if hasattr(env, 'terminal_state') and hasattr(env.terminal_state, 'container_properties'):
                        top_priority = env.terminal_state.container_properties[top_container, 2].numpy()
                
                below_priority = getattr(below_container, 'priority', 0)
                if isinstance(below_container, (int, np.integer)):
                    # It's a container index - get priority from registry
                    if hasattr(env, 'terminal_state') and hasattr(env.terminal_state, 'container_properties'):
                        below_priority = env.terminal_state.container_properties[below_container, 2].numpy()
                
                # If removing top improves priority order
                if top_priority < below_priority:
                    improvement += 3.0
            except Exception as e:
                # Failed to compare below container
                print(f"Error checking stack: {e}")
        
        return improvement
    
    def _idx_to_position(self, idx, env):
        """
        Convert position index to position string based on environment.
        
        Args:
            idx: Position index
            env: Terminal environment for mapping reference
            
        Returns:
            Position string
        """
        # Get mapping from environment if available
        if hasattr(env, 'idx_to_position') and idx in env.idx_to_position:
            return env.idx_to_position[idx]
        
        # Fallback approach if environment mapping not accessible
        if idx >= self.storage_position_threshold:
            # This is a storage position
            storage_idx = idx - self.storage_position_threshold
            
            # Calculate row and bay based on estimated dimensions
            if hasattr(env, 'storage_yard') and hasattr(env.storage_yard, 'num_bays'):
                estimated_bays_per_row = env.storage_yard.num_bays
            else:
                estimated_bays_per_row = 20  # Default
                
            row_idx = storage_idx // estimated_bays_per_row
            bay_idx = storage_idx % estimated_bays_per_row
            
            # Check if row_idx is in range
            if hasattr(env, 'storage_yard') and hasattr(env.storage_yard, 'row_names'):
                row_names = env.storage_yard.row_names
                if row_idx < len(row_names):
                    row_letter = row_names[row_idx]
                    return f"{row_letter}{bay_idx + 1}"
            
            # Fallback if index is out of range or no row names
            return f"S{idx}"
        else:
            # This is a rail or parking position
            rail_threshold = 100  # Estimate - adjust based on environment
            
            if idx < rail_threshold:
                # Rail position
                if hasattr(env, 'terminal') and hasattr(env.terminal, 'track_names'):
                    num_tracks = len(env.terminal.track_names)
                    slots_per_track = rail_threshold // max(1, num_tracks)
                    
                    track_idx = idx // slots_per_track
                    slot_idx = idx % slots_per_track
                    
                    if track_idx < num_tracks:
                        track_name = env.terminal.track_names[track_idx].lower()
                        return f"{track_name}_{slot_idx + 1}"
                
                # Fallback
                return f"r{idx}"
            else:
                # Parking position
                parking_idx = idx - rail_threshold
                return f"p_{parking_idx + 1}"
    
    def save(self, filepath):
        """Save model parameters."""
        torch.save({
            'state_dim': self.state_dim,
            'backbone': self.backbone.state_dict(),
            'transfer_head': self.transfer_head.state_dict(),
            'yard_head': self.yard_head.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'storage_position_threshold': self.storage_position_threshold
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Update state dimension if different
        if 'state_dim' in checkpoint and checkpoint['state_dim'] != self.state_dim:
            self.state_dim = checkpoint['state_dim']
            print(f"Updating state dimension to {self.state_dim}")
        
        # Load parameters
        self.backbone.load_state_dict(checkpoint['backbone'])
        self.transfer_head.load_state_dict(checkpoint['transfer_head'])
        self.yard_head.load_state_dict(checkpoint['yard_head'])
        
        if 'target_network' in checkpoint:
            self.target_network.load_state_dict(checkpoint['target_network'])
        else:
            # Backward compatibility for older checkpoints
            self.target_network.backbone.load_state_dict(checkpoint.get('target_backbone', checkpoint['backbone']))
            self.target_network.transfer_head.load_state_dict(checkpoint.get('target_transfer_head', checkpoint['transfer_head']))
            self.target_network.yard_head.load_state_dict(checkpoint.get('target_yard_head', checkpoint['yard_head']))
        
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        if 'epsilon' in checkpoint:
            self.epsilon = checkpoint['epsilon']
        
        if 'storage_position_threshold' in checkpoint:
            self.storage_position_threshold = checkpoint['storage_position_threshold']
        
        print(f"Model loaded from {filepath}")