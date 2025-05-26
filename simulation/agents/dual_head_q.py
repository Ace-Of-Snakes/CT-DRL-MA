import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple
import copy

# Define experience tuple structure
Experience = namedtuple('Experience', 
                        ['state', 'action', 'action_type', 'reward', 'next_state', 'done', 'action_mask'])

class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""
    
    def __init__(self, capacity=50000):
        """Initialize replay buffer with fixed capacity."""
        self.buffer = deque(maxlen=capacity)
        self.state_dim = None
    
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
        
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer and reset state dimension."""
        self.buffer.clear()
        self.state_dim = None


class OptimizedTerminalAgent(nn.Module):
    """
    Optimized agent for container terminal operations with specialized heads.
    Focuses on transfer operations and conditional pre-marshalling.
    """
    
    def __init__(self, 
                 state_dim, 
                 action_dims, 
                 hidden_dims=[256, 256],  # List of hidden layer dimensions for backbone
                 head_dims=[128],         # List of hidden layer dimensions for each head
                 dropout_rate=0.0,        # Dropout probability (0 to disable)
                 use_batch_norm=False,    # Whether to use batch normalization
                 learning_rate=0.0005,    # Learning rate for optimizer
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the agent with configurable architecture.
        
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
        super(OptimizedTerminalAgent, self).__init__()
        
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
        
        # Build yard optimization head (with same architecture as transfer head)
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
        
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(capacity=100000)
        
        # Training parameters
        self.batch_size = 128
        self.gamma = 0.99          # Discount factor
        self.tau = 0.001           # Soft update parameter - slower updates for stability
        self.epsilon = 1.0         # Start with full exploration
        self.epsilon_decay = 0.998 # Slower decay
        self.epsilon_min = 0.1     # Higher minimum exploration
        self.update_every = 4      # Update target network every N steps
        self.step_count = 0
        
        # Position type detection - will be set dynamically
        self.storage_position_threshold = int(0.6 * state_dim)
        
        # Log the architecture
        self._log_architecture()

    def _extract_transfer_mask(self, action_masks):
        """Extract mask for transfer actions (not storage-to-storage with N=1)."""
        if 'crane_movement' not in action_masks:
            return None
        
        crane_mask = action_masks['crane_movement']
        transfer_mask = np.zeros_like(crane_mask)
        
        # For transfer movements, we want:
        # 1. Any non-storage source to any destination (N=5)
        # 2. Storage source to non-storage destination (N=5)
        # 3. Exclude storage-to-storage with N=1 (that's pre-marshalling)
        
        for i in range(crane_mask.shape[0]):  # For each crane
            for j in range(crane_mask.shape[1]):  # For each source
                for k in range(crane_mask.shape[2]):  # For each destination
                    # Skip storage-to-storage moves within 1 bay (those are pre-marshalling)
                    if (j >= self.storage_position_threshold and 
                        k >= self.storage_position_threshold):
                        # This is storage-to-storage, check if it's within N=1 distance
                        # For now, we'll consider all storage-to-storage as pre-marshalling
                        # and exclude them from transfer mask
                        continue
                    
                    # This is a transfer movement (truck/train involved)
                    transfer_mask[i, j, k] = crane_mask[i, j, k]
        
        return transfer_mask

    def _extract_yard_mask(self, action_masks):
        """Extract mask for yard optimization actions (storage-to-storage with N=1 only)."""
        if 'crane_movement' not in action_masks:
            return None
        
        crane_mask = action_masks['crane_movement']
        yard_mask = np.zeros_like(crane_mask)
        
        # For yard optimization, we only want storage-to-storage moves within N=1
        for i in range(crane_mask.shape[0]):  # For each crane
            for j in range(crane_mask.shape[1]):  # For each source
                # Skip if source is not storage
                if j < self.storage_position_threshold:
                    continue
                    
                for k in range(crane_mask.shape[2]):  # For each destination
                    # Skip if destination is not storage
                    if k < self.storage_position_threshold:
                        continue
                    
                    # Both source and destination are storage
                    # This represents pre-marshalling moves (N=1)
                    yard_mask[i, j, k] = crane_mask[i, j, k]
        
        return yard_mask

    def _log_architecture(self):
        """Log the network architecture for debugging."""
        print(f"Initialized OptimizedTerminalAgent with:")
        print(f"- State dimension: {self.state_dim}")
        print(f"- Backbone: {self.backbone}")
        print(f"- Transfer head: {self.transfer_head}")
        print(f"- Yard head: {self.yard_head}")
        print(f"- Device: {self.device}")
    
    def select_action(self, state, action_masks, env, epsilon=None):
        """
        Select an action using epsilon-greedy policy with conditional pre-marshalling.
        
        Args:
            state: Current state observation
            action_masks: Dictionary of valid action masks for each action type
            env: Terminal environment (for yard state evaluation)
            epsilon: Exploration rate (uses self.epsilon if None)
            
        Returns:
            Tuple of (action, action_type, flattened_state)
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        # Extract masks for different movement types
        transfer_mask = self._extract_transfer_mask(action_masks)
        
        # Only consider pre-marshalling if needed based on yard state
        need_premarshalling = self._evaluate_need_for_premarshalling(env)
        if need_premarshalling:
            yard_mask = self._extract_yard_mask(action_masks)
            # print("Pre-marshalling enabled - yard state indicates optimization needed")
        else:
            yard_mask = None
            # print("Pre-marshalling disabled - yard state is optimal or near-optimal")
        
        # Determine which action types are available
        available_types = []
        
        if transfer_mask is not None and transfer_mask.sum() > 0:
            available_types.append(0)  # Transfer movements
            
        if yard_mask is not None and yard_mask.sum() > 0:
            available_types.append(1)  # Yard optimization (pre-marshalling)
        
        # If no actions available, return None to signal waiting
        if not available_types:
            # print("No valid actions available - signaling to wait")
            return None, None, None
        
        # With probability epsilon, choose a random action
        if random.random() < epsilon:
            # Try to find a valid action
            while available_types:
                # Choose random action type
                action_type_idx = random.randint(0, len(available_types) - 1)
                action_type = available_types[action_type_idx]
                
                # Get appropriate mask for this action type
                if action_type == 0:  # Transfer movements
                    mask = transfer_mask
                else:  # Yard optimization
                    mask = yard_mask
                
                # Find valid actions for this type
                valid_actions = np.argwhere(mask == 1)
                if valid_actions.size > 0:
                    # Randomly select among valid actions
                    action_idx = random.randint(0, len(valid_actions) - 1)
                    action = tuple(valid_actions[action_idx])
                    break
                else:
                    # No valid actions for this type, remove it and try another
                    available_types.pop(action_type_idx)
                    continue
            
            # If we exhausted all types without finding a valid action
            if 'action' not in locals():
                print("WARNING: No valid actions found despite available_types being non-empty")
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
                features = self.backbone(state_tensor)
                
                # Evaluate value for each available action type
                for action_type in available_types:
                    # Get appropriate mask and head for this action type
                    if action_type == 0:  # Transfer movements
                        mask = transfer_mask
                        head = self.transfer_head
                    else:  # Yard optimization
                        mask = yard_mask
                        head = self.yard_head
                    
                    # Get valid actions for this type
                    valid_indices = np.argwhere(mask == 1)
                    
                    if len(valid_indices) == 0:
                        continue
                    
                    # Calculate Q-values for all valid actions
                    for idx in valid_indices:
                        # For each valid action, create state-action pair
                        action_tensor = torch.LongTensor([idx[0], idx[1], idx[2]]).to(self.device)
                        
                        # Get Q-value prediction
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
                    action = best_action
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
                else:
                    return None, None, None
        
        # Map to environment action format
        # Both action types (transfer and yard optimization) use crane movement in the environment
        env_action = {
            'action_type': 0,  # Crane movement
            'crane_movement': np.array(action, dtype=np.int64),
            'truck_parking': np.zeros(2, dtype=np.int64)  # Placeholder
        }
        
        # Print debugging info
        # print(f"Selected action type: {'Transfer' if action_type == 0 else 'Yard optimization'}")
        # print(f"Action: {action}")
        
        # Return action and flattened state
        flat_state = self._flatten_state(state)
        return env_action, action_type, flat_state
    
    def store_experience(self, state, action, action_type, reward, next_state, done, action_mask):
        """Store experience in replay buffer."""
        # Clip extreme rewards to prevent unstable learning
        reward = np.clip(reward, -10, 10)
        
        # Only store if state dimensions match
        if state is not None and next_state is not None and len(state) == len(next_state):
            self.replay_buffer.add(state, action, action_type, reward, next_state, done, action_mask)
    
    def update(self):
        """Update Q-network from sampled experiences."""
        # Only update periodically
        self.step_count += 1
        if self.step_count % self.update_every != 0:
            return
        
        # Sample experiences from replay buffer
        experiences = self.replay_buffer.sample(self.batch_size)
        if experiences is None:
            return  # Not enough experiences yet
        
        try:
            # Check for consistent state dimensions
            state_dims = set(len(e.state) for e in experiences)
            if len(state_dims) > 1:
                print(f"WARNING: Mixed state dimensions in batch: {state_dims}")
                return  # Skip this update
            
            # Get state dimension from experiences
            state_dim = len(experiences[0].state)
            
            # Ensure network matches the state dimension
            if state_dim != self.state_dim:
                print(f"WARNING: State dimension changed from {self.state_dim} to {state_dim}")
                self.state_dim = state_dim
                # Recreate networks and optimizer with new dimension
                self = OptimizedTerminalAgent(state_dim, self.action_dims, self.device)
                return  # Skip this update
            
            # Unpack experiences
            states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
            actions = [e.action for e in experiences]
            action_types = [e.action_type for e in experiences]
            rewards = torch.FloatTensor([e.reward for e in experiences]).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
            dones = torch.FloatTensor([e.done for e in experiences]).unsqueeze(1).to(self.device)
            
            # Process by action type and compute loss separately for each
            loss = 0
            
            # Group experiences by action type
            type_indices = {
                0: [i for i, t in enumerate(action_types) if t == 0],  # Transfer operations
                1: [i for i, t in enumerate(action_types) if t == 1]   # Yard optimization
            }
            
            # Process each action type separately
            for action_type, indices in type_indices.items():
                if not indices:
                    continue  # Skip if no experiences of this type
                
                # Extract batch for this action type
                type_states = states[indices]
                type_actions = [actions[i] for i in indices]
                type_rewards = rewards[indices]
                type_next_states = next_states[indices]
                type_dones = dones[indices]
                
                # Get features from backbone
                current_features = self.backbone(type_states)
                
                # Select appropriate head
                if action_type == 0:  # Transfer
                    head = self.transfer_head
                    target_head = self.target_network.transfer_head
                else:  # Yard optimization
                    head = self.yard_head
                    target_head = self.target_network.yard_head
                
                # Get Q-values for actions taken
                q_values = head(current_features)
                
                # Compute target Q-values
                with torch.no_grad():
                    # Get features from target network backbone
                    next_features = self.target_network.backbone(type_next_states)
                    
                    # Get maximum Q-value for next state from target network
                    next_q_values = target_head(next_features)
                    max_next_q = next_q_values.max(1)[0].unsqueeze(1)
                    
                    # Compute target Q-value
                    target_q = type_rewards + (self.gamma * max_next_q * (1 - type_dones))
                
                # Compute loss - simplified without action selection
                # This approach treats the Q-values as a direct prediction of value
                type_loss = F.mse_loss(q_values, target_q)
                loss += type_loss
            
            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()
            
            # Soft update target network
            self._update_target_network()
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            return loss.item()
            
        except Exception as e:
            print(f"Error during update: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _update_target_network(self):
        """Soft update of target network."""
        for target_param, param in zip(self.target_network.parameters(), self.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def _flatten_state(self, state):
        """Flatten state dictionary to vector."""
        # Extract relevant features and flatten
        crane_positions = state['crane_positions'].flatten()
        crane_available_times = state['crane_available_times'].flatten()
        current_time = state['current_time'].flatten()
        yard_state = state['yard_state'].flatten()
        parking_status = state['parking_status'].flatten()
        rail_status = state['rail_status'].flatten()
        queue_sizes = state['queue_sizes'].flatten()
        
        # Concatenate all features
        flat_state = np.concatenate([
            crane_positions, 
            crane_available_times, 
            current_time, 
            yard_state, 
            parking_status, 
            rail_status, 
            queue_sizes
        ])
        
        return flat_state
    
    def _extract_transfer_mask(self, action_masks):
        """
        Extract mask for transfer actions (not storage-to-storage).
        
        Args:
            action_masks: Dictionary of action masks from environment
            
        Returns:
            Binary mask where 1 indicates valid transfer movement
        """
        if 'crane_movement' not in action_masks:
            return None
        
        crane_mask = action_masks['crane_movement']
        transfer_mask = np.zeros_like(crane_mask)
        
        # Determine threshold for storage positions if not already set
        if not hasattr(self, 'storage_position_threshold'):
            total_positions = crane_mask.shape[1]  # Number of source positions
            # Assume storage positions are in the latter part of position indices
            self.storage_position_threshold = int(total_positions * 0.6)
            print(f"Setting storage threshold to {self.storage_position_threshold} out of {total_positions} positions")
        
        # Extract mask for transfer movements (where at least one end is not storage)
        for i in range(crane_mask.shape[0]):  # For each crane
            for j in range(crane_mask.shape[1]):  # For each source
                for k in range(crane_mask.shape[2]):  # For each destination
                    # Skip storage-to-storage movements
                    if j >= self.storage_position_threshold and k >= self.storage_position_threshold:
                        continue
                        
                    # This is a transfer movement (not storage-to-storage)
                    transfer_mask[i, j, k] = crane_mask[i, j, k]
        
        return transfer_mask
    
    def _extract_yard_mask(self, action_masks):
        """
        Extract mask for yard optimization actions (storage-to-storage only).
        
        Args:
            action_masks: Dictionary of action masks from environment
            
        Returns:
            Binary mask where 1 indicates valid yard optimization movement
        """
        if 'crane_movement' not in action_masks:
            return None
        
        crane_mask = action_masks['crane_movement']
        yard_mask = np.zeros_like(crane_mask)
        
        # Determine threshold for storage positions if not already set
        if not hasattr(self, 'storage_position_threshold'):
            total_positions = crane_mask.shape[1]  # Number of source positions
            # Assume storage positions are in the latter part of position indices
            self.storage_position_threshold = int(total_positions * 0.6)
            print(f"Setting storage threshold to {self.storage_position_threshold} out of {total_positions} positions")
        
        # Extract mask for storage-to-storage movements
        for i in range(crane_mask.shape[0]):  # For each crane
            for j in range(crane_mask.shape[1]):  # For each source
                # Skip if source is not storage
                if j < self.storage_position_threshold:
                    continue
                    
                for k in range(crane_mask.shape[2]):  # For each destination
                    # Skip if destination is not storage
                    if k < self.storage_position_threshold:
                        continue
                    
                    # Both source and destination are storage - this is a yard optimization action
                    yard_mask[i, j, k] = crane_mask[i, j, k]
        
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
        
        for row in yard.row_names:
            for bay in range(1, yard.num_bays + 1):
                position = f"{row}{bay}"
                containers = yard.get_containers_at_position(position)
                
                if len(containers) > 1:
                    total_stacks += 1
                    # Check if priorities are ordered correctly (higher on top)
                    tiers = sorted(containers.keys())
                    priorities = [containers[tier].priority for tier in tiers]
                    
                    # Check if priorities are in descending order
                    if not all(priorities[i] >= priorities[i+1] for i in range(len(priorities)-1)):
                        problem_stacks += 1
        
        # Only allow pre-marshalling if enough problematic stacks exist
        needs_premarshalling = problem_stacks > 0 and problem_stacks / max(1, total_stacks) > 0.2
        
        # print(f"Yard state: {problem_stacks}/{total_stacks} problematic stacks - " +
            #   f"{'Needs' if needs_premarshalling else 'Does not need'} pre-marshalling")
        
        return needs_premarshalling
    
    def _calculate_yard_state_score(self, env):
        """
        Calculate a score for the current yard state (higher is better).
        
        Args:
            env: Terminal environment
            
        Returns:
            Numeric score of yard state quality
        """
        yard = env.storage_yard
        score = 0
        
        # Component 1: Priority-based stacking score
        for row in yard.row_names:
            for bay in range(1, yard.num_bays + 1):
                position = f"{row}{bay}"
                containers = yard.get_containers_at_position(position)
                
                if len(containers) > 1:
                    # Check if priorities are ordered correctly (higher on top)
                    tiers = sorted(containers.keys())
                    priorities = [containers[tier].priority for tier in tiers]
                    
                    # Score each container's position
                    for i in range(len(priorities) - 1):
                        if priorities[i] <= priorities[i+1]:  # Lower priority below higher
                            score += 1
                        else:
                            score -= 2  # Penalty for incorrect order
        
        # Component 2: Distribution score - avoid having all containers in a few stacks
        occupied_positions = 0
        total_containers = 0
        
        for row in yard.row_names:
            for bay in range(1, yard.num_bays + 1):
                position = f"{row}{bay}"
                containers = yard.get_containers_at_position(position)
                if containers:
                    occupied_positions += 1
                    total_containers += len(containers)
        
        # Higher score for better distribution
        if occupied_positions > 0:
            avg_height = total_containers / occupied_positions
            if avg_height <= 2:  # Ideal height
                score += 50
            else:
                score -= (avg_height - 2) * 10  # Penalty for tall stacks
        
        return score
    
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
        # This is a simplified version - a full implementation would
        # actually simulate the move and calculate exact scores
        
        # Get containers at source and destination
        yard = env.storage_yard
        source_containers = yard.get_containers_at_position(source_position)
        dest_containers = yard.get_containers_at_position(dest_position)
        
        if not source_containers:
            return -5.0  # No container to move
        
        # Get top container at source
        source_tier = max(source_containers.keys())
        top_container = source_containers[source_tier]
        
        # Estimate improvement based on simple heuristics
        improvement = 0
        
        # Check destination stack height
        dest_height = len(dest_containers)
        if dest_height >= 4:
            improvement -= 3.0  # Penalty for creating tall stacks
        
        # Check if creating a better priority order
        if dest_containers:
            dest_tier = max(dest_containers.keys())
            bottom_container = dest_containers[dest_tier]
            
            # Better to have lower priority containers below higher priority ones
            if hasattr(top_container, 'priority') and hasattr(bottom_container, 'priority'):
                if top_container.priority >= bottom_container.priority:
                    improvement += 2.0
                else:
                    improvement -= 2.0
        
        # Check if freeing up a problematic stack at source
        if len(source_containers) > 1:
            # Container below the top one
            source_below_tier = max([t for t in source_containers.keys() if t < source_tier])
            below_container = source_containers[source_below_tier]
            
            # If removing top improves priority order
            if hasattr(top_container, 'priority') and hasattr(below_container, 'priority'):
                if top_container.priority < below_container.priority:
                    improvement += 3.0
        
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
            estimated_bays_per_row = 20  # This should be adjusted based on actual layout
            row_idx = storage_idx // estimated_bays_per_row
            bay_idx = storage_idx % estimated_bays_per_row
            
            # Check if row_idx is in range
            if row_idx < len(env.storage_yard.row_names):
                row_letter = env.storage_yard.row_names[row_idx]
                return f"{row_letter}{bay_idx + 1}"
            else:
                # Fallback if index is out of range
                return f"S{idx}"
        else:
            # This is a rail or parking position
            rail_threshold = 100  # Estimate - adjust based on environment
            
            if idx < rail_threshold:
                # Rail position
                if hasattr(env, 'terminal') and hasattr(env.terminal, 'track_names'):
                    num_tracks = len(env.terminal.track_names)
                    slots_per_track = rail_threshold // num_tracks
                    
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
            'target_backbone': self.target_network.backbone.state_dict(),
            'target_transfer_head': self.target_network.transfer_head.state_dict(),
            'target_yard_head': self.target_network.yard_head.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'storage_position_threshold': self.storage_position_threshold
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model parameters."""
        checkpoint = torch.load(filepath)
        
        # Update state dimension if different
        if 'state_dim' in checkpoint and checkpoint['state_dim'] != self.state_dim:
            self.state_dim = checkpoint['state_dim']
            print(f"Updating state dimension to {self.state_dim}")
            # Recreate networks with new dimension
            self = OptimizedTerminalAgent(self.state_dim, self.action_dims, self.device)
        
        # Load parameters
        self.backbone.load_state_dict(checkpoint['backbone'])
        self.transfer_head.load_state_dict(checkpoint['transfer_head'])
        self.yard_head.load_state_dict(checkpoint['yard_head'])
        
        self.target_network.backbone.load_state_dict(checkpoint['target_backbone'])
        self.target_network.transfer_head.load_state_dict(checkpoint['target_transfer_head'])
        self.target_network.yard_head.load_state_dict(checkpoint['target_yard_head'])
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.0005)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        self.epsilon = checkpoint['epsilon']
        
        if 'storage_position_threshold' in checkpoint:
            self.storage_position_threshold = checkpoint['storage_position_threshold']
        
        print(f"Model loaded from {filepath}")

    def select_premarshalling_action(self, state, action_masks, env):
        """
        Select an action specifically for pre-marshalling, prioritizing storage-to-storage moves.
        
        Args:
            state: Current state
            action_masks: Action masks
            env: Environment
            
        Returns:
            Tuple of (selected_action, action_type, flat_state)
        """
        # Always calculate flat_state at the beginning to avoid UnboundLocalError
        flat_state = self._flatten_state(state)
        
        # Extract yard optimization mask (specifically storage-to-storage moves)
        yard_mask = self._extract_yard_mask(action_masks)
        
        # If no valid yard optimization moves, try regular transfer actions
        if yard_mask is None or yard_mask.sum() == 0:
            transfer_mask = self._extract_transfer_mask(action_masks)
            if transfer_mask is not None and transfer_mask.sum() > 0:
                # Create action based on transfer mask
                valid_indices = np.argwhere(transfer_mask == 1)
                if len(valid_indices) > 0:
                    # Select an action with lower exploration rate
                    if random.random() < 0.1:  # Lower epsilon for pre-marshalling
                        action_idx = random.randint(0, len(valid_indices) - 1)
                        action = tuple(valid_indices[action_idx])
                    else:
                        # Use model to select best action
                        try:
                            # Convert state to tensor for network
                            state_tensor = torch.FloatTensor(flat_state).unsqueeze(0).to(self.device)
                            
                            # Get features from backbone
                            features = self.backbone(state_tensor)
                            
                            # Find best action using transfer head
                            best_action = None
                            best_value = float('-inf')
                            
                            for idx in valid_indices:
                                # For each valid action, evaluate its value
                                with torch.no_grad():
                                    # Use transfer head
                                    q_value = self.transfer_head(features).item()
                                
                                # Update if better
                                if q_value > best_value:
                                    best_value = q_value
                                    best_action = tuple(idx)
                            
                            if best_action is not None:
                                action = best_action
                            else:
                                # Fallback to random if model fails
                                action_idx = random.randint(0, len(valid_indices) - 1)
                                action = tuple(valid_indices[action_idx])
                        except Exception:
                            # Fallback to random on error
                            action_idx = random.randint(0, len(valid_indices) - 1)
                            action = tuple(valid_indices[action_idx])
                    
                    # Map to environment action format
                    env_action = {
                        'action_type': 0,  # Crane movement
                        'crane_movement': np.array(action, dtype=np.int64),
                        'truck_parking': np.zeros(2, dtype=np.int64)  # Placeholder
                    }
                    
                    return env_action, 0, flat_state
        
        # If we have valid yard optimization moves
        elif yard_mask is not None and yard_mask.sum() > 0:
            valid_indices = np.argwhere(yard_mask == 1)
            
            # Select an action with lower exploration rate
            if random.random() < 0.1:  # Lower epsilon for pre-marshalling 
                action_idx = random.randint(0, len(valid_indices) - 1)
                action = tuple(valid_indices[action_idx])
            else:
                # Use model to select best action
                try:
                    # Convert state to tensor for network
                    state_tensor = torch.FloatTensor(flat_state).unsqueeze(0).to(self.device)
                    
                    # Get features from backbone
                    features = self.backbone(state_tensor)
                    
                    # Find best action using yard head
                    best_action = None
                    best_value = float('-inf')
                    
                    for idx in valid_indices:
                        # For each valid action, evaluate its value
                        with torch.no_grad():
                            # Use yard head
                            q_value = self.yard_head(features).item()
                            
                            # Evaluate if this specific pre-marshalling action improves yard
                            source_pos = self._idx_to_position(idx[1], env)
                            dest_pos = self._idx_to_position(idx[2], env)
                            
                            # Get yard improvement estimate
                            improvement = self._estimate_move_improvement(env, source_pos, dest_pos)
                            
                            # Boost or penalize based on estimated improvement
                            q_value = q_value * (1.0 + improvement * 0.5)
                        
                        # Update if better
                        if q_value > best_value:
                            best_value = q_value
                            best_action = tuple(idx)
                    
                    if best_action is not None:
                        action = best_action
                    else:
                        # Fallback to random if model fails
                        action_idx = random.randint(0, len(valid_indices) - 1)
                        action = tuple(valid_indices[action_idx])
                except Exception:
                    # Fallback to random on error
                    action_idx = random.randint(0, len(valid_indices) - 1)
                    action = tuple(valid_indices[action_idx])
            
            # Map to environment action format
            env_action = {
                'action_type': 0,  # Crane movement
                'crane_movement': np.array(action, dtype=np.int64),
                'truck_parking': np.zeros(2, dtype=np.int64)  # Placeholder
            }
            
            return env_action, 1, flat_state
        
        # No valid actions
        return None, None, flat_state

def handle_truck_parking(env):
    """
    Rule-based truck parking assignment function - can be used separately from the agent.
    
    Args:
        env: Terminal environment
    
    Returns:
        Boolean indicating if a truck was parked
    """
    # Check if there are trucks waiting and available parking spots
    if env.truck_queue.size() == 0:
        return False
        
    available_spots = [spot for spot in env.parking_spots if spot not in env.trucks_in_terminal]
    if not available_spots:
        return False
    
    # Get the next truck from the queue
    truck = env.truck_queue.get_next_vehicle()
    
    # Choose the best spot based on truck purpose
    if truck.is_pickup_truck and hasattr(truck, 'pickup_container_ids') and truck.pickup_container_ids:
        # This is a pickup truck - place it close to the containers it needs
        container_positions = []
        
        # Find container locations
        for container_id in truck.pickup_container_ids:
            position = env.storage_yard.find_container(container_id)
            if position:
                container_positions.append(position[0])
        
        if container_positions:
            # Find the parking spot closest to the containers
            best_spot = None
            best_distance = float('inf')
            
            for spot in available_spots:
                avg_distance = 0
                for pos in container_positions:
                    avg_distance += env.terminal.get_distance(spot, pos)
                avg_distance /= len(container_positions)
                
                if avg_distance < best_distance:
                    best_distance = avg_distance
                    best_spot = spot
        else:
            # No container positions found, use first available spot
            best_spot = available_spots[0]
    else:
        # This is a delivery truck - place it near rail tracks
        rail_positions = []
        for track in env.terminal.track_names:
            if track in env.trains_in_terminal:
                # Find a rail slot for this track
                slot = f"{track.lower()}_1"  # First slot
                rail_positions.append(slot)
        
        if rail_positions:
            # Find the parking spot closest to the rail positions
            best_spot = None
            best_distance = float('inf')
            
            for spot in available_spots:
                avg_distance = 0
                for pos in rail_positions:
                    avg_distance += env.terminal.get_distance(spot, pos)
                avg_distance /= len(rail_positions)
                
                if avg_distance < best_distance:
                    best_distance = avg_distance
                    best_spot = spot
        else:
            # No rail positions with trains, use first available spot
            best_spot = available_spots[0]
    
    # Assign truck to the best spot
    truck.parking_spot = best_spot
    env.trucks_in_terminal[best_spot] = truck
    print(f"Truck {truck.truck_id} parked at {best_spot}")
    
    return True