import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional, Any
import os
import time
import re
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import our custom components
from terminal_layout.CTSimulator import ContainerTerminal
from terminal_components.Container import Container, ContainerFactory
from terminal_components.Train import Train
from terminal_components.Truck import Truck
from terminal_components.Storage_Yard import StorageYard
from terminal_components.RMGCrane import RMGCrane
from terminal_components.Vehicle_Queue import VehicleQueue


class TerminalEnvironment(gym.Env):
    """
    Container Terminal Environment for RL agents.
    
    This environment simulates a container terminal with:
    - Rail tracks for trains
    - Parking spots for trucks
    - Storage yard for containers
    - RMG cranes for moving containers
    
    The agent's task is to efficiently move containers between trains, trucks, 
    and storage areas to maximize throughput while minimizing waiting times.
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, 
                 terminal_config_path: str = None,
                 distance_matrix_path: str = None,
                 truck_arrival_model_path: str = None,
                 train_arrival_model_path: str = None,
                 max_simulation_time: float = 86400,  # 24 hours in seconds
                 num_cranes: int = 2):
        """
        Initialize the terminal environment.
        
        Args:
            terminal_config_path: Path to terminal configuration file
            distance_matrix_path: Path to pre-calculated distance matrix
            truck_arrival_model_path: Path to KDE model for truck arrivals
            train_arrival_model_path: Path to KDE model for train arrivals
            max_simulation_time: Maximum simulation time in seconds
            num_cranes: Number of RMG cranes in the terminal
        """
        super(TerminalEnvironment, self).__init__()
        
        # Load terminal configuration
        if terminal_config_path and os.path.exists(terminal_config_path):
            # TODO: Load terminal configuration from file
            pass
        else:
            # Create default terminal configuration
            self.terminal = ContainerTerminal(
                layout_order=['rails', 'parking', 'driving_lane', 'yard_storage'],
                num_railtracks=6,      
                num_railslots_per_track=29,
                num_storage_rows=5,   
                # Ratio parameters
                parking_to_railslot_ratio=1.0,
                storage_to_railslot_ratio=2.0,
                # Dimension parameters
                rail_slot_length=24.384,
                track_width=2.44,
                space_between_tracks=2.05,
                space_rails_to_parking=1.05,
                space_driving_to_storage=0.26,
                parking_width=4.0,
                driving_lane_width=4.0,
                storage_slot_width=2.5
            )
        
        # Load distance matrix if available
        if distance_matrix_path and os.path.exists(distance_matrix_path):
            self.terminal.load_distance_matrix(distance_matrix_path)
        
        # Load arrival models if available
        self.truck_arrival_model = None
        self.train_arrival_model = None
        if truck_arrival_model_path and os.path.exists(truck_arrival_model_path):
            with open(truck_arrival_model_path, 'rb') as f:
                self.truck_arrival_model = pickle.load(f)
                
        if train_arrival_model_path and os.path.exists(train_arrival_model_path):
            with open(train_arrival_model_path, 'rb') as f:
                self.train_arrival_model = pickle.load(f)
        
        # Initialize environment components
        self.storage_yard = StorageYard(
            num_rows=self.terminal.num_storage_rows,
            num_bays=self.terminal.num_storage_slots_per_row,
            max_tier_height=5,  # Maximum stacking height
            row_names=self.terminal.storage_row_names,
            special_areas={
                'reefer': [('A', 1, 1)],  # Example: Row A is for reefer containers
                'reefer': [('B', 1, 1)],
                'reefer': [('C', 1, 1)],
                'reefer': [('D', 1, 1)],
                'reefer': [('E', 1, 1)],
                'reefer': [('A', 58, 58)],  # Example: Row A is for reefer containers
                'reefer': [('B', 58, 58)],
                'reefer': [('C', 58, 58)],
                'reefer': [('D', 58, 58)],
                'reefer': [('E', 58, 58)],
                'dangerous': [('A', 33, 35)],  # Example: Row F is for dangerous goods
                'dangerous': [('B', 33, 35)],
                'dangerous': [('C', 33, 35)],
                'dangerous': [('D', 33, 35)],
                'dangerous': [('E', 33, 35)]
            }
        )
        
        # Create RMG cranes
        self.cranes = []
        for i in range(num_cranes):
            # Divide the terminal space among the cranes
            start_bay = i * (self.terminal.num_storage_slots_per_row // num_cranes)
            end_bay = (i + 1) * (self.terminal.num_storage_slots_per_row // num_cranes) - 1
            
            crane = RMGCrane(
                crane_id=f"RMG{i+1}",
                terminal=self.terminal,
                start_bay=start_bay,
                end_bay=end_bay,
                current_position=(start_bay, 0)  # Start at first rail track
            )
            self.cranes.append(crane)
        
        # Create vehicle queues
        self.truck_queue = VehicleQueue(vehicle_type="Truck")
        self.train_queue = VehicleQueue(vehicle_type="Train")
        
        # Track current state
        self.current_simulation_time = 0.0  # Starting time in seconds from beginning of simulation
        self.max_simulation_time = max_simulation_time
        self.crane_available_times = [0.0] * num_cranes  # When each crane becomes available
        
        self.trucks_in_terminal = {}  # Map from parking_spot to Truck
        self.trains_in_terminal = {}  # Map from rail_track to Train
        
        # Create parking spot mapping
        self.parking_spots = [f"p_{i+1}" for i in range(self.terminal.num_parking_spots)]
        self.rail_slots = {}
        for track in self.terminal.track_names:
            self.rail_slots[track] = [f"{track.lower()}_{i+1}" for i in range(self.terminal.num_railslots_per_track)]
        
        # Define action and observation spaces
        # Action space: (crane_id, source, destination)
        # Where source and destination are indices into the flattened list of all positions
        all_positions = []
        
        # Flatten the rail slots lists into a single list
        for rail_list in self.rail_slots.values():
            all_positions.extend(rail_list)
            
        # Add parking spots and storage positions
        all_positions.extend(self.parking_spots)
        all_positions.extend([f"{row}{i+1}" for row in self.terminal.storage_row_names 
                            for i in range(self.terminal.num_storage_slots_per_row)])
        
        self.position_to_idx = {pos: i for i, pos in enumerate(all_positions)}
        self.idx_to_position = {i: pos for i, pos in enumerate(all_positions)}
        
        self.action_space = spaces.MultiDiscrete([
            num_cranes,  # Crane index
            len(all_positions),  # Source position index
            len(all_positions)   # Destination position index
        ])
        
        # Create an action mask space
        self.action_mask_space = spaces.Box(
            low=0,
            high=1,
            shape=(
                num_cranes,  # Crane index
                len(all_positions),  # Source position index
                len(all_positions)   # Destination position index
            ),
            dtype=np.int8
        )
        
        # Observation space: complex representation of terminal state
        # This will need to be refined based on what information is relevant to the agent
        # For now, we'll use a simplified representation
        self.observation_space = spaces.Dict({
            # Crane positions and availability
            'crane_positions': spaces.Box(
                low=0, 
                high=max(self.terminal.num_storage_slots_per_row, self.terminal.num_railslots_per_track), 
                shape=(num_cranes, 2), 
                dtype=np.int32
            ),
            'crane_available_times': spaces.Box(
                low=0,
                high=np.inf,
                shape=(num_cranes,),
                dtype=np.float32
            ),
            # Current simulation time
            'current_time': spaces.Box(
                low=0,
                high=np.inf,
                shape=(1,),
                dtype=np.float32
            ),
            # Storage yard state (simplified)
            'yard_state': spaces.Box(
                low=0,
                high=1,  # Binary representation for now (occupied/free)
                shape=(self.terminal.num_storage_rows, self.terminal.num_storage_slots_per_row, 5),  # Rows, bays, tiers
                dtype=np.int32
            ),
            # Truck parking status
            'parking_status': spaces.Box(
                low=0,
                high=1,  # Binary representation (occupied/free)
                shape=(len(self.parking_spots),),
                dtype=np.int32
            ),
            # Rail track status
            'rail_status': spaces.Box(
                low=0,
                high=1,  # Binary representation (occupied/free)
                shape=(len(self.terminal.track_names), self.terminal.num_railslots_per_track),
                dtype=np.int32
            ),
            # Queue sizes
            'queue_sizes': spaces.Box(
                low=0,
                high=np.inf,
                shape=(2,),  # [truck_queue_size, train_queue_size]
                dtype=np.int32
            ),
            # Action mask (which actions are valid)
            'action_mask': self.action_mask_space
        })
        
        # Initialize the environment state
        self.reset()
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Random seed
            options: Additional options for resetting
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset time tracking
        self.current_simulation_time = 0.0
        self.crane_available_times = [0.0] * len(self.cranes)
        
        # Clear terminal state
        self.trucks_in_terminal = {}
        self.trains_in_terminal = {}
        self.truck_queue.clear()
        self.train_queue.clear()
        self.storage_yard.clear()
        
        # Reset cranes to initial positions
        for i, crane in enumerate(self.cranes):
            start_bay = i * (self.terminal.num_storage_slots_per_row // len(self.cranes))
            crane.reset(position=(start_bay, 0))
        
        # Initialize with some random containers in the storage yard
        self._initialize_storage_yard()
        
        # Add some trucks and trains to the queues
        self._generate_initial_vehicles()
        
        # Get initial observation
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """
        Take a step in the environment by executing the specified action.
        
        Args:
            action: Action to take, in the form [crane_idx, source_idx, destination_idx]
            
        Returns:
            observation: Next observation
            reward: Reward for the action
            terminated: Whether the episode is over
            truncated: Whether the episode was truncated
            info: Additional information
        """
        # Decode the action
        crane_idx, source_idx, destination_idx = action
        
        # Get current observation with action mask
        current_obs = self._get_observation()
        
        # Check if the action is valid according to the mask
        if current_obs['action_mask'][crane_idx, source_idx, destination_idx] == 0:
            # Invalid action - choose a random valid action instead
            valid_actions = np.argwhere(current_obs['action_mask'] == 1)
            if len(valid_actions) > 0:
                # Choose a random valid action
                valid_idx = np.random.randint(0, len(valid_actions))
                crane_idx, source_idx, destination_idx = valid_actions[valid_idx]
                # Inform about the replacement
                print(f"Warning: Invalid action replaced with valid action: "
                     f"Crane {crane_idx+1}: {self.idx_to_position[source_idx]} → {self.idx_to_position[destination_idx]}")
            else:
                # No valid actions - wait until next time
                observation = current_obs
                reward = 0  # Neutral reward for waiting
                terminated = self.current_simulation_time >= self.max_simulation_time
                truncated = False
                info = {"action": "wait", "reason": "No valid actions available"}
                return observation, reward, terminated, truncated, info
        
        # Now proceed with the (valid) action
        source_position = self.idx_to_position[source_idx]
        destination_position = self.idx_to_position[destination_idx]
        
        # Check if the selected crane is available
        if self.current_simulation_time < self.crane_available_times[crane_idx]:
            # Crane is not available yet - skip to when it becomes available
            time_advanced = self.crane_available_times[crane_idx] - self.current_simulation_time
            self.current_simulation_time = self.crane_available_times[crane_idx]
            
            # Process any events that happen during this time
            self._process_time_advancement(time_advanced)
            
            # Return observation with no reward (waiting is neutral)
            observation = self._get_observation()
            reward = 0
            terminated = self.current_simulation_time >= self.max_simulation_time
            truncated = False
            info = {"action": "wait", "time_advanced": time_advanced}
            return observation, reward, terminated, truncated, info
        
        # Get the crane and execute the move
        crane = self.cranes[crane_idx]
        container, time_taken = crane.move_container(
            source_position, 
            destination_position, 
            self.storage_yard, 
            self.trucks_in_terminal, 
            self.trains_in_terminal
        )
        
        # Calculate the reward based on the move
        reward = self._calculate_reward(container, source_position, destination_position, time_taken)
        
        # Update crane availability time
        self.crane_available_times[crane_idx] = self.current_simulation_time + time_taken
        
        # Advance simulation time to the earliest available crane
        next_available_time = min(self.crane_available_times)
        time_advanced = next_available_time - self.current_simulation_time
        self.current_simulation_time = next_available_time
        
        # Process any events that happen during this time advancement
        self._process_time_advancement(time_advanced)
        
        # Get the next observation
        observation = self._get_observation()
        
        # Check if the episode is over
        terminated = self.current_simulation_time >= self.max_simulation_time
        truncated = False
        
        # Additional info
        info = {
            "time_taken": time_taken,
            "container_moved": container.container_id if container else None,
            "crane_position": crane.current_position,
            "trucks_waiting": self.truck_queue.size(),
            "trains_waiting": self.train_queue.size(),
            "current_time": self.current_simulation_time,
            "valid_action": True
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode ('human' or 'rgb_array')
            
        Returns:
            Rendered representation of the environment
        """
        if mode == 'human':
            # Render the terminal for human viewing
            fig, ax = self.terminal.visualize(figsize=(15, 10), show_labels=True)
            
            # Add containers, cranes, trucks, and trains to visualization
            
            # Draw cranes
            for i, crane in enumerate(self.cranes):
                pos = crane.current_position
                crane_color = f'C{i}'  # Different color for each crane
                
                # Convert bay, row to x, y
                # This is approximate - would need better coordinates in a real system
                x_pos = pos[0] * self.terminal.rail_slot_length + self.terminal.rail_slot_length / 2
                y_pos = 0  # Placeholder - would need conversion based on terminal layout
                
                # Draw crane as a rectangle
                crane_width = self.terminal.rail_slot_length
                crane_height = 3.0
                crane_rect = plt.Rectangle((x_pos - crane_width/2, y_pos - crane_height/2), 
                                        crane_width, crane_height, 
                                        color=crane_color, alpha=0.7, label=f"Crane {i+1}")
                ax.add_patch(crane_rect)
                
                # Indicate if crane is busy
                if self.current_simulation_time < self.crane_available_times[i]:
                    busy_until = self.crane_available_times[i] - self.current_simulation_time
                    ax.text(x_pos, y_pos + crane_height, 
                           f"Busy for {busy_until:.1f}s", 
                           ha='center', va='bottom', color=crane_color)
            
            # Add key simulation information
            title = f"Container Terminal Simulation - Time: {self.current_simulation_time:.1f}s"
            if hasattr(self, 'last_action'):
                title += f" - Last Action: {self.last_action}"
            ax.set_title(title, fontsize=16)
            
            # Add simulation stats as text
            stats_text = [
                f"Trucks in terminal: {len(self.trucks_in_terminal)}",
                f"Trucks waiting: {self.truck_queue.size()}",
                f"Trains in terminal: {len(self.trains_in_terminal)}",
                f"Trains waiting: {self.train_queue.size()}",
                f"Containers in yard: {self.storage_yard.get_container_count()}"
            ]
            
            for i, text in enumerate(stats_text):
                fig.text(0.02, 0.95 - i*0.05, text, fontsize=10, 
                       ha='left', va='top', transform=fig.transFigure)
            
            return fig
        
        elif mode == 'rgb_array':
            # Return a numpy array representation of the rendering
            fig = self.render(mode='human')
            
            # Convert figure to RGB array
            fig.canvas.draw()
            img = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close(fig)
            
            return img
    
    def close(self):
        """Clean up environment resources."""
        pass
    
    def _get_observation(self):
        """
        Get the current observation of the environment.
        
        Returns:
            Dictionary containing observation data
        """
        # Create observation dictionary with existing components
        observation = {
            # Crane positions and availability
            'crane_positions': np.array([crane.current_position for crane in self.cranes], dtype=np.int32),
            'crane_available_times': np.array(self.crane_available_times, dtype=np.float32),
            
            # Current simulation time
            'current_time': np.array([self.current_simulation_time], dtype=np.float32),
            
            # Storage yard state
            'yard_state': self.storage_yard.get_state_representation(),
            
            # Truck parking status
            'parking_status': np.zeros(len(self.parking_spots), dtype=np.int32),
            
            # Rail track status
            'rail_status': np.zeros((len(self.terminal.track_names), self.terminal.num_railslots_per_track), dtype=np.int32),
            
            # Queue sizes
            'queue_sizes': np.array([self.truck_queue.size(), self.train_queue.size()], dtype=np.int32)
        }
        
        # Update existing components as before
        for i, spot in enumerate(self.parking_spots):
            if spot in self.trucks_in_terminal:
                observation['parking_status'][i] = 1
        
        for i, track in enumerate(self.terminal.track_names):
            if track in self.trains_in_terminal:
                train = self.trains_in_terminal[track]
                for j in range(len(train.wagons)):
                    if j < self.terminal.num_railslots_per_track:
                        observation['rail_status'][i, j] = 1
        
        # Generate action mask
        action_mask = np.zeros((
            len(self.cranes), 
            len(self.position_to_idx),
            len(self.position_to_idx)
        ), dtype=np.int8)
        
        # For each crane, calculate valid moves
        for i, crane in enumerate(self.cranes):
            # Skip if crane is not available yet
            if self.current_simulation_time < self.crane_available_times[i]:
                continue
                
            valid_moves = crane.get_valid_moves(self.storage_yard, self.trucks_in_terminal, self.trains_in_terminal)
            
            # Apply additional restrictions:
            # 1. No rail slot to rail slot movements
            # 2. No pre-marshalling moves exceeding 5 slots
            # 3. Special container (Trailer/Swap Body) placement restrictions
            
            filtered_moves = {}
            for (source, dest), time in valid_moves.items():
                source_idx = self.position_to_idx[source]
                dest_idx = self.position_to_idx[dest]
                
                # Check rail slot to rail slot restriction
                if self._is_rail_position(source) and self._is_rail_position(dest):
                    continue
                    
                # Check pre-marshalling distance constraint
                if self._is_storage_position(source) and self._is_storage_position(dest):
                    # Extract bay numbers
                    source_bay = int(re.findall(r'\d+', source)[0]) - 1
                    dest_bay = int(re.findall(r'\d+', dest)[0]) - 1
                    if abs(source_bay - dest_bay) > 5:
                        continue
                
                # Check special container placement
                container = self._get_container_at_position(source, self.storage_yard, 
                                                         self.trucks_in_terminal, self.trains_in_terminal)
                if (container and container.container_type in ["Trailer", "Swap Body"] 
                    and self._is_storage_position(dest)):
                    # Get row of destination
                    dest_row = dest[0]
                    # Get row nearest to driving lane
                    nearest_row = self.terminal.storage_row_names[0]
                    if dest_row != nearest_row:
                        continue
                
                # If passed all checks, mark as valid in the action mask
                action_mask[i, source_idx, dest_idx] = 1
                filtered_moves[(source, dest)] = time
            
            # Update crane's valid moves with the filtered set
            if hasattr(crane, 'valid_moves'):
                crane.valid_moves = filtered_moves
        
        # Add action mask to observation
        observation['action_mask'] = action_mask
        
        return observation
    
    def _calculate_reward(self, container, source, destination, time_taken):
        """
        Calculate the reward for moving a container.
        
        Args:
            container: The container that was moved
            source: Source position
            destination: Destination position
            time_taken: Time taken for the move in seconds
            
        Returns:
            Calculated reward value
        """
        if container is None:
            # Penalty for moves that don't involve containers (crane moving empty)
            # Apply punishment per distance unit the RMG travels without carrying containers
            return -5 - (time_taken / 10)  # Increase penalty based on travel time
        
        # Identify the type of move
        source_type = self._get_position_type(source)
        dest_type = self._get_position_type(destination)
        
        # Base reward initialization
        reward = 0
        
        # DIRECT MOVE FROM TRAIN TO TRUCK - BIG REWARD
        if source_type == 'train' and dest_type == 'truck':
            # Direct transfer train to truck (very good)
            reward += 10  # Big reward
        
        # SPLIT MOVE FROM YARD TO TRUCK/TRAIN - SMALL REWARD
        elif source_type == 'storage' and (dest_type == 'truck' or dest_type == 'train'):
            # Loading truck/train from storage (good)
            reward += 3  # Small reward
            
            # SCALAR REWARD - Container moved before deadline
            if hasattr(container, 'departure_date') and container.departure_date:
                time_until_deadline = (container.departure_date - datetime.now()).total_seconds()
                if time_until_deadline > 0:
                    # Scale reward based on time till deadline (higher for tighter deadlines)
                    time_factor = min(1.0, 24*3600 / max(3600, time_until_deadline))  # Cap at 24 hours
                    reward += 5 * time_factor  # Maximum +5 bonus for very tight deadlines
        
        # ASSIGNING TRUCK TO PARKING SLOT - SMALL REWARD
        elif source_type == 'truck' and dest_type == 'parking':
            reward += 2  # Small reward for proper truck assignment
            
        # STANDARD MOVES
        elif source_type == 'train' and dest_type == 'storage':
            # Unloading train to storage (standard operation)
            reward += 2
        elif source_type == 'truck' and dest_type == 'storage':
            # Unloading truck to storage (standard operation)
            reward += 2
            
        # RESHUFFLING/PRE-MARSHALLING - PUNISHMENT
        elif source_type == 'storage' and dest_type == 'storage':
            # Reshuffling (necessary but not ideal) - PUNISHMENT
            reward -= 4
        
        # SCALAR PUNISHMENT - Shipping after deadline
        if hasattr(container, 'departure_date') and container.departure_date:
            time_past_deadline = (datetime.now() - container.departure_date).total_seconds()
            if time_past_deadline > 0:
                # More punishment the longer past deadline
                past_deadline_hours = time_past_deadline / 3600
                reward -= min(10, past_deadline_hours * 0.5)  # Cap at -10
        
        # Adjust reward based on container priority
        if hasattr(container, 'priority'):
            # Higher priority (lower number) gives higher reward
            priority_factor = max(0, (100 - container.priority) / 100)
            reward += priority_factor * 2
        
        # Penalize for time taken, but not too harshly
        time_penalty = min(time_taken / 120, 1)  # Cap at 1 for moves taking over 2 minutes
        reward -= time_penalty
        
        return reward
    
    def _get_position_type(self, position):
        """
        Determine the type of a position (train, truck, storage).
        
        Args:
            position: Position string
            
        Returns:
            Type of the position ('train', 'truck', or 'storage')
        """
        if position.startswith('t') and '_' in position:
            return 'train'
        elif position.startswith('p_'):
            return 'truck'
        else:
            return 'storage'
    
    def _is_storage_position(self, position: str) -> bool:
        """Check if a position is in the storage yard."""
        # Storage positions typically start with a letter and are followed by a number
        return position[0].isalpha() and position[1:].isdigit()
    
    def _is_rail_position(self, position: str) -> bool:
        """Check if a position is a rail slot."""
        # Rail positions typically start with t and have an underscore
        return position.startswith('t') and '_' in position
    
    def _get_container_at_position(self, position: str, storage_yard, trucks_in_terminal, trains_in_terminal):
        """Helper to get container at a position."""
        if self._is_storage_position(position):
            # Get top container from storage
            container, _ = storage_yard.get_top_container(position)
            return container
            
        elif position.startswith('p_'):
            # Get container from truck
            truck = trucks_in_terminal.get(position)
            if truck and hasattr(truck, 'containers') and truck.containers:
                return truck.containers[0]  # Return the first container
            return None
            
        elif position.startswith('t') and '_' in position:
            # Parse train position
            parts = position.split('_')
            if len(parts) != 2:
                return None
                
            track_num = parts[0][1:]
            slot_num = int(parts[1])
            
            # Find the train
            track_id = f"T{track_num}"
            train = trains_in_terminal.get(track_id)
            
            if train and 0 <= slot_num - 1 < len(train.wagons):
                wagon = train.wagons[slot_num - 1]
                if wagon.containers:
                    return wagon.containers[0]
                    
        return None
    
    def _initialize_storage_yard(self):
        """Initialize the storage yard with random containers."""
        # Fill about 30% of the yard with random containers
        num_rows = self.terminal.num_storage_rows
        num_bays = self.terminal.num_storage_slots_per_row
        num_positions = num_rows * num_bays
        
        # Number of positions to fill
        num_to_fill = int(num_positions * 0.3)
        
        # Randomly select positions to fill
        positions_to_fill = np.random.choice(num_positions, num_to_fill, replace=False)
        
        for pos_idx in positions_to_fill:
            # Convert flat index to row, bay
            row_idx = pos_idx // num_bays
            bay_idx = pos_idx % num_bays
            
            # Convert to position string
            row = self.terminal.storage_row_names[row_idx]
            position = f"{row}{bay_idx+1}"
            
            # Create a random container
            container = ContainerFactory.create_random()
            
            # Respect trailer/swap body placement constraint
            if container.container_type in ["Trailer", "Swap Body"]:
                # Only place in the row nearest to driving lane
                if row != self.terminal.storage_row_names[0]:
                    # Create a different type of container
                    container = ContainerFactory.create_random()
                    # If still a special type, skip this position
                    if container.container_type in ["Trailer", "Swap Body"]:
                        continue
            
            # Add to storage yard
            self.storage_yard.add_container(position, container)
            
            # Randomly add a second container (20% chance)
            if np.random.random() < 0.2:
                # Create another container
                container2 = ContainerFactory.create_random()
                # Check stacking compatibility
                if container2.can_stack_with(container):
                    self.storage_yard.add_container(position, container2, tier=2)
    
    def _generate_initial_vehicles(self):
        """Generate initial trucks and trains to populate the environment."""
        # Add some trucks to the queue
        for i in range(5):
            # Create a random truck
            truck = Truck(truck_id=f"TRK{i+1}")
            
            # Randomly decide if it's bringing a container or picking one up
            if np.random.random() < 0.5:
                # Truck bringing a container
                container = ContainerFactory.create_random()
                truck.add_container(container)
            else:
                # Truck coming to pick up a container
                truck.add_pickup_container_id(f"CONT{np.random.randint(1000, 9999)}")
            
            self.truck_queue.add_vehicle(truck)
        
        # Add a train to the queue
        train = Train(train_id="TRN1", num_wagons=5)
        
        # Add some containers to the train
        for i in range(3):
            container = ContainerFactory.create_random()
            train.add_container(container)
        
        # Add some pickup requests to the train
        for i in range(2):
            train.add_pickup_container(f"CONT{np.random.randint(1000, 9999)}")
        
        self.train_queue.add_vehicle(train)
    
    def _process_time_advancement(self, time_advanced):
        """
        Process events that occur during time advancement.
        
        Args:
            time_advanced: Amount of time advanced in seconds
        """
        # Process vehicle arrivals based on time advancement
        self._process_vehicle_arrivals(time_advanced)
        
        # Process vehicle departures
        self._process_vehicle_departures()
    
    def _process_vehicle_arrivals(self, time_advanced):
        """
        Process vehicle arrivals based on elapsed time.
        
        Args:
            time_advanced: Amount of time advanced in seconds
        """
        # Use KDE models or simple random generation
        # For now, use simple probability-based arrivals
        
        # Chance of truck arrival increases with time
        truck_arrival_probability = min(0.8, time_advanced / 3600)  # Cap at 80% per hour
        if np.random.random() < truck_arrival_probability:
            # Create a random truck and add to queue
            truck = Truck()
            # Randomly decide if bringing container or picking up
            if np.random.random() < 0.5:
                # Truck bringing a container
                container = ContainerFactory.create_random()
                truck.add_container(container)
            else:
                # Truck coming to pick up
                # Use a random container ID for now - in a real system, this would be specific
                truck.add_pickup_container_id(f"CONT{np.random.randint(1000, 9999)}")
            
            self.truck_queue.add_vehicle(truck)
        
        # Chance of train arrival (lower than trucks)
        train_arrival_probability = min(0.2, time_advanced / 7200)  # Cap at 20% per two hours
        if np.random.random() < train_arrival_probability:
            # Create a random train and add to queue
            train = Train(num_wagons=np.random.randint(3, 8))
            
            # Add some containers (30-70% of wagons have containers)
            container_count = int(len(train.wagons) * np.random.uniform(0.3, 0.7))
            for _ in range(container_count):
                container = ContainerFactory.create_random()
                train.add_container(container)
                
            # Add some pickup requests (10-30% of wagons)
            pickup_count = int(len(train.wagons) * np.random.uniform(0.1, 0.3))
            for _ in range(pickup_count):
                train.add_pickup_container(f"CONT{np.random.randint(1000, 9999)}")
                
            self.train_queue.add_vehicle(train)
        
        # Process arrivals from queues
        self._process_truck_arrivals()
        self._process_train_arrivals()
    
    def _process_vehicle_departures(self):
        """Process vehicle departures."""
        self._process_truck_departures()
        self._process_train_departures()
    
    def _process_truck_arrivals(self):
        """Process trucks from the queue into available parking spots."""
        # Check for empty parking spots
        empty_spots = [spot for spot in self.parking_spots if spot not in self.trucks_in_terminal]
        
        # Move trucks from queue to empty spots
        while empty_spots and not self.truck_queue.is_empty():
            spot = empty_spots.pop(0)
            truck = self.truck_queue.get_next_vehicle()
            truck.parking_spot = spot
            truck.status = "waiting"
            self.trucks_in_terminal[spot] = truck
    
    def _process_train_arrivals(self):
        """Process trains from the queue into available rail tracks."""
        # Check for empty rail tracks
        empty_tracks = [track for track in self.terminal.track_names if track not in self.trains_in_terminal]
        
        # Move trains from queue to empty tracks
        while empty_tracks and not self.train_queue.is_empty():
            track = empty_tracks.pop(0)
            train = self.train_queue.get_next_vehicle()
            train.rail_track = track
            train.status = "waiting"
            self.trains_in_terminal[track] = train
    
    def _process_truck_departures(self):
        """Process trucks that are ready to depart."""
        spots_to_remove = []
        
        for spot, truck in self.trucks_in_terminal.items():
            if truck.is_ready_to_depart():
                truck.depart()
                spots_to_remove.append(spot)
        
        for spot in spots_to_remove:
            del self.trucks_in_terminal[spot]
    
    def _process_train_departures(self):
        """Process trains that are ready to depart."""
        tracks_to_remove = []
        
        for track, train in self.trains_in_terminal.items():
            if train.is_fully_loaded():
                train.depart()
                tracks_to_remove.append(track)
        
        for track in tracks_to_remove:
            del self.trains_in_terminal[track]


def test_environment(num_steps=5, render=True):
    """Test the terminal environment with random actions."""
    env = TerminalEnvironment()
    obs, info = env.reset()
    
    print(f"Environment initialized with {len(env.cranes)} cranes")
    
    # Print all valid moves
    print("\n=== VALID MOVES BEFORE FIRST ACTION ===")
    for i, crane in enumerate(env.cranes):
        valid_moves = crane.get_valid_moves(env.storage_yard, env.trucks_in_terminal, env.trains_in_terminal)
        print(f"\nCrane {i+1} can make {len(valid_moves)} valid moves:")
        
        # Sort moves by source for better readability
        sorted_moves = sorted(valid_moves.items(), key=lambda x: x[0][0])
        
        # Print first 10 moves as sample (there could be hundreds)
        for j, ((source, dest), time) in enumerate(sorted_moves[:10]):
            print(f"  {source} → {dest} (Est. time: {time:.1f}s)")
        
        # If there are more moves, show a count
        if len(valid_moves) > 10:
            print(f"  ...and {len(valid_moves) - 10} more moves")
    
    total_reward = 0
    for i in range(num_steps):
        # Get a random action
        action = env.action_space.sample()  # Random action
        
        # Take a step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Print step information
        print(f"\nStep {i+1}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")
        print(f"Simulation Time: {obs['current_time'][0]:.2f}s")
        print(f"Crane Positions: {obs['crane_positions']}")
        print(f"Crane Available Times: {obs['crane_available_times']}")
        
        # Track last action for rendering
        if "container_moved" in info and info["container_moved"]:
            action_str = f"Crane {action[0]+1}: {env.idx_to_position[action[1]]} → {env.idx_to_position[action[2]]}"
            action_str += f" - Container: {info['container_moved']}"
            env.last_action = action_str
        else:
            env.last_action = f"Crane {action[0]+1}: {env.idx_to_position[action[1]]} → {env.idx_to_position[action[2]]}"
        
        # Print valid moves after action
        if i < num_steps - 1:  # Skip on last step
            print("\n=== VALID MOVES AFTER ACTION ===")
            for j, crane in enumerate(env.cranes):
                valid_moves = crane.get_valid_moves(env.storage_yard, env.trucks_in_terminal, env.trains_in_terminal)
                print(f"\nCrane {j+1} can make {len(valid_moves)} valid moves:")
                
                # Sort moves by source for better readability
                sorted_moves = sorted(valid_moves.items(), key=lambda x: x[0][0])
                
                # Print first 5 moves as sample (there could be hundreds)
                for k, ((source, dest), time) in enumerate(sorted_moves[:5]):
                    print(f"  {source} → {dest} (Est. time: {time:.1f}s)")
                
                # If there are more moves, show a count
                if len(valid_moves) > 5:
                    print(f"  ...and {len(valid_moves) - 5} more moves")
        
        # Render the environment
        if render:
            fig = env.render()
            plt.savefig(f"terminal_step_{i+1}.png")
            plt.show()
            plt.close()
        
        if terminated or truncated:
            print("Episode finished!")
            break
    
    # Final render
    if render:
        fig = env.render()
        plt.savefig("terminal_final.png")
        plt.show()
        plt.close()
    
    env.close()
    
    return total_reward


if __name__ == "__main__":
    # Test the environment with rendering
    print("Testing environment with 5 steps:")
    total_reward = test_environment(num_steps=5, render=True)
    print(f"\nTotal reward: {total_reward:.2f}")