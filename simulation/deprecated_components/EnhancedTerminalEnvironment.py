import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional, Any, Set
import os
import time
import re
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import torch

# Import our custom tensor-based components
from simulation.terminal_layout.CTSimulator import ContainerTerminal
from simulation.terminal_components.Container import Container, ContainerFactory
from simulation.terminal_components.Train import Train
from simulation.terminal_components.Truck import Truck
from simulation.terminal_components.TerminalTruck import TerminalTruck
from simulation.deprecated_components.TensorStorageYard import TensorStorageYard
from simulation.deprecated_components.TensorRMGCrane import TensorRMGCrane
from simulation.deprecated_components.TensorActionMaskGenerator import TensorActionMaskGenerator
from simulation.terminal_components.Vehicle_Queue import VehicleQueue


class EnhancedTerminalEnvironment(gym.Env):
    """
    Enhanced Container Terminal Environment for RL agents.
    Uses tensor operations for faster computation of action masks and valid moves.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, 
                terminal_config_path: str = None,
                terminal_config=None,
                distance_matrix_path: str = None,
                max_simulation_time: float = 86400,  # 24 hours in seconds
                num_cranes: int = 2,
                num_terminal_trucks: int = 3,
                device: str = None):
        """Initialize the enhanced terminal environment with tensor support."""
        super(EnhancedTerminalEnvironment, self).__init__()

        # Set device for tensor operations
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing enhanced terminal environment with device: {self.device}")
        
        # Performance logging flag
        self.log_performance = False
        
        # Initialize container ID storage
        self.stored_container_ids = []

        # Load or create terminal configuration
        self.config = terminal_config or self._load_config(terminal_config_path)

        # Create terminal
        self.terminal = self._create_terminal()
        
        # Load distance matrix if available
        if distance_matrix_path and os.path.exists(distance_matrix_path):
            self.terminal.load_distance_matrix(distance_matrix_path)
        
        # Initialize caches and lookup tables
        self._position_type_cache = {}  # Cache for position type lookups
        
        # Pre-initialize matrix attributes that will be filled later
        self._storage_positions = np.array([])
        self._rail_positions = np.array([])
        self._truck_positions = np.array([])
        self._storage_pos_to_rowbay = {}
        
        # Create position mappings - this needs to happen before precalculate_location_matrices
        self._setup_position_mappings()
        
        # Now precalculate location matrices using the position mappings
        self._precalculate_location_matrices()
        
        # Initialize environment components with tensor-based versions
        self.storage_yard = self._create_tensor_storage_yard()
        self.cranes = self._create_tensor_cranes(num_cranes)
        self.truck_queue = VehicleQueue(vehicle_type="Truck")
        self.train_queue = VehicleQueue(vehicle_type="Train")
        
        # Terminal trucks for handling swap bodies and trailers
        self.terminal_trucks = [TerminalTruck(f"TTR{i+1}") for i in range(num_terminal_trucks)]
        self.terminal_truck_available_times = np.zeros(num_terminal_trucks)
        
        # Track current state
        self.current_simulation_time = 0.0
        self.max_simulation_time = max_simulation_time
        self.crane_available_times = np.zeros(num_cranes)
        
        self.trucks_in_terminal = {}
        self.trains_in_terminal = {}
        
        # Create the tensor-based action mask generator
        self.action_mask_generator = TensorActionMaskGenerator(self, device=self.device)
        
        # Define action and observation spaces
        self._setup_spaces()

        # Base simulation date for scheduling
        self.base_simulation_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.current_simulation_datetime = self.base_simulation_date
        
        # Generate train arrival schedule
        self.train_schedule = self.config.generate_train_arrival_schedule(
            n_trains=20,
            base_date=self.base_simulation_date
        )

        # Initialize simplified rendering flag
        self.simplified_rendering = False

        # Initialize the environment state
        self.reset()
        
        # Performance tracking
        self.mask_generation_times = []
        self.action_exec_times = []

    def _load_config(self, config_path):
        """Load terminal configuration."""
        from simulation.config import TerminalConfig
        return TerminalConfig(config_path)
    
    def _create_terminal(self):
        """Create the terminal simulation."""
        return ContainerTerminal(
            layout_order=['rails', 'parking', 'driving_lane', 'yard_storage'],
            num_railtracks=6,      
            num_railslots_per_track=29,
            num_storage_rows=5,
            parking_to_railslot_ratio=1.0,
            storage_to_railslot_ratio=2.0,
            rail_slot_length=24.384,
            track_width=2.44,
            space_between_tracks=2.05,
            space_rails_to_parking=1.05,
            space_driving_to_storage=0.26,
            parking_width=4.0,
            driving_lane_width=4.0,
            storage_slot_width=2.5
        )
    
    def _create_tensor_storage_yard(self):
        """Create a tensor-based storage yard."""
        # Define special areas for different container types
        special_areas = {
            'reefer': [],
            'dangerous': [],
            'trailer': [],        # Section for trailers
            'swap_body': []       # Section for swap bodies
        }
        
        # Add reefer areas in first and last column of each row
        for row in self.terminal.storage_row_names:
            special_areas['reefer'].append((row, 1, 1))
            special_areas['reefer'].append((row, self.terminal.num_storage_slots_per_row, self.terminal.num_storage_slots_per_row))
            
        # Add dangerous goods area in middle columns
        for row in self.terminal.storage_row_names:
            special_areas['dangerous'].append((row, 33, 35))
        
        # Add trailer and swap body areas - only in the first row (closest to driving lane)
        first_row = self.terminal.storage_row_names[0]
        special_areas['trailer'].append((first_row, 5, 15))
        special_areas['swap_body'].append((first_row, 20, 30))
        
        return TensorStorageYard(
            num_rows=self.terminal.num_storage_rows,
            num_bays=self.terminal.num_storage_slots_per_row,
            max_tier_height=5,
            row_names=self.terminal.storage_row_names,
            special_areas=special_areas,
            device=self.device
        )
    
    def _create_tensor_cranes(self, num_cranes):
        """Create tensor-based RMG cranes with divided operational areas."""
        cranes = []
        bays_per_crane = self.terminal.num_storage_slots_per_row // num_cranes
        
        for i in range(num_cranes):
            start_bay = i * bays_per_crane
            end_bay = (i + 1) * bays_per_crane - 1 if i < num_cranes - 1 else self.terminal.num_storage_slots_per_row - 1
            
            crane = TensorRMGCrane(
                crane_id=f"RMG{i+1}",
                terminal=self.terminal,
                start_bay=start_bay,
                end_bay=end_bay,
                current_position=(start_bay, 0),
                device=self.device
            )
            cranes.append(crane)
        
        return cranes
    
    def _setup_position_mappings(self):
        """Set up mappings between positions and indices."""
        # Create parking spot mapping
        self.parking_spots = [f"p_{i+1}" for i in range(self.terminal.num_parking_spots)]
        
        # Create rail slot mapping
        self.rail_slots = {}
        for track in self.terminal.track_names:
            self.rail_slots[track] = [f"{track.lower()}_{i+1}" for i in range(self.terminal.num_railslots_per_track)]
        
        # Create position to index mapping
        all_positions = []
        
        # Add rail slots
        for rail_list in self.rail_slots.values():
            all_positions.extend(rail_list)
            
        # Add parking spots and storage positions
        all_positions.extend(self.parking_spots)
        
        # Add storage positions
        storage_positions = [f"{row}{i+1}" for row in self.terminal.storage_row_names 
                           for i in range(self.terminal.num_storage_slots_per_row)]
        all_positions.extend(storage_positions)
        
        # Create mapping dictionaries
        self.position_to_idx = {pos: i for i, pos in enumerate(all_positions)}
        self.idx_to_position = {i: pos for i, pos in enumerate(all_positions)}
        
        # Pre-calculate position types for faster lookups
        for pos in all_positions:
            self._position_type_cache[pos] = self._get_position_type_direct(pos)
    
    def _precalculate_location_matrices(self):
        """Pre-calculate matrices for faster location lookups."""
        # Create binary matrices for position types
        num_positions = len(self.position_to_idx)
        self._storage_positions = np.zeros(num_positions, dtype=bool)
        self._rail_positions = np.zeros(num_positions, dtype=bool)
        self._truck_positions = np.zeros(num_positions, dtype=bool)
        
        # Fill matrices based on position types
        for pos, idx in self.position_to_idx.items():
            pos_type = self._position_type_cache[pos]
            if pos_type == 'storage':
                self._storage_positions[idx] = True
            elif pos_type == 'train':
                self._rail_positions[idx] = True
            elif pos_type == 'truck':
                self._truck_positions[idx] = True
        
        # Create a map from positions to rows/bays for storage positions
        self._storage_pos_to_rowbay = {}
        
        # Extract row and bay number for each storage position
        for row in self.terminal.storage_row_names:
            for bay in range(1, self.terminal.num_storage_slots_per_row + 1):
                position = f"{row}{bay}"
                if position in self.position_to_idx:
                    self._storage_pos_to_rowbay[position] = (row, bay)
    
    def _setup_spaces(self):
        """Set up action and observation spaces."""
        num_positions = len(self.position_to_idx)
        num_cranes = len(self.cranes)
        num_terminal_trucks = len(self.terminal_trucks)
        
        # Action space: action_type, crane_id, source, destination or truck_idx, parking_spot
        self.action_space = spaces.Dict({
            'action_type': spaces.Discrete(3),  # 0: crane movement, 1: truck parking, 2: terminal truck 
            'crane_movement': spaces.MultiDiscrete([
                num_cranes,        # Crane index
                num_positions,     # Source position index
                num_positions      # Destination position index
            ]),
            'truck_parking': spaces.MultiDiscrete([
                10,                # Max trucks in queue to consider
                len(self.parking_spots)  # Parking spot index
            ]),
            'terminal_truck': spaces.MultiDiscrete([
                num_terminal_trucks,  # Terminal truck index
                num_positions,       # Source position index
                num_positions        # Destination position index
            ])
        })
        
        # Create an action mask space
        self.action_mask_space = spaces.Dict({
            'crane_movement': spaces.Box(
                low=0,
                high=1,
                shape=(num_cranes, num_positions, num_positions),
                dtype=np.int8
            ),
            'truck_parking': spaces.Box(
                low=0,
                high=1,
                shape=(10, len(self.parking_spots)),
                dtype=np.int8
            ),
            'terminal_truck': spaces.Box(
                low=0,
                high=1,
                shape=(num_terminal_trucks, num_positions, num_positions),
                dtype=np.int8
            )
        })
        
        # Observation space - enhanced for tensor-based operations
        self.observation_space = spaces.Dict({
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
            'terminal_truck_available_times': spaces.Box(
                low=0,
                high=np.inf,
                shape=(num_terminal_trucks,),
                dtype=np.float32
            ),
            'current_time': spaces.Box(
                low=0,
                high=np.inf,
                shape=(1,),
                dtype=np.float32
            ),
            'yard_state': spaces.Box(
                low=0,
                high=1,
                shape=(self.terminal.num_storage_rows, self.terminal.num_storage_slots_per_row, 11),  # Enhanced feature space
                dtype=np.float32
            ),
            'parking_status': spaces.Box(
                low=0,
                high=1,
                shape=(len(self.parking_spots),),
                dtype=np.int32
            ),
            'rail_status': spaces.Box(
                low=0,
                high=1,
                shape=(len(self.terminal.track_names), self.terminal.num_railslots_per_track),
                dtype=np.int32
            ),
            'queue_sizes': spaces.Box(
                low=0,
                high=np.inf,
                shape=(2,),  # [truck_queue_size, train_queue_size]
                dtype=np.int32
            ),
            'action_mask': self.action_mask_space
        })
    
    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state."""
        super().reset(seed=seed)
        # Reset container ID storage and time tracking
        self.stored_container_ids = []
        self.current_simulation_time = 0.0
        self.current_simulation_datetime = self.base_simulation_date
        self.crane_available_times = np.zeros(len(self.cranes))
        self.terminal_truck_available_times = np.zeros(len(self.terminal_trucks))
        
        # Clear terminal state
        self.trucks_in_terminal = {}
        self.trains_in_terminal = {}
        self.truck_queue.clear()
        self.train_queue.clear()
        
        # Reset storage yard
        self.storage_yard.clear()
        
        # Reset cranes to initial positions
        for i, crane in enumerate(self.cranes):
            start_bay = i * (self.terminal.num_storage_slots_per_row // len(self.cranes))
            crane.reset(position=(start_bay, 0))
        
        # Reset terminal trucks
        for truck in self.terminal_trucks:
            truck.containers = []
        
        # Initialize with some random containers in the storage yard
        self._initialize_storage_yard()
        
        # Schedule trains and trucks
        self._schedule_trains()
        self._schedule_trucks_for_existing_containers()
        
        # Get initial observation using tensor-based method
        observation = self._get_observation_tensor()
        # Convert tensors to numpy arrays for gym compatibility
        observation = self._tensors_to_numpy(observation)
        
        info = {}
        
        return observation, info
    
    def _tensors_to_numpy(self, observation_dict):
        """Convert tensor observations to numpy arrays for gym compatibility."""
        result = {}
        for key, value in observation_dict.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.cpu().numpy()
            elif isinstance(value, dict):
                result[key] = self._tensors_to_numpy(value)
            else:
                result[key] = value
        return result
    def create_wait_action(self):
        """
        Create a properly formatted wait action with valid indices for this environment instance.
        This ensures the wait action uses indices that are valid in this specific environment.
        
        Returns:
            Dict: A properly formatted wait action
        """
        # Create a wait action with valid indices for this environment
        # Use the first available indices to ensure they're valid
        first_crane_idx = 0
        first_position_idx = 0  # First valid position index
        
        # Ensure we have valid position indices
        if len(self.idx_to_position) > 0:
            first_position_idx = min(self.idx_to_position.keys())
        
        # Create the wait action with proper, valid indices
        wait_action = {
            'action_type': 0,  # Crane movement (arbitrary, but must be valid)
            'crane_movement': np.array([first_crane_idx, first_position_idx, first_position_idx], dtype=np.int32),
            'truck_parking': np.array([0, 0], dtype=np.int32),
            'terminal_truck': np.array([0, first_position_idx, first_position_idx], dtype=np.int32)
        }
        
        return wait_action

    def step(self, action):
        """Take a step in the environment by executing the specified action."""
        # Get current observation with action mask
        start_time = time.time()
        current_obs = self._get_observation_tensor()
        mask_time = time.time() - start_time
        
        if self.log_performance:
            self.mask_generation_times.append(mask_time)
        
        # Determine action type
        action_type = action['action_type']
        
        start_time = time.time()
        
        # Detect if this is a wait action (no movement)
        is_wait_action = False
        if action_type == 0:  # Crane movement
            if isinstance(action['crane_movement'], np.ndarray) and np.array_equal(action['crane_movement'], np.array([0, 0, 0])):
                is_wait_action = True
            elif isinstance(action['crane_movement'], torch.Tensor) and torch.all(action['crane_movement'] == 0):
                is_wait_action = True
        
        # Handle wait action specially - advance time without actual movement
        if is_wait_action:
            # Small time advancement (1 minute)
            time_advance = 60
            self.current_simulation_time += time_advance
            self.current_simulation_datetime += timedelta(seconds=time_advance)
            
            # Process vehicle arrivals and departures
            self._process_vehicle_arrivals(time_advance)
            self._process_vehicle_departures()
            
            # Return updated observation with no reward
            observation = self._get_observation_tensor()
            reward = 0
            terminated = self.current_simulation_time >= self.max_simulation_time
            truncated = False
            info = {"action": "wait", "time_advanced": time_advance}
            return observation, reward, terminated, truncated, info
        
        # Convert numpy action to tensor format if needed
        if isinstance(action['crane_movement'], np.ndarray):
            crane_movement = torch.tensor(action['crane_movement'], device=self.device)
            truck_parking = torch.tensor(action['truck_parking'], device=self.device)
            terminal_truck = torch.tensor(action.get('terminal_truck', [0, 0, 0]), device=self.device)
        else:
            crane_movement = action['crane_movement']
            truck_parking = action['truck_parking']
            terminal_truck = action['terminal_truck']
        
        if action_type == 0:  # Crane movement
            crane_idx, source_idx, destination_idx = crane_movement
            
            # Check if the crane movement action is valid according to the mask
            crane_mask = current_obs['action_mask']['crane_movement']
            if isinstance(crane_mask, torch.Tensor):
                valid = crane_mask[crane_idx, source_idx, destination_idx].item() == 1
            else:
                valid = crane_mask[crane_idx, source_idx, destination_idx] == 1
                
            if not valid:
                # Try to find a valid action instead
                if isinstance(crane_mask, torch.Tensor):
                    valid_actions = torch.nonzero(crane_mask == 1)
                else:
                    valid_actions = np.argwhere(crane_mask == 1)
                    
                if len(valid_actions) > 0:
                    valid_idx = np.random.randint(0, len(valid_actions))
                    if isinstance(valid_actions, torch.Tensor):
                        crane_idx, source_idx, destination_idx = valid_actions[valid_idx]
                    else:
                        crane_idx, source_idx, destination_idx = valid_actions[valid_idx]
                else:
                    # No valid crane actions - wait until next time
                    observation = current_obs
                    observation = self._tensors_to_numpy(observation)
                    reward = 0
                    terminated = self.current_simulation_time >= self.max_simulation_time
                    truncated = False
                    info = {"action": "wait", "reason": "No valid crane actions available"}
                    return observation, reward, terminated, truncated, info
            
            # Execute crane movement action
            result = self._execute_crane_movement(int(crane_idx), int(source_idx), int(destination_idx))
        
        elif action_type == 1:  # Truck parking
            truck_idx, parking_spot_idx = truck_parking
            
            # Check if the truck parking action is valid
            truck_mask = current_obs['action_mask']['truck_parking']
            if isinstance(truck_mask, torch.Tensor):
                valid = truck_mask[truck_idx, parking_spot_idx].item() == 1
            else:
                valid = truck_mask[truck_idx, parking_spot_idx] == 1
                
            if not valid:
                # Try to find a valid action instead
                if isinstance(truck_mask, torch.Tensor):
                    valid_actions = torch.nonzero(truck_mask == 1)
                else:
                    valid_actions = np.argwhere(truck_mask == 1)
                    
                if len(valid_actions) > 0:
                    valid_idx = np.random.randint(0, len(valid_actions))
                    if isinstance(valid_actions, torch.Tensor):
                        truck_idx, parking_spot_idx = valid_actions[valid_idx]
                    else:
                        truck_idx, parking_spot_idx = valid_actions[valid_idx]
                else:
                    # No valid truck parking actions - wait until next time
                    observation = current_obs
                    observation = self._tensors_to_numpy(observation)
                    reward = 0
                    terminated = self.current_simulation_time >= self.max_simulation_time
                    truncated = False
                    info = {"action": "wait", "reason": "No valid truck parking actions available"}
                    return observation, reward, terminated, truncated, info
            
            # Execute truck parking action
            result = self._execute_truck_parking(int(truck_idx), int(parking_spot_idx))
            
        elif action_type == 2:  # Terminal truck action
            truck_idx, source_idx, destination_idx = terminal_truck
            
            # Check if the terminal truck action is valid
            terminal_mask = current_obs['action_mask']['terminal_truck']
            if isinstance(terminal_mask, torch.Tensor):
                valid = terminal_mask[truck_idx, source_idx, destination_idx].item() == 1
            else:
                valid = terminal_mask[truck_idx, source_idx, destination_idx] == 1
                
            if not valid:
                # Try to find a valid action instead
                if isinstance(terminal_mask, torch.Tensor):
                    valid_actions = torch.nonzero(terminal_mask == 1)
                else:
                    valid_actions = np.argwhere(terminal_mask == 1)
                    
                if len(valid_actions) > 0:
                    valid_idx = np.random.randint(0, len(valid_actions))
                    if isinstance(valid_actions, torch.Tensor):
                        truck_idx, source_idx, destination_idx = valid_actions[valid_idx]
                    else:
                        truck_idx, source_idx, destination_idx = valid_actions[valid_idx]
                else:
                    # No valid terminal truck actions - wait until next time
                    observation = current_obs
                    observation = self._tensors_to_numpy(observation)
                    reward = 0
                    terminated = self.current_simulation_time >= self.max_simulation_time
                    truncated = False
                    info = {"action": "wait", "reason": "No valid terminal truck actions available"}
                    return observation, reward, terminated, truncated, info
            
            # Execute terminal truck action
            result = self._execute_terminal_truck_movement(int(truck_idx), int(source_idx), int(destination_idx))
        
        action_time = time.time() - start_time
        if self.log_performance:
            self.action_exec_times.append(action_time)
        
        # Get the tensor observation and convert to numpy for gym
        observation, reward, terminated, truncated, info = result
        if isinstance(observation, dict) and any(isinstance(v, torch.Tensor) for v in observation.values()):
            observation = self._tensors_to_numpy(observation)
            
        return observation, reward, terminated, truncated, info

    def _execute_terminal_truck_movement(self, truck_idx, source_idx, destination_idx):
        """Execute a terminal truck movement action."""
        source_position = self.idx_to_position[source_idx]
        destination_position = self.idx_to_position[destination_idx]
        
        # Check if the selected terminal truck is available
        if self.current_simulation_time < self.terminal_truck_available_times[truck_idx]:
            # Truck is not available yet - skip to when it becomes available
            time_advanced = self.terminal_truck_available_times[truck_idx] - self.current_simulation_time
            self.current_simulation_time = self.terminal_truck_available_times[truck_idx]
            
            # Process time advancement
            self._process_time_advancement(time_advanced)
            
            # Return observation with no reward (waiting is neutral)
            observation = self._get_observation_tensor()
            reward = 0
            terminated = self.current_simulation_time >= self.max_simulation_time
            truncated = False
            info = {"action": "wait", "time_advanced": time_advanced}
            return observation, reward, terminated, truncated, info
        
        # Get terminal truck and source container
        terminal_truck = self.terminal_trucks[truck_idx]
        container = self._get_container_at_position(source_position)
        
        # Only allow terminal trucks to move swap bodies and trailers
        if container is None or container.container_type not in ["Trailer", "Swap Body"]:
            observation = self._get_observation_tensor()
            reward = -1  # Penalty for invalid container type
            terminated = self.current_simulation_time >= self.max_simulation_time
            truncated = False
            info = {"action": "terminal_truck", "result": "invalid_container_type"}
            return observation, reward, terminated, truncated, info
        
        # Calculate time needed for the movement (faster than crane)
        # Simple distance-based calculation
        source_pos = self.terminal.positions.get(source_position, (0, 0))
        dest_pos = self.terminal.positions.get(destination_position, (0, 0))
        distance = np.sqrt(np.sum((np.array(source_pos) - np.array(dest_pos))**2))
        
        # Terminal trucks move at 25 km/h = ~7 m/s
        terminal_truck_speed = 7.0  # m/s
        time_taken = max(60, distance / terminal_truck_speed + 120)  # At least 1 minute, plus 2 minutes for loading/unloading
        
        # Remove container from source position
        if self._is_storage_position(source_position):
            removed_container = self.storage_yard.remove_container(source_position)
        else:
            removed_container = None  # Not implemented for other source types
        
        # Place container at destination (only storage positions supported)
        success = False
        if removed_container is not None and self._is_storage_position(destination_position):
            success = self.storage_yard.add_container(destination_position, removed_container)
        
        # Calculate reward (higher for freeing up trailer/swap body spots)
        if success:
            # Check if we freed up a valuable spot
            is_trailer_area = self._is_in_special_area(source_position, 'trailer')
            is_swap_body_area = self._is_in_special_area(source_position, 'swap_body')
            
            if is_trailer_area or is_swap_body_area:
                # Higher reward for freeing up specialized areas
                reward = 5.0  # Significant bonus
            else:
                # Lower reward for regular moves
                reward = 2.0
        else:
            # Failed to move container
            reward = -2.0
            # Put container back
            if removed_container is not None:
                self.storage_yard.add_container(source_position, removed_container)
        
        # Update terminal truck availability time
        self.terminal_truck_available_times[truck_idx] = self.current_simulation_time + time_taken
        
        # Check if any trucks are still available
        if not any(t <= self.current_simulation_time for t in self.terminal_truck_available_times):
            # All terminal trucks busy, advance to earliest available
            next_available_time = min(self.terminal_truck_available_times)
            time_advanced = next_available_time - self.current_simulation_time
            self.current_simulation_time = next_available_time
            
            # Process time advancement
            self._process_time_advancement(time_advanced)
        
        # Get the next observation
        observation = self._get_observation_tensor()
        
        # Check if the episode is over
        terminated = self.current_simulation_time >= self.max_simulation_time
        truncated = False
        
        # Additional info
        info = {
            "action_type": "terminal_truck",
            "time_taken": time_taken,
            "container_moved": removed_container.container_id if removed_container else None,
            "success": success,
            "current_time": self.current_simulation_time
        }
        
        return observation, reward, terminated, truncated, info
    
    def _is_in_special_area(self, position, area_type):
        """Check if a position is in a special area like trailer or swap body section."""
        if not self._is_storage_position(position):
            return False
            
        if position not in self._storage_pos_to_rowbay:
            return False
            
        row, bay = self._storage_pos_to_rowbay[position]
        
        for area_row, start_bay, end_bay in self.storage_yard.special_areas.get(area_type, []):
            if row == area_row and start_bay <= bay <= end_bay:
                return True
                
        return False

    def _initialize_storage_yard(self):
        """Initialize the storage yard with random containers."""
        # Fill about 30% of the yard with random containers
        num_rows = self.terminal.num_storage_rows
        num_bays = self.terminal.num_storage_slots_per_row
        num_positions = num_rows * num_bays
        num_to_fill = int(num_positions * 0.3)
        
        # Randomly select positions to fill
        positions_to_fill = np.random.choice(num_positions, num_to_fill, replace=False)
        self.stored_container_ids = []
        
        for pos_idx in positions_to_fill:
            # Convert flat index to row, bay
            row_idx = pos_idx // num_bays
            bay_idx = pos_idx % num_bays
            
            # Convert to position string
            row = self.terminal.storage_row_names[row_idx]
            position = f"{row}{bay_idx+1}"
            
            # Create a random container using probability model
            container = ContainerFactory.create_random(config=self.config)
            self.stored_container_ids.append(container.container_id)
            
            # Calculate priority based on pickup wait time
            wait_time = self.config.sample_from_kde('pickup_wait', n_samples=1, min_val=0, max_val=72)[0]
            container.priority = self._calculate_priority_from_wait_time(wait_time)
            
            # Set departure date
            container.departure_date = self.current_simulation_datetime + timedelta(hours=wait_time)
            
            # Respect trailer/swap body placement constraint
            if container.container_type in ["Trailer", "Swap Body"]:
                # Check if this is a valid area for trailer/swap body
                if container.container_type == "Trailer" and not self._is_in_special_area(position, 'trailer'):
                    container = ContainerFactory.create_random(config=self.config)
                    if container.container_type in ["Trailer", "Swap Body"]:
                        continue
                elif container.container_type == "Swap Body" and not self._is_in_special_area(position, 'swap_body'):
                    container = ContainerFactory.create_random(config=self.config)
                    if container.container_type in ["Trailer", "Swap Body"]:
                        continue
            
            # Add to storage yard
            if self.storage_yard.add_container(position, container):
                # Randomly add a second container (20% chance)
                if np.random.random() < 0.2:
                    container2 = ContainerFactory.create_random(config=self.config)
                    if container2.can_stack_with(container):
                        self.storage_yard.add_container(position, container2, tier=2)
                        self.stored_container_ids.append(container2.container_id)
    
    def _schedule_trains(self):
        """Schedule trains based on KDE model."""
        for train_id, planned_arrival, realized_arrival in self.train_schedule:
            # Convert datetime to seconds from simulation start
            arrival_time_seconds = (realized_arrival - self.base_simulation_date).total_seconds()
            
            if arrival_time_seconds < 0 or arrival_time_seconds > self.max_simulation_time:
                continue  # Skip trains outside simulation timeframe
            
            # Create train with random container setup
            num_wagons = random.randint(5, 10)
            train = Train(train_id=train_id, num_wagons=num_wagons)
            
            # Randomly fill some wagons with containers
            for _ in range(random.randint(1, num_wagons)):
                container = ContainerFactory.create_random(config=self.config)
                train.add_container(container)
            
            # Schedule the train arrival
            self.train_queue.schedule_arrival(train, realized_arrival)
    
    def _schedule_trucks_for_existing_containers(self):
        """Schedule trucks to pick up existing containers in storage."""
        if not self.stored_container_ids:
            return
                
        # Generate pickup schedule for stored containers
        pickup_schedule = self.config.generate_truck_pickup_schedule(
            self.stored_container_ids,
            base_date=self.base_simulation_date
        )
        
        # Create and schedule trucks for each container
        for container_id, pickup_time in pickup_schedule.items():
            # Create a pickup truck
            truck = Truck(truck_id=f"TRK{container_id}")
            truck.add_pickup_container_id(container_id)
            
            # Calculate priority based on pickup time
            wait_time_hours = (pickup_time - self.current_simulation_datetime).total_seconds() / 3600
            truck.priority = self._calculate_priority_from_wait_time(wait_time_hours)
            
            # Schedule truck arrival
            self.truck_queue.schedule_arrival(truck, pickup_time)
    
    def _calculate_priority_from_wait_time(self, wait_time):
        """Calculate container priority based on wait time."""
        priority = 100  # Base priority
        
        if wait_time < 24:      # Less than a day
            priority -= 50
        elif wait_time < 48:    # Less than two days
            priority -= 30
        elif wait_time < 72:    # Less than three days
            priority -= 10
        
        return max(1, priority)
    
    def _execute_crane_movement(self, crane_idx, source_idx, destination_idx):
        """Execute a crane movement action."""
        source_position = self.idx_to_position[source_idx]
        destination_position = self.idx_to_position[destination_idx]
        
        # Check if the selected crane is available
        if self.current_simulation_time < self.crane_available_times[crane_idx]:
            # Crane is not available yet - skip to when it becomes available
            time_advanced = self.crane_available_times[crane_idx] - self.current_simulation_time
            self.current_simulation_time = self.crane_available_times[crane_idx]
            
            # Process time advancement
            self._process_time_advancement(time_advanced)
            
            # Return observation with no reward (waiting is neutral)
            observation = self._get_observation_tensor()
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
        
        # Calculate the reward
        reward = self._calculate_reward(container, source_position, destination_position, time_taken)
        
        # Update crane availability time
        self.crane_available_times[crane_idx] = self.current_simulation_time + time_taken
        
        # Check if any crane is still available at the current time
        if not np.any(self.crane_available_times <= self.current_simulation_time):
            # All cranes busy, advance to earliest available
            next_available_time = np.min(self.crane_available_times)
            time_advanced = next_available_time - self.current_simulation_time
            self.current_simulation_time = next_available_time
            
            # Process time advancement
            self._process_time_advancement(time_advanced)
        
        # Get the next observation
        observation = self._get_observation_tensor()
        
        # Check if the episode is over
        terminated = self.current_simulation_time >= self.max_simulation_time
        truncated = False
        
        # Additional info
        info = {
            "action_type": "crane_movement",
            "time_taken": time_taken,
            "container_moved": container.container_id if container else None,
            "crane_position": crane.current_position,
            "trucks_waiting": self.truck_queue.size(),
            "trains_waiting": self.train_queue.size(),
            "current_time": self.current_simulation_time
        }
        
        return observation, reward, terminated, truncated, info
    
    def _execute_truck_parking(self, truck_idx, parking_spot_idx):
        """Execute a truck parking action."""
        # Get the truck from the queue
        if truck_idx >= self.truck_queue.size():
            # Invalid truck index
            observation = self._get_observation_tensor()
            reward = -1  # Small penalty for invalid action
            terminated = self.current_simulation_time >= self.max_simulation_time
            truncated = False
            info = {"action_type": "truck_parking", "result": "invalid_truck_index"}
            return observation, reward, terminated, truncated, info
        
        # Get the parking spot
        parking_spot = self.parking_spots[parking_spot_idx]
        
        # Check if the parking spot is available
        if parking_spot in self.trucks_in_terminal:
            # Parking spot already occupied
            observation = self._get_observation_tensor()
            reward = -1  # Small penalty for invalid action
            terminated = self.current_simulation_time >= self.max_simulation_time
            truncated = False
            info = {"action_type": "truck_parking", "result": "parking_spot_occupied"}
            return observation, reward, terminated, truncated, info
        
        # Get the truck (without removing it yet)
        truck_list = list(self.truck_queue.vehicles.queue)
        truck = truck_list[truck_idx]
        
        # Perform the assignment
        self.truck_queue.vehicles.queue.remove(truck)  # Remove from queue
        truck.parking_spot = parking_spot
        truck.status = "waiting"
        self.trucks_in_terminal[parking_spot] = truck
        
        # Calculate reward
        reward = self._calculate_truck_parking_reward(truck, parking_spot)
        
        # Advance time slightly (1 minute = 60 seconds)
        time_advanced = 60.0
        self.current_simulation_time += time_advanced
        
        # Process time advancement
        self._process_time_advancement(time_advanced)
        
        # Get the next observation
        observation = self._get_observation_tensor()
        
        # Check if the episode is over
        terminated = self.current_simulation_time >= self.max_simulation_time
        truncated = False
        
        # Additional info
        info = {
            "action_type": "truck_parking",
            "time_taken": time_advanced,
            "truck_id": truck.truck_id,
            "parking_spot": parking_spot,
            "trucks_waiting": self.truck_queue.size(),
            "trains_waiting": self.train_queue.size(),
            "current_time": self.current_simulation_time
        }
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_truck_parking_reward(self, truck, parking_spot):
        """Calculate reward for parking a truck."""
        # Base reward for successful truck parking
        reward = 1.0
        
        # Check if truck is empty (pickup truck)
        if truck.is_pickup_truck and truck.pickup_container_ids:
            target_position = None
            
            # Find the wagon that has the container this truck needs to pick up
            for track_id, train in self.trains_in_terminal.items():
                for i, wagon in enumerate(train.wagons):
                    for container in wagon.containers:
                        if container.container_id in truck.pickup_container_ids:
                            target_position = f"{track_id.lower()}_{i+1}"
                            break
                    if target_position:
                        break
                if target_position:
                    break
            
            # If we found a wagon with the needed container, check if truck is parked optimally
            if target_position:
                # Get the rail slot index
                track_id = target_position.split('_')[0]
                slot_num = int(target_position.split('_')[1])
                
                # Get the parallel parking spots and one spot on each side
                parallel_spots = []
                if slot_num > 1:
                    parallel_spots.append(f"p_{slot_num-1}")
                parallel_spots.append(f"p_{slot_num}")
                if slot_num < self.terminal.num_railslots_per_track:
                    parallel_spots.append(f"p_{slot_num+1}")
                
                # Check if truck is parked in an optimal spot
                if parking_spot in parallel_spots:
                    # Higher reward for exact parallel spot
                    if parking_spot == f"p_{slot_num}":
                        reward += 3.0
                    else:
                        reward += 2.0
                else:
                    # Penalty if not optimally placed, based on distance
                    parking_idx = int(parking_spot.split('_')[1])
                    distance = abs(parking_idx - slot_num)
                    distance_penalty = min(0.5 * distance, 5.0)  # Cap the penalty
                    reward -= distance_penalty
        
        # For delivery trucks (bringing containers)
        elif not truck.is_pickup_truck and truck.containers:
            # Check if truck has containers for specific wagons
            for container in truck.containers:
                if hasattr(container, 'destination_id'):
                    # Find the wagon that needs this container
                    for track_id, train in self.trains_in_terminal.items():
                        for i, wagon in enumerate(train.wagons):
                            if container.destination_id in wagon.pickup_container_ids:
                                target_position = f"{track_id.lower()}_{i+1}"
                                
                                # Get the rail slot index
                                slot_num = int(target_position.split('_')[1])
                                
                                # Get the parallel parking spots and one spot on each side
                                parallel_spots = []
                                if slot_num > 1:
                                    parallel_spots.append(f"p_{slot_num-1}")
                                parallel_spots.append(f"p_{slot_num}")
                                if slot_num < self.terminal.num_railslots_per_track:
                                    parallel_spots.append(f"p_{slot_num+1}")
                                
                                # Check if truck is parked in an optimal spot
                                if parking_spot in parallel_spots:
                                    # Higher reward for exact parallel spot
                                    if parking_spot == f"p_{slot_num}":
                                        reward += 3.0
                                    else:
                                        reward += 2.0
                                else:
                                    # Penalty for suboptimal placement
                                    parking_idx = int(parking_spot.split('_')[1])
                                    distance = abs(parking_idx - slot_num)
                                    distance_penalty = min(0.5 * distance, 5.0)  # Cap the penalty
                                    reward -= distance_penalty
                                break
                        if 'target_position' in locals():
                            break
        
        return reward
    def set_vehicle_limits(self, max_trucks=None, max_trains=None):
        """
        Set limits on the number of trucks and trains that can be generated per day
        
        Args:
            max_trucks: Maximum number of trucks per day (None for unlimited)
            max_trains: Maximum number of trains per day (None for unlimited)
        """
        self.max_trucks_per_day = max_trucks
        self.max_trains_per_day = max_trains
        self.daily_truck_count = 0
        self.daily_train_count = 0
        self.last_sim_day = 0
        
        # Store original functions if not already saved
        if not hasattr(self, 'original_schedule_trucks'):
            self.original_schedule_trucks = self._schedule_trucks_for_existing_containers
        
        if not hasattr(self, 'original_schedule_trains'):
            self.original_schedule_trains = self._schedule_trains
        
        # Override with limited versions
        if max_trucks is not None or max_trains is not None:
            self._schedule_trucks_for_existing_containers = self._limited_schedule_trucks
            self._schedule_trains = self._limited_schedule_trains
        else:
            # Reset to originals if limits removed
            self._schedule_trucks_for_existing_containers = self.original_schedule_trucks
            self._schedule_trains = self.original_schedule_trains

    def _limited_schedule_trucks(self):
        """Limited version of truck scheduling that respects max_trucks_per_day"""
        # Check if we've reached the limit
        if hasattr(self, 'max_trucks_per_day') and self.max_trucks_per_day is not None:
            # Calculate what day we're on
            sim_day = int(self.current_simulation_time / 86400)
            
            # Reset counter if it's a new day
            if sim_day > self.last_sim_day:
                self.daily_truck_count = 0
                self.last_sim_day = sim_day
            
            # Check if we're at the limit
            if self.daily_truck_count >= self.max_trucks_per_day:
                return  # Don't schedule more trucks
            
            # Count how many we're going to schedule
            available_slots = self.max_trucks_per_day - self.daily_truck_count
        else:
            available_slots = len(self.stored_container_ids)  # No limit
        
        # Call original but limit how many we schedule
        original_queue_size = self.truck_queue.size()
        self.original_schedule_trucks()
        
        # Count how many were added
        new_trucks = self.truck_queue.size() - original_queue_size
        self.daily_truck_count += new_trucks
        
        # Remove excess if we went over the limit
        if hasattr(self, 'max_trucks_per_day') and self.max_trucks_per_day is not None:
            excess = self.daily_truck_count - self.max_trucks_per_day
            if excess > 0:
                # Remove the excess trucks from the end of the queue
                for _ in range(excess):
                    # Find a truck in the queue that hasn't been assigned to the terminal yet
                    if not self.truck_queue.is_empty():
                        self.truck_queue.vehicles.queue.pop()
                self.daily_truck_count = self.max_trucks_per_day

    def _limited_schedule_trains(self):
        """Limited version of train scheduling that respects max_trains_per_day"""
        # Similar implementation to _limited_schedule_trucks
        if hasattr(self, 'max_trains_per_day') and self.max_trains_per_day is not None:
            # Calculate what day we're on
            sim_day = int(self.current_simulation_time / 86400)
            
            # Reset counter if it's a new day
            if sim_day > self.last_sim_day:
                self.daily_train_count = 0
                self.last_sim_day = sim_day
            
            # Check if we're at the limit
            if self.daily_train_count >= self.max_trains_per_day:
                return  # Don't schedule more trains
        
        # Call original implementation
        original_queue_size = self.train_queue.size()
        self.original_schedule_trains()
        
        # Count how many were added
        new_trains = self.train_queue.size() - original_queue_size
        self.daily_train_count += new_trains
        
        # Remove excess if we went over the limit
        if hasattr(self, 'max_trains_per_day') and self.max_trains_per_day is not None:
            excess = self.daily_train_count - self.max_trains_per_day
            if excess > 0:
                # Remove the excess trains from the end of the queue
                for _ in range(excess):
                    if not self.train_queue.is_empty():
                        self.train_queue.vehicles.queue.pop()
                self.daily_train_count = self.max_trains_per_day

    def _calculate_reward(self, container, source_position, destination_position, time_taken):
        """Calculate the reward for moving a container."""
        reward = 0.0  # Base reward
        
        # Get source and destination information
        source_type = self._get_position_type(source_position)
        dest_type = self._get_position_type(destination_position)
        
        # Calculate distance for container movement (source to destination)
        source_to_dest_distance = 0.0
        if hasattr(self.terminal, 'get_distance'):
            try:
                source_to_dest_distance = self.terminal.get_distance(source_position, destination_position)
            except:
                # If distance calculation fails, estimate based on time
                source_to_dest_distance = time_taken * 0.4  # Rough estimate - lower factor since time includes crane movement
        
        # Get crane that performed the operation
        crane_idx = None
        for i, crane in enumerate(self.cranes):
            if crane._is_position_in_crane_area(source_position, self.env.storage_yard) and crane._is_position_in_crane_area(destination_position, self.env.storage_yard):
                crane_idx = i
                break
        
        # Calculate crane position to source distance (for empty movement penalty)
        crane_to_source_distance = 0.0
        crane_to_source_time = 0.0
        if crane_idx is not None:
            crane = self.cranes[crane_idx]
            prev_position = crane.current_position  # Position before this move
            
            # Calculate approximate distance from previous position to source
            if hasattr(crane, '_calculate_position_to_source_time'):
                # Extract row and bay from source position
                if hasattr(self.storage_yard, 'position_to_indices') and source_position in self.storage_yard.position_to_indices:
                    src_indices = self.storage_yard.position_to_indices[source_position]
                    crane_to_source_time = crane._calculate_position_to_source_time(src_indices)
                    # Rough distance estimate based on time and speed
                    crane_to_source_distance = crane_to_source_time * crane.gantry_speed * 0.5
        
        # Empty crane movement penalty
        if container is None:
            empty_move_penalty = -5.0
            distance_time_penalty = -0.05 * source_to_dest_distance - time_taken / 60.0
            return empty_move_penalty + distance_time_penalty
        
        # Add penalty for empty crane movement from previous position to source
        empty_movement_penalty = 0.0
        if crane_to_source_distance > 0:
            # Apply smaller penalty for this implicit empty movement
            empty_movement_penalty = -0.02 * crane_to_source_distance - crane_to_source_time / 120.0
        
        # Determine the reward based on move type
        if source_type == 'train' and dest_type == 'truck':
            # GOLDEN MOVE: DIRECT TRAIN TO TRUCK
            move_type_reward = 10.0
        elif source_type == 'truck' and dest_type == 'train':
            # GOLDEN MOVE: DIRECT TRUCK TO TRAIN
            move_type_reward = 10.0
        elif source_type == 'storage' and (dest_type == 'truck' or dest_type == 'train'):
            # GOOD MOVE: STORAGE TO TRUCK OR TRAIN
            move_type_reward = 3.0
            
            # DEADLINE BONUS: Container moved before deadline
            if hasattr(container, 'departure_date') and container.departure_date:
                time_until_deadline = (container.departure_date - self.current_simulation_datetime).total_seconds()
                if time_until_deadline > 0:
                    # Scale reward based on time till deadline
                    time_factor = min(1.0, 24*3600 / max(3600, time_until_deadline))
                    deadline_bonus = 5.0 * time_factor
                    move_type_reward += deadline_bonus
        elif (source_type == 'train' or source_type == 'truck') and dest_type == 'storage':
            # STANDARD MOVES: TRAIN/TRUCK TO STORAGE
            move_type_reward = 2.0
        elif source_type == 'storage' and dest_type == 'storage':
            # RESHUFFLING: STORAGE TO STORAGE
            move_type_reward = -4.0
        elif source_type == 'truck' and dest_type == 'parking':
            # TRUCK PARKING: ASSIGN TRUCK TO SPOT
            move_type_reward = 2.0
        
        # Add the move type reward
        reward += move_type_reward
        
        # DEADLINE PENALTY: Container moved after deadline
        if hasattr(container, 'departure_date') and container.departure_date:
            time_past_deadline = (self.current_simulation_datetime - container.departure_date).total_seconds()
            if time_past_deadline > 0:
                past_deadline_hours = time_past_deadline / 3600
                deadline_penalty = -min(10.0, past_deadline_hours * 0.5)  # Cap at -10
                reward += deadline_penalty
        
        # PRIORITY BONUS: Based on container priority
        if hasattr(container, 'priority'):
            priority_factor = max(0, (100 - container.priority) / 100)
            priority_bonus = priority_factor * 2.0
            reward += priority_bonus
        
        # DISTANCE AND TIME PENALTY for the actual container movement
        source_to_dest_penalty = -0.02 * source_to_dest_distance  # -0.02 per meter
        time_penalty = -min(time_taken / 120, 1.0)  # Cap at -1 for moves over 2 minutes
        
        # Combine penalties for both empty movement and container movement
        total_distance_time_penalty = source_to_dest_penalty + time_penalty + empty_movement_penalty
        reward += total_distance_time_penalty
        
        # Special bonus for moving swap bodies and trailers
        if container and container.container_type in ["Trailer", "Swap Body"]:
            # Check if moved to appropriate area
            if dest_type == 'storage':
                if container.container_type == "Trailer" and self._is_in_special_area(destination_position, 'trailer'):
                    reward += 2.0  # Bonus for placing trailer in correct area
                elif container.container_type == "Swap Body" and self._is_in_special_area(destination_position, 'swap_body'):
                    reward += 2.0  # Bonus for placing swap body in correct area
            
            # Bonus for handling these special containers
            reward += 1.0
        
        # Store the crane position data for future reference
        if crane_idx is not None:
            self.cranes[crane_idx].previous_position = self.cranes[crane_idx].current_position
            
        return reward

    def _process_time_advancement(self, time_advanced):
        """Process events that occur during time advancement."""
        # Update current simulation datetime
        self.current_simulation_datetime += timedelta(seconds=time_advanced)
        
        # Process vehicle arrivals based on time advancement
        self._process_vehicle_arrivals(time_advanced)
        
        # Process vehicle departures
        self._process_vehicle_departures()
        
        # Signal for mask regeneration
        self.action_mask_generator.last_update_time = -1  # Force mask regeneration
    
    def _process_vehicle_arrivals(self, time_advanced):
        """Process vehicle arrivals based on elapsed time."""
        # Update queues with current time
        self.train_queue.update(self.current_simulation_datetime)
        self.truck_queue.update(self.current_simulation_datetime)
        
        # Generate additional trucks based on KDE model if needed
        truck_arrival_probability = min(0.8, time_advanced / 3600)  # Cap at 80% per hour
        if np.random.random() < truck_arrival_probability:
            # Create a random truck
            truck = Truck()
            
            # Sample arrival time
            truck_hour = self.config.sample_from_kde('truck_pickup', n_samples=1)[0]
            arrival_time = self.config.hours_to_datetime(truck_hour, self.current_simulation_datetime)
            
            # Randomly decide if bringing or picking up container
            if np.random.random() < 0.5:
                # Truck bringing a container
                container = ContainerFactory.create_random(config=self.config)
                truck.add_container(container)
            else:
                # Truck coming to pick up
                pickup_id = f"CONT{np.random.randint(1000, 9999)}"
                truck.add_pickup_container_id(pickup_id)
            
            self.truck_queue.schedule_arrival(truck, arrival_time)
        
        # Process train arrivals
        self._process_train_arrivals()
    
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
    
    def _process_vehicle_departures(self):
        """Process vehicle departures."""
        self._process_truck_departures()
        self._process_train_departures()
    
    def _process_truck_departures(self):
        """Process trucks that are ready to depart."""
        spots_to_remove = []
        
        for spot, truck in self.trucks_in_terminal.items():
            if (not truck.is_pickup_truck and not truck.containers) or \
               (truck.is_pickup_truck and not truck.pickup_container_ids):
                truck.status = "departed"
                spots_to_remove.append(spot)
        
        for spot in spots_to_remove:
            del self.trucks_in_terminal[spot]
    
    def _process_train_departures(self):
        """Process trains that are ready to depart."""
        tracks_to_remove = []
        
        for track, train in self.trains_in_terminal.items():
            # A train is ready to depart if all pickup requests are fulfilled
            if not any(len(wagon.pickup_container_ids) > 0 for wagon in train.wagons):
                train.status = "departed"
                tracks_to_remove.append(track)
        
        for track in tracks_to_remove:
            del self.trains_in_terminal[track]
    
    def _get_observation_tensor(self):
        """Get the current observation as tensors for faster processing."""
        # Create observation tensors with torch for GPU compatibility
        
        # Crane positions and availability
        crane_positions = torch.tensor([crane.current_position for crane in self.cranes], 
                                      dtype=torch.float32, device=self.device)
        crane_available_times = torch.tensor(self.crane_available_times, 
                                           dtype=torch.float32, device=self.device)
        terminal_truck_available_times = torch.tensor(self.terminal_truck_available_times, 
                                                    dtype=torch.float32, device=self.device)
        current_time = torch.tensor([self.current_simulation_time], 
                                   dtype=torch.float32, device=self.device)
        
        # Get yard state tensor
        if hasattr(self.storage_yard, 'get_state_representation') and callable(getattr(self.storage_yard, 'get_state_representation')):
            yard_state = self.storage_yard.get_state_representation()
        else:
            yard_state = torch.zeros((self.terminal.num_storage_rows, self.terminal.num_storage_slots_per_row, 11), 
                                    dtype=torch.float32, device=self.device)
        
        # Parking status - use binary tensor
        parking_status = torch.zeros(len(self.parking_spots), 
                                    dtype=torch.int8, device=self.device)
        for i, spot in enumerate(self.parking_spots):
            if spot in self.trucks_in_terminal:
                parking_status[i] = 1
        
        # Rail status - use binary tensor
        rail_status = torch.zeros((len(self.terminal.track_names), self.terminal.num_railslots_per_track), 
                                 dtype=torch.int8, device=self.device)
        for i, track in enumerate(self.terminal.track_names):
            if track in self.trains_in_terminal:
                rail_status[i, :] = 1
        
        # Queue sizes
        queue_sizes = torch.tensor([self.truck_queue.size(), self.train_queue.size()], 
                                 dtype=torch.int32, device=self.device)
        
        # Action masks (the most computationally expensive part)
        action_mask = self.action_mask_generator.generate_masks()
        
        # Return combined observation dict
        return {
            'crane_positions': crane_positions,
            'crane_available_times': crane_available_times,
            'terminal_truck_available_times': terminal_truck_available_times,
            'current_time': current_time,
            'yard_state': yard_state,
            'parking_status': parking_status,
            'rail_status': rail_status,
            'queue_sizes': queue_sizes,
            'action_mask': action_mask
        }
    
    def _get_container_at_position(self, position):
        """Helper to get container at a position."""
        # Use cached position type if available
        position_type = self._position_type_cache.get(position)
        
        if position_type == 'storage':
            # Get top container from storage
            container, _ = self.storage_yard.get_top_container(position)
            return container
        elif position_type == 'truck':
            # Get container from truck
            truck = self.trucks_in_terminal.get(position)
            if truck and hasattr(truck, 'containers') and truck.containers:
                return truck.containers[0]  # Return the first container
            return None
        elif position_type == 'train':
            # Parse train position
            parts = position.split('_')
            if len(parts) != 2:
                return None
                
            track_num = parts[0][1:]
            slot_num = int(parts[1])
            
            # Find the train
            track_id = f"T{track_num}"
            train = self.trains_in_terminal.get(track_id)
            
            if train and 0 <= slot_num - 1 < len(train.wagons):
                wagon = train.wagons[slot_num - 1]
                if wagon.containers:
                    return wagon.containers[0]
        return None
    
    def _get_position_type(self, position):
        """Determine the type of a position (train, truck, storage) with caching."""
        if position in self._position_type_cache:
            return self._position_type_cache[position]
        else:
            pos_type = self._get_position_type_direct(position)
            self._position_type_cache[position] = pos_type
            return pos_type
    
    def _get_position_type_direct(self, position):
        """Directly determine position type without caching."""
        if self._is_rail_position(position):
            return 'train'
        elif self._is_truck_position(position):
            return 'truck'
        else:
            return 'storage'
    
    def _is_storage_position(self, position):
        """Check if a position is in the storage yard."""
        if position in self._position_type_cache:
            return self._position_type_cache[position] == 'storage'
        
        # Look up position in the position index
        if position in self.position_to_idx:
            pos_idx = self.position_to_idx[position]
            if pos_idx < len(self._storage_positions):
                return self._storage_positions[pos_idx]
        
        # Fallback to regex check
        return bool(position and position[0].isalpha() and position[1:].isdigit())
    
    def _is_truck_position(self, position):
        """Check if a position is a truck parking spot."""
        if position in self._position_type_cache:
            return self._position_type_cache[position] == 'truck'
        
        # Look up position in the position index
        if position in self.position_to_idx:
            pos_idx = self.position_to_idx[position]
            if pos_idx < len(self._truck_positions):
                return self._truck_positions[pos_idx]
        
        # Fallback to string check
        return position.startswith('p_')
    
    def _is_rail_position(self, position):
        """Check if a position is a rail slot."""
        if position in self._position_type_cache:
            return self._position_type_cache[position] == 'train'
        
        # Look up position in the position index
        if position in self.position_to_idx:
            pos_idx = self.position_to_idx[position]
            if pos_idx < len(self._rail_positions):
                return self._rail_positions[pos_idx]
        
        # Fallback to string check
        return position.startswith('t') and '_' in position
    def set_simplified_rendering(self, simplified=True):
        """
        Set simplified rendering mode for faster training.
        
        Args:
            simplified: Whether to use simplified rendering
        """
        self.simplified_rendering = simplified
    def render(self, mode='human'):
        """
        Render the terminal environment with optimization for training.
        
        Args:
            mode: Rendering mode ('human' or 'rgb_array')
            
        Returns:
            Figure or numpy array depending on mode
        """
        if hasattr(self, 'simplified_rendering') and self.simplified_rendering and mode == 'human':
            # During training, don't actually render to save time
            # Just return a dummy figure object
            from matplotlib.figure import Figure
            return Figure()
        
        # Regular rendering for human viewing or rgb_array mode
        if mode == 'human':
            # Render the terminal for human viewing
            fig, ax = self.terminal.visualize(figsize=(15, 10), show_labels=True)
            
            # Add key simulation information
            title = f"Terminal Simulation - Time: {self.current_simulation_time:.1f}s"
            ax.set_title(title, fontsize=16)
            
            plt.close()  # Close the figure to avoid memory issues
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
        plt.close('all')
        # Move all tensors to CPU to free GPU memory
        if hasattr(self, 'device') and self.device != 'cpu':
            self.storage_yard.to_cpu()
            for crane in self.cranes:
                if hasattr(crane, 'to_cpu'):
                    crane.to_cpu()
            self.action_mask_generator.to_cpu()
            
    def print_performance_stats(self):
        """Print performance statistics of the environment."""
        if not self.log_performance:
            print("Performance logging is disabled. Set env.log_performance = True to enable.")
            return
            
        print("\n=== Performance Statistics ===")
        
        mask_times = np.array(self.mask_generation_times)
        action_times = np.array(self.action_exec_times)
        
        print(f"Action mask generation: {mask_times.mean()*1000:.2f}ms avg, {mask_times.max()*1000:.2f}ms max")
        print(f"Action execution: {action_times.mean()*1000:.2f}ms avg, {action_times.max()*1000:.2f}ms max")
        
        # Total environment step time
        total_step_time = mask_times + action_times
        print(f"Total step time: {total_step_time.mean()*1000:.2f}ms avg, {total_step_time.max()*1000:.2f}ms max")
        
        # If using tensor operations, compare with CPU vs GPU
        if hasattr(self, 'device') and self.device != 'cpu':
            print("\nSwitch to CPU for performance comparison:")
            print("env.storage_yard.to_cpu()")
            print("for crane in env.cranes: crane.to_cpu()")
            print("env.action_mask_generator.to_cpu()")
            print("env.device = 'cpu'")