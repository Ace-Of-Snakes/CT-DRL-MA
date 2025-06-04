# simulation/terminal_env_optimized.py (optimized version with bitmap storage)
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import pickle
from typing import List, Dict, Tuple, Optional, Any, Set
import os
import time
import re
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

# Import our custom components
from simulation.deprecated_components.terminal_layout.CTSimulator import ContainerTerminal
from simulation.terminal_components.Container import Container, ContainerFactory
from simulation.terminal_components.Train import Train
from simulation.terminal_components.Truck import Truck
from simulation.terminal_components.TerminalTruck import TerminalTruck
from simulation.deprecated_components.BitmapYard import BitmapStorageYard  # Updated import
from simulation.deprecated_components.RMGCrane import RMGCrane
from simulation.terminal_components.Vehicle_Queue import VehicleQueue


class OptimizedTerminalEnvironment(gym.Env):
    """
    Optimized Container Terminal Environment with GPU-accelerated bitmap storage.
    
    Key improvements:
    - GPU-accelerated storage yard operations
    - Vectorized action mask generation
    - Batch proximity calculations
    - Efficient move validation
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, 
                terminal_config_path: str = None,
                terminal_config=None,
                distance_matrix_path: str = None,
                max_simulation_time: float = 86400,  # 24 hours in seconds
                num_cranes: int = 2,
                num_terminal_trucks: int = 3,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the optimized terminal environment.
        
        Args:
            terminal_config_path: Path to terminal configuration
            terminal_config: Terminal configuration object
            distance_matrix_path: Path to distance matrix
            max_simulation_time: Maximum simulation time in seconds
            num_cranes: Number of RMG cranes
            num_terminal_trucks: Number of terminal trucks
            device: Computation device ('cuda' or 'cpu')
        """
        super(OptimizedTerminalEnvironment, self).__init__()

        # Device for GPU acceleration
        self.device = device
        print(f"Initializing Terminal Environment on device: {device}")

        # Initialize container ID storage
        self.stored_container_ids = []

        # Load or create terminal configuration
        self.config = terminal_config or self._load_config(terminal_config_path)

        # Create terminal
        self.terminal = self._create_terminal()
        
        # Load distance matrix if available
        if distance_matrix_path and os.path.exists(distance_matrix_path):
            self.terminal.load_distance_matrix(distance_matrix_path)
        
        # Initialize caches and lookup tables (now with GPU support)
        self._position_type_cache = {}
        self._valid_actions_cache = {}
        self._action_mask_cache = None
        self._last_action_mask_time = -1
        
        # Pre-initialize matrix attributes
        self._storage_positions = torch.tensor([], dtype=torch.bool, device=device)
        self._rail_positions = torch.tensor([], dtype=torch.bool, device=device)
        self._truck_positions = torch.tensor([], dtype=torch.bool, device=device)
        self._storage_pos_to_rowbay = {}
        
        # Initialize environment components with optimized storage
        self.storage_yard = self._create_optimized_storage_yard()
        self.cranes = self._create_cranes(num_cranes)
        self.truck_queue = VehicleQueue(vehicle_type="Truck")
        self.train_queue = VehicleQueue(vehicle_type="Train")
        
        # Terminal trucks for handling swap bodies and trailers
        self.terminal_trucks = [TerminalTruck(f"TTR{i+1}") for i in range(num_terminal_trucks)]
        self.terminal_truck_available_times = torch.zeros(num_terminal_trucks, device=device)
        
        # Track current state
        self.current_simulation_time = 0.0
        self.max_simulation_time = max_simulation_time
        self.crane_available_times = torch.zeros(num_cranes, device=device)
        
        self.trucks_in_terminal = {}
        self.trains_in_terminal = {}
        
        # Create position mappings
        self._setup_position_mappings()
        
        # Now precalculate location matrices using the position mappings
        self._precalculate_location_matrices()
        
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

        # Action mask caching
        self._action_mask_cache = None
        self._last_action_mask_time = -1
        self._last_truck_count = 0
        self._last_train_count = 0
        self._last_storage_container_count = 0
        self._action_mask_cache_duration = 300  # 5 minutes cache duration
        
        # Pre-compute valid destinations for different container types
        self._precompute_valid_destinations()
        
        # Performance monitoring
        self.performance_stats = {
            'action_mask_time': 0.0,
            'storage_operations': 0,
            'proximity_calculations': 0
        }

        # Initialize the environment state
        self.reset()

    def _load_config(self, config_path):
        """Load terminal configuration."""
        from simulation.TerminalConfig import TerminalConfig
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
    
    def _create_optimized_storage_yard(self):
        """Create the optimized bitmap-based storage yard."""
        # Define special areas for different container types
        special_areas = {
            'reefer': [],
            'dangerous': [],
            'trailer': [],
            'swap_body': []
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
        
        return BitmapStorageYard(
            num_rows=self.terminal.num_storage_rows,
            num_bays=self.terminal.num_storage_slots_per_row,
            max_tier_height=5,
            row_names=self.terminal.storage_row_names,
            special_areas=special_areas,
            device=self.device  # GPU acceleration
        )
    
    def _create_cranes(self, num_cranes):
        """Create RMG cranes with divided operational areas."""
        cranes = []
        bays_per_crane = self.terminal.num_storage_slots_per_row // num_cranes
        
        for i in range(num_cranes):
            start_bay = i * bays_per_crane
            end_bay = (i + 1) * bays_per_crane - 1 if i < num_cranes - 1 else self.terminal.num_storage_slots_per_row - 1
            
            crane = RMGCrane(
                crane_id=f"RMG{i+1}",
                terminal=self.terminal,
                start_bay=start_bay,
                end_bay=end_bay,
                current_position=(start_bay, 0)
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
        """Pre-calculate matrices for faster location lookups using GPU tensors."""
        # Create binary tensors for position types
        num_positions = len(self.position_to_idx)
        self._storage_positions = torch.zeros(num_positions, dtype=torch.bool, device=self.device)
        self._rail_positions = torch.zeros(num_positions, dtype=torch.bool, device=self.device)
        self._truck_positions = torch.zeros(num_positions, dtype=torch.bool, device=self.device)
        
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
        
        # Action space
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
        
        # Action mask space
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
        
        # Observation space
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
                shape=(22,),  # Compact features: 5+5+12 = 22
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
        self.crane_available_times = torch.zeros(len(self.cranes), device=self.device)
        self.terminal_truck_available_times = torch.zeros(len(self.terminal_trucks), device=self.device)
        
        # Clear terminal state
        self.trucks_in_terminal = {}
        self.trains_in_terminal = {}
        self.truck_queue.clear()
        self.train_queue.clear()
        self.storage_yard.clear()  # GPU-accelerated clear
        
        # Reset cranes to initial positions
        for i, crane in enumerate(self.cranes):
            start_bay = i * (self.terminal.num_storage_slots_per_row // len(self.cranes))
            crane.reset(position=(start_bay, 0))
        
        # Reset terminal trucks
        for truck in self.terminal_trucks:
            truck.containers = []
        
        # Clear caches
        self._valid_actions_cache = {}
        self._action_mask_cache = None
        self._last_action_mask_time = -1
        
        # Initialize with some random containers in the storage yard
        self._initialize_storage_yard()
        
        # Schedule trains and trucks
        self._schedule_trains()
        self._schedule_trucks_for_existing_containers()
        
        # Get initial observation
        observation = self._get_observation()
        info = {'performance_stats': self.performance_stats.copy()}
        
        return observation, info
    
    def _initialize_storage_yard(self):
        """Initialize the storage yard with random containers using GPU-accelerated operations."""
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
            
            # Add to storage yard (GPU-accelerated operation)
            if self.storage_yard.add_container(position, container):
                # Randomly add a second container (20% chance)
                if np.random.random() < 0.2:
                    container2 = ContainerFactory.create_random(config=self.config)
                    if container2.can_stack_with(container):
                        self.storage_yard.add_container(position, container2, tier=2)
                        self.stored_container_ids.append(container2.container_id)

    def _generate_action_masks_optimized(self):
        """Generate action masks using GPU-accelerated bitmap operations."""
        start_time = time.perf_counter()
        
        num_cranes = len(self.cranes)
        num_terminal_trucks = len(self.terminal_trucks)
        num_positions = len(self.position_to_idx)
        
        # Pre-allocate arrays
        crane_action_mask = np.zeros((num_cranes, num_positions, num_positions), dtype=np.int8)
        truck_parking_mask = np.zeros((10, len(self.parking_spots)), dtype=np.int8)
        terminal_truck_mask = np.zeros((num_terminal_trucks, num_positions, num_positions), dtype=np.int8)
        
        # Generate truck parking mask (simplest)
        available_spots = np.array([i for i, spot in enumerate(self.parking_spots) 
                                if spot not in self.trucks_in_terminal])
        
        trucks_in_queue = list(self.truck_queue.vehicles.queue)
        for truck_idx, _ in enumerate(trucks_in_queue[:10]):
            truck_parking_mask[truck_idx, available_spots] = 1
        
        # Generate crane movement masks using optimized storage yard operations
        self._generate_crane_action_mask_optimized(crane_action_mask)
        
        # Generate terminal truck masks
        self._generate_terminal_truck_mask_optimized(terminal_truck_mask)
        
        # Update performance stats
        self.performance_stats['action_mask_time'] += time.perf_counter() - start_time
        
        return {
            'crane_movement': crane_action_mask,
            'truck_parking': truck_parking_mask,
            'terminal_truck': terminal_truck_mask
        }
    
    def _get_compact_yard_features(self):
        """Get compact yard representation with additional learnable features."""
        # Basic occupancy by row (much smaller than full 3D tensor)
        row_occupancy = np.zeros(self.terminal.num_storage_rows, dtype=np.float32)
        row_heights = np.zeros(self.terminal.num_storage_rows, dtype=np.float32)
        
        # Container type distribution
        container_type_counts = {
            'regular': 0, 'reefer': 0, 'dangerous': 0, 
            'trailer': 0, 'swap_body': 0
        }
        
        # Priority and deadline features
        total_containers = 0
        high_priority_count = 0
        overdue_count = 0
        
        # Process all containers efficiently
        for position, tiers in self.storage_yard.container_registry.items():
            row_idx = self.terminal.storage_row_names.index(position[0])
            containers_at_pos = len(tiers)
            
            row_occupancy[row_idx] += containers_at_pos
            row_heights[row_idx] = max(row_heights[row_idx], containers_at_pos)
            
            for tier, container in tiers.items():
                total_containers += 1
                
                # Count container types
                if hasattr(container, 'container_type'):
                    if container.container_type == 'Trailer':
                        container_type_counts['trailer'] += 1
                    elif container.container_type == 'Swap Body':
                        container_type_counts['swap_body'] += 1
                
                if hasattr(container, 'goods_type'):
                    if container.goods_type == 'Reefer':
                        container_type_counts['reefer'] += 1
                    elif container.goods_type == 'Dangerous':
                        container_type_counts['dangerous'] += 1
                    else:
                        container_type_counts['regular'] += 1
                
                # Priority analysis
                if hasattr(container, 'priority') and container.priority < 50:
                    high_priority_count += 1
                
                # Deadline analysis
                if hasattr(container, 'departure_date') and container.departure_date:
                    if container.departure_date < self.current_simulation_datetime:
                        overdue_count += 1
        
        # Normalize occupancy
        if self.terminal.num_storage_slots_per_row > 0:
            row_occupancy = row_occupancy / self.terminal.num_storage_slots_per_row
            row_heights = row_heights / self.storage_yard.max_tier_height
        
        # Yard efficiency metrics
        total_capacity = self.terminal.num_storage_rows * self.terminal.num_storage_slots_per_row * self.storage_yard.max_tier_height
        utilization = total_containers / max(1, total_capacity)
        
        # Stack quality metric (how well-organized stacks are)
        stack_quality = self._calculate_stack_quality()
        
        # Compile compact features
        compact_features = np.concatenate([
            row_occupancy,  # 5 features (one per row)
            row_heights,    # 5 features (max height per row)
            [
                utilization,  # Overall yard utilization
                stack_quality,  # How well stacks are organized
                high_priority_count / max(1, total_containers),  # Ratio of high priority
                overdue_count / max(1, total_containers),  # Ratio of overdue containers
                container_type_counts['regular'] / max(1, total_containers),
                container_type_counts['reefer'] / max(1, total_containers),
                container_type_counts['dangerous'] / max(1, total_containers),
                container_type_counts['trailer'] / max(1, total_containers),
                container_type_counts['swap_body'] / max(1, total_containers),
                # Additional features for learning
                len(self.trucks_in_terminal) / len(self.parking_spots),  # Parking utilization
                len(self.trains_in_terminal) / len(self.terminal.track_names),  # Rail utilization
                self.current_simulation_time / self.max_simulation_time,  # Progress through simulation
            ]
        ])
        
        return compact_features.astype(np.float32)

    def _calculate_stack_quality(self):
        """Calculate how well-organized the stacks are (0=bad, 1=perfect)."""
        total_stacks = 0
        well_organized_stacks = 0
        
        for position, tiers in self.storage_yard.container_registry.items():
            if len(tiers) > 1:  # Only consider actual stacks
                total_stacks += 1
                
                # Check if priorities are ordered correctly (higher priority on top)
                sorted_tiers = sorted(tiers.keys())
                priorities = []
                for tier in sorted_tiers:
                    container = tiers[tier]
                    if hasattr(container, 'priority'):
                        priorities.append(container.priority)
                
                # Check if priorities decrease as we go up (lower number = higher priority)
                if len(priorities) > 1:
                    is_well_organized = all(priorities[i] >= priorities[i+1] 
                                        for i in range(len(priorities)-1))
                    if is_well_organized:
                        well_organized_stacks += 1
        
        return well_organized_stacks / max(1, total_stacks)

    def _precompute_valid_destinations(self):
        """Pre-compute valid destination positions for different container types."""
        self.valid_destinations_cache = {}
        
        # Define container type combinations
        container_combinations = [
            ('TWEU', 'Regular'), ('TWEU', 'Reefer'), ('TWEU', 'Dangerous'),
            ('FEU', 'Regular'), ('FEU', 'Reefer'), ('FEU', 'Dangerous'),
            ('THEU', 'Regular'), ('THEU', 'Reefer'), ('THEU', 'Dangerous'),
            ('Trailer', 'Regular'), ('Swap Body', 'Regular')
        ]
        
        for container_type, goods_type in container_combinations:
            valid_positions = []
            
            # Check all storage positions
            for row in self.terminal.storage_row_names:
                for bay in range(1, self.terminal.num_storage_slots_per_row + 1):
                    position = f"{row}{bay}"
                    
                    # Quick validation based on type
                    if self._is_position_valid_for_container_type(position, container_type, goods_type):
                        valid_positions.append(position)
            
            # Add truck and train positions for regular containers
            if container_type not in ['Trailer', 'Swap Body']:
                valid_positions.extend(self.parking_spots)
                # Add rail positions
                for track in self.terminal.track_names:
                    for slot in range(1, self.terminal.num_railslots_per_track + 1):
                        valid_positions.append(f"{track.lower()}_{slot}")
            
            self.valid_destinations_cache[(container_type, goods_type)] = valid_positions

    def _is_position_valid_for_container_type(self, position, container_type, goods_type):
        """Fast validation check for container type at position."""
        # Trailer/Swap Body checks
        if container_type == 'Trailer':
            return self.storage_yard.is_position_in_special_area(position, 'trailer')
        elif container_type == 'Swap Body':
            return self.storage_yard.is_position_in_special_area(position, 'swap_body')
        
        # Goods type checks
        if goods_type == 'Reefer':
            return self.storage_yard.is_position_in_special_area(position, 'reefer')
        elif goods_type == 'Dangerous':
            return self.storage_yard.is_position_in_special_area(position, 'dangerous')
        
        # Regular containers - avoid special areas
        special_areas = ['reefer', 'dangerous', 'trailer', 'swap_body']
        return not any(self.storage_yard.is_position_in_special_area(position, area) for area in special_areas)

    def _get_valid_destinations_fast(self, container):
        """Get valid destinations for a container using pre-computed cache."""
        container_type = getattr(container, 'container_type', 'TWEU')
        goods_type = getattr(container, 'goods_type', 'Regular')
        
        cache_key = (container_type, goods_type)
        return self.valid_destinations_cache.get(cache_key, [])
    
    def _generate_crane_action_mask_optimized(self, crane_action_mask):
        """Generate crane action masks using pre-computed valid destinations."""
        for i, crane in enumerate(self.cranes):
            # Skip if crane is not available yet
            if self.current_simulation_time < self.crane_available_times[i].item():
                continue
            
            # Get all containers in crane's operational area
            crane_containers = []
            for row_idx, row in enumerate(self.terminal.storage_row_names):
                for bay in range(crane.start_bay + 1, crane.end_bay + 2):
                    if bay <= self.terminal.num_storage_slots_per_row:
                        position = f"{row}{bay}"
                        container, _ = self.storage_yard.get_top_container(position)
                        if container is not None:
                            crane_containers.append((position, container))
            
            # Add truck and train containers in crane area
            for spot, truck in self.trucks_in_terminal.items():
                if truck.has_containers():
                    spot_idx = int(spot.split('_')[1]) - 1
                    if crane.start_bay <= spot_idx <= crane.end_bay:
                        crane_containers.append((spot, truck.containers[0]))
            
            # Process each container using fast destination lookup
            for source_pos, container in crane_containers:
                if source_pos in self.position_to_idx:
                    source_idx = self.position_to_idx[source_pos]
                    
                    # Get valid destinations from cache
                    valid_destinations = self._get_valid_destinations_fast(container)
                    
                    # Filter by proximity and crane area
                    for dest_pos in valid_destinations:
                        if (dest_pos in self.position_to_idx and 
                            self._is_within_crane_reach(source_pos, dest_pos, crane)):
                            
                            # Quick height check instead of full can_accept_container
                            if self._quick_height_check(dest_pos):
                                dest_idx = self.position_to_idx[dest_pos]
                                crane_action_mask[i, source_idx, dest_idx] = 1

    def _is_within_crane_reach(self, source_pos, dest_pos, crane):
        """Quick check if destination is within crane's operational reach."""
        if self._is_storage_position(dest_pos):
            dest_bay = int(re.findall(r'\d+', dest_pos)[0]) - 1
            return crane.start_bay <= dest_bay <= crane.end_bay
        return True  # Trucks/trains are always reachable if in area

    def _quick_height_check(self, position):
        """Quick height check without full validation."""
        if self._is_storage_position(position):
            return self.storage_yard.get_stack_height(position) < self.storage_yard.max_tier_height
        return True
    
    def _validate_crane_move(self, source_pos, dest_pos, crane):
        """Validate a crane move with additional business rules."""
        # No rail slot to rail slot movements
        if self._is_rail_position(source_pos) and self._is_rail_position(dest_pos):
            return False
        
        # For storage-to-storage moves, check distance constraint
        if self._is_storage_position(source_pos) and self._is_storage_position(dest_pos):
            source_bay = int(re.findall(r'\d+', source_pos)[0]) - 1
            dest_bay = int(re.findall(r'\d+', dest_pos)[0]) - 1
            
            # Check pre-marshalling distance constraint
            if abs(source_bay - dest_bay) > 5:
                return False
        
        return True
    
    def _generate_terminal_truck_mask_optimized(self, terminal_truck_mask):
        """Generate terminal truck action masks using GPU operations."""
        for truck_idx, truck in enumerate(self.terminal_trucks):
            # Skip if truck is not available yet
            if self.current_simulation_time < self.terminal_truck_available_times[truck_idx].item():
                continue
            
            # Find all trailers and swap bodies using GPU-accelerated search
            special_containers = self.storage_yard.get_containers_by_type("Trailer") + \
                               self.storage_yard.get_containers_by_type("Swap Body")
            
            for position, tier, container in special_containers:
                if position in self.position_to_idx:
                    source_idx = self.position_to_idx[position]
                    
                    # Find valid destinations using proximity calculation
                    dest_positions = self.storage_yard.calc_possible_moves(position, n=10)
                    
                    for dest_pos in dest_positions:
                        if dest_pos in self.position_to_idx:
                            dest_idx = self.position_to_idx[dest_pos]
                            
                            # Ensure destination is appropriate for container type
                            if container.container_type == "Trailer" and \
                               self._is_in_special_area(dest_pos, 'trailer'):
                                terminal_truck_mask[truck_idx, source_idx, dest_idx] = 1
                            elif container.container_type == "Swap Body" and \
                                 self._is_in_special_area(dest_pos, 'swap_body'):
                                terminal_truck_mask[truck_idx, source_idx, dest_idx] = 1

    def _get_observation(self):
        """Get the current observation with aggressive action mask caching."""
        # Check if we need to regenerate action masks
        if self._should_regenerate_action_mask():
            action_mask = self._generate_action_masks_optimized()
            # Cache the result
            self._action_mask_cache = action_mask
            self._last_action_mask_time = self.current_simulation_time
            self._last_truck_count = len(self.trucks_in_terminal)
            self._last_train_count = len(self.trains_in_terminal)
            self._last_storage_container_count = self.storage_yard.get_container_count()
        else:
            action_mask = self._action_mask_cache
        
        # Create observation dictionary
        observation = {
            'crane_positions': np.array([crane.current_position for crane in self.cranes], dtype=np.int32),
            'crane_available_times': self.crane_available_times.cpu().numpy().astype(np.float32),
            'terminal_truck_available_times': self.terminal_truck_available_times.cpu().numpy().astype(np.float32),
            'current_time': np.array([self.current_simulation_time], dtype=np.float32),
            'yard_state': self._get_compact_yard_features(),  # New compact representation
            'parking_status': self._generate_parking_status(),
            'rail_status': self._generate_rail_status(),
            'queue_sizes': np.array([self.truck_queue.size(), self.train_queue.size()], dtype=np.int32),
            'action_mask': action_mask
        }
        
        return observation

    def _should_regenerate_action_mask(self):
        """Check if action masks need to be regenerated based on state changes."""
        if self._action_mask_cache is None:
            return True
        
        # Time-based cache expiry
        time_expired = (self.current_simulation_time - self._last_action_mask_time) > self._action_mask_cache_duration
        
        # State-based cache invalidation
        truck_count_changed = len(self.trucks_in_terminal) != self._last_truck_count
        train_count_changed = len(self.trains_in_terminal) != self._last_train_count
        storage_changed = self.storage_yard.get_container_count() != self._last_storage_container_count
        
        return time_expired or truck_count_changed or train_count_changed or storage_changed
    
    def _get_yard_state_optimized(self):
        """Get yard state using GPU-accelerated tensor operations."""
        # Get state representation from bitmap storage yard (returns GPU tensor)
        yard_tensor = self.storage_yard.get_state_representation()
        
        # Convert to numpy for observation space compatibility
        yard_state = yard_tensor.cpu().numpy().astype(np.int32)
        
        return yard_state
    
    def step(self, action):
        """Take a step in the environment by executing the specified action."""
        # Get current observation with action mask
        current_obs = self._get_observation()
        
        # Determine action type
        action_type = action['action_type']
        
        if action_type == 0:  # Crane movement
            crane_idx, source_idx, destination_idx = action['crane_movement']
            
            # Check if the crane movement action is valid according to the mask
            if current_obs['action_mask']['crane_movement'][crane_idx, source_idx, destination_idx] == 0:
                # Try to find a valid action instead
                valid_actions = np.argwhere(current_obs['action_mask']['crane_movement'] == 1)
                if len(valid_actions) > 0:
                    valid_idx = np.random.randint(0, len(valid_actions))
                    crane_idx, source_idx, destination_idx = valid_actions[valid_idx]
                else:
                    # No valid crane actions - wait until next time
                    observation = current_obs
                    reward = 0
                    terminated = self.current_simulation_time >= self.max_simulation_time
                    truncated = False
                    info = {"action": "wait", "reason": "No valid crane actions available"}
                    return observation, reward, terminated, truncated, info
            
            # Execute crane movement action
            return self._execute_crane_movement(crane_idx, source_idx, destination_idx)
        
        elif action_type == 1:  # Truck parking
            truck_idx, parking_spot_idx = action['truck_parking']
            
            # Check if the truck parking action is valid
            if current_obs['action_mask']['truck_parking'][truck_idx, parking_spot_idx] == 0:
                # Try to find a valid action instead
                valid_actions = np.argwhere(current_obs['action_mask']['truck_parking'] == 1)
                if len(valid_actions) > 0:
                    valid_idx = np.random.randint(0, len(valid_actions))
                    truck_idx, parking_spot_idx = valid_actions[valid_idx]
                else:
                    # No valid truck parking actions - wait until next time
                    observation = current_obs
                    reward = 0
                    terminated = self.current_simulation_time >= self.max_simulation_time
                    truncated = False
                    info = {"action": "wait", "reason": "No valid truck parking actions available"}
                    return observation, reward, terminated, truncated, info
            
            # Execute truck parking action
            return self._execute_truck_parking(truck_idx, parking_spot_idx)
            
        elif action_type == 2:  # Terminal truck action
            truck_idx, source_idx, destination_idx = action['terminal_truck']
            
            # Check if the terminal truck action is valid
            if current_obs['action_mask']['terminal_truck'][truck_idx, source_idx, destination_idx] == 0:
                # Try to find a valid action instead
                valid_actions = np.argwhere(current_obs['action_mask']['terminal_truck'] == 1)
                if len(valid_actions) > 0:
                    valid_idx = np.random.randint(0, len(valid_actions))
                    truck_idx, source_idx, destination_idx = valid_actions[valid_idx]
                else:
                    # No valid terminal truck actions - wait until next time
                    observation = current_obs
                    reward = 0
                    terminated = self.current_simulation_time >= self.max_simulation_time
                    truncated = False
                    info = {"action": "wait", "reason": "No valid terminal truck actions available"}
                    return observation, reward, terminated, truncated, info
            
            # Execute terminal truck action
            return self._execute_terminal_truck_movement(truck_idx, source_idx, destination_idx)

    def _execute_crane_movement(self, crane_idx, source_idx, destination_idx):
        """Execute a crane movement action with GPU-accelerated operations."""
        source_position = self.idx_to_position[source_idx]
        destination_position = self.idx_to_position[destination_idx]
        
        # Check if the selected crane is available
        if self.current_simulation_time < self.crane_available_times[crane_idx].item():
            # Crane is not available yet - skip to when it becomes available
            time_advanced = self.crane_available_times[crane_idx].item() - self.current_simulation_time
            self.current_simulation_time = self.crane_available_times[crane_idx].item()
            
            # Process time advancement
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
        
        # Update performance stats
        self.performance_stats['storage_operations'] += 1
        
        # Calculate the reward
        reward = self._calculate_reward(container, source_position, destination_position, time_taken)
        
        # Update crane availability time (convert to tensor)
        self.crane_available_times[crane_idx] = self.current_simulation_time + time_taken
        
        # Check if any crane is still available at the current time
        if not torch.any(self.crane_available_times <= self.current_simulation_time):
            # All cranes busy, advance to earliest available
            next_available_time = torch.min(self.crane_available_times).item()
            time_advanced = next_available_time - self.current_simulation_time
            self.current_simulation_time = next_available_time
            
            # Process time advancement
            self._process_time_advancement(time_advanced)
        
        # Get the next observation
        observation = self._get_observation()
        
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
            "current_time": self.current_simulation_time,
            "performance_stats": self.performance_stats.copy()
        }
        
        return observation, reward, terminated, truncated, info

    def _execute_truck_parking(self, truck_idx, parking_spot_idx):
        """Execute a truck parking action."""
        # Get the truck from the queue
        if truck_idx >= self.truck_queue.size():
            # Invalid truck index
            observation = self._get_observation()
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
            observation = self._get_observation()
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
        observation = self._get_observation()
        
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
            "current_time": self.current_simulation_time,
            "performance_stats": self.performance_stats.copy()
        }
        
        return observation, reward, terminated, truncated, info

    def _execute_terminal_truck_movement(self, truck_idx, source_idx, destination_idx):
        """Execute a terminal truck movement action with GPU-accelerated validation."""
        source_position = self.idx_to_position[source_idx]
        destination_position = self.idx_to_position[destination_idx]
        
        # Check if the selected terminal truck is available
        if self.current_simulation_time < self.terminal_truck_available_times[truck_idx].item():
            # Truck is not available yet - skip to when it becomes available
            time_advanced = self.terminal_truck_available_times[truck_idx].item() - self.current_simulation_time
            self.current_simulation_time = self.terminal_truck_available_times[truck_idx].item()
            
            # Process time advancement
            self._process_time_advancement(time_advanced)
            
            # Return observation with no reward (waiting is neutral)
            observation = self._get_observation()
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
            observation = self._get_observation()
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
        
        # Remove container from source position (GPU-accelerated)
        if self._is_storage_position(source_position):
            removed_container = self.storage_yard.remove_container(source_position)
            self.performance_stats['storage_operations'] += 1
        else:
            removed_container = None  # Not implemented for other source types
        
        # Place container at destination (only storage positions supported)
        success = False
        if removed_container is not None and self._is_storage_position(destination_position):
            success = self.storage_yard.add_container(destination_position, removed_container)
            self.performance_stats['storage_operations'] += 1
        
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
                self.performance_stats['storage_operations'] += 1
        
        # Update terminal truck availability time (convert to tensor)
        self.terminal_truck_available_times[truck_idx] = self.current_simulation_time + time_taken
        
        # Check if any trucks are still available
        if not torch.any(self.terminal_truck_available_times <= self.current_simulation_time):
            # All terminal trucks busy, advance to earliest available
            next_available_time = torch.min(self.terminal_truck_available_times).item()
            time_advanced = next_available_time - self.current_simulation_time
            self.current_simulation_time = next_available_time
            
            # Process time advancement
            self._process_time_advancement(time_advanced)
        
        # Get the next observation
        observation = self._get_observation()
        
        # Check if the episode is over
        terminated = self.current_simulation_time >= self.max_simulation_time
        truncated = False
        
        # Additional info
        info = {
            "action_type": "terminal_truck",
            "time_taken": time_taken,
            "container_moved": removed_container.container_id if removed_container else None,
            "success": success,
            "current_time": self.current_simulation_time,
            "performance_stats": self.performance_stats.copy()
        }
        
        return observation, reward, terminated, truncated, info

    def _generate_crane_action_mask_optimized(self, crane_action_mask):
        """Generate crane action masks using GPU-accelerated proximity calculations with proper N values."""
        for i, crane in enumerate(self.cranes):
            # Skip if crane is not available yet
            if self.current_simulation_time < self.crane_available_times[i].item():
                continue
            
            # Get all storage positions with containers in this crane's operational area
            crane_positions = []
            for row_idx, row in enumerate(self.terminal.storage_row_names):
                for bay in range(crane.start_bay + 1, crane.end_bay + 2):
                    if bay <= self.terminal.num_storage_slots_per_row:
                        position = f"{row}{bay}"
                        container, _ = self.storage_yard.get_top_container(position)
                        if container is not None:
                            crane_positions.append(position)
            
            # Add truck and train positions within crane area
            for spot, truck in self.trucks_in_terminal.items():
                if truck.has_containers():
                    spot_idx = int(spot.split('_')[1]) - 1
                    if crane.start_bay <= spot_idx <= crane.end_bay:
                        crane_positions.append(spot)
            
            for track, train in self.trains_in_terminal.items():
                for j, wagon in enumerate(train.wagons):
                    if not wagon.is_empty():
                        slot = f"{track.lower()}_{j+1}"
                        slot_idx = j
                        if crane.start_bay <= slot_idx <= crane.end_bay:
                            crane_positions.append(slot)
            
            # Process each position with appropriate N value
            if crane_positions:
                for source_pos in crane_positions:
                    if source_pos in self.position_to_idx:
                        source_idx = self.position_to_idx[source_pos]
                        container, _ = self.storage_yard.get_top_container(source_pos)
                        
                        if container:
                            # Determine N value based on move type
                            if self._is_storage_position(source_pos):
                                # For storage positions, check what type of moves are possible
                                
                                # N=1 for pre-marshalling (storage-to-storage within 1 bay)
                                premarshalling_moves = self.storage_yard.calc_possible_moves(source_pos, n=1)
                                for dest_pos in premarshalling_moves:
                                    if (dest_pos in self.position_to_idx and 
                                        self._is_storage_position(dest_pos) and
                                        self._validate_crane_move(source_pos, dest_pos, crane)):
                                        dest_idx = self.position_to_idx[dest_pos]
                                        crane_action_mask[i, source_idx, dest_idx] = 1
                                
                                # N=5 for moving to trucks/trains (transfer operations)
                                transfer_moves = self.storage_yard.calc_possible_moves(source_pos, n=5)
                                for dest_pos in transfer_moves:
                                    if (dest_pos in self.position_to_idx and 
                                        not self._is_storage_position(dest_pos) and  # Only trucks/trains
                                        self._validate_crane_move(source_pos, dest_pos, crane)):
                                        dest_idx = self.position_to_idx[dest_pos]
                                        crane_action_mask[i, source_idx, dest_idx] = 1
                            
                            else:
                                # For truck/train positions, use N=5 to reach parallel storage positions
                                possible_moves = self.storage_yard.calc_possible_moves(source_pos, n=5)
                                for dest_pos in possible_moves:
                                    if (dest_pos in self.position_to_idx and
                                        self._validate_crane_move(source_pos, dest_pos, crane)):
                                        dest_idx = self.position_to_idx[dest_pos]
                                        crane_action_mask[i, source_idx, dest_idx] = 1



    def evaluate_need_for_premarshalling(self):
        """Determine if pre-marshalling is needed based on yard state."""
        # Use GPU-accelerated operations to check yard state
        return self.storage_yard.get_container_count() > 0  # Simplified check
    
    def set_vehicle_limits(self, max_trucks=None, max_trains=None):
        """Set limits on the number of trucks and trains that can be generated per day."""
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

    def set_simplified_rendering(self, simplified=True):
        """Set simplified rendering mode for faster training."""
        self.simplified_rendering = simplified

    def render(self, mode='human'):
        """Render the terminal environment with optimization for training."""
        if hasattr(self, 'simplified_rendering') and self.simplified_rendering and mode == 'human':
            # During training, don't actually render to save time
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
        
        # Print final performance stats
        print(f"Environment Performance Stats:")
        print(f"  Action mask generation time: {self.performance_stats['action_mask_time']:.4f}s")
        print(f"  Storage operations: {self.performance_stats['storage_operations']}")
        print(f"  Proximity calculations: {self.performance_stats['proximity_calculations']}")
    
    # Helper methods - copying essential ones from original with minimal changes
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
        return bool(position and position[0].isalpha() and position[1:].isdigit())
    
    def _is_truck_position(self, position):
        """Check if a position is a truck parking spot."""
        return position.startswith('p_')
    
    def _is_rail_position(self, position):
        """Check if a position is a rail slot."""
        return position.startswith('t') and '_' in position
    
    def _is_in_special_area(self, position, area_type):
        """Check if a position is in a special area using GPU-accelerated operations."""
        return self.storage_yard.is_position_in_special_area(position, area_type)
    
    def _get_container_at_position(self, position):
        """Helper to get container at a position using GPU-accelerated operations."""
        if self._is_storage_position(position):
            container, _ = self.storage_yard.get_top_container(position)
            return container
        elif self._is_truck_position(position):
            truck = self.trucks_in_terminal.get(position)
            if truck and hasattr(truck, 'containers') and truck.containers:
                return truck.containers[0]
            return None
        elif self._is_rail_position(position):
            # Parse train position and get container
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
    
    def _process_time_advancement(self, time_advanced):
        """Process events that occur during time advancement."""
        # Update current simulation datetime
        self.current_simulation_datetime += timedelta(seconds=time_advanced)
        
        # Process vehicle arrivals based on time advancement
        self._process_vehicle_arrivals(time_advanced)
        
        # Process vehicle departures
        self._process_vehicle_departures()
        
        # Invalidate action mask cache
        self._action_mask_cache = None
    
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
    
    def _generate_parking_status(self):
        """Generate parking status array."""
        parking_status = np.zeros(len(self.parking_spots), dtype=np.int32)
        
        for i, spot in enumerate(self.parking_spots):
            if spot in self.trucks_in_terminal:
                parking_status[i] = 1
                
        return parking_status
    
    def _generate_rail_status(self):
        """Generate rail status array."""
        rail_status = np.zeros((len(self.terminal.track_names), self.terminal.num_railslots_per_track), dtype=np.int32)
        
        for i, track in enumerate(self.terminal.track_names):
            if track in self.trains_in_terminal:
                # Mark the entire track as occupied
                rail_status[i, :] = 1
                
        return rail_status
    
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
        
        return reward
    
    def _calculate_reward(self, container, source_position, destination_position, time_taken):
        """Calculate the reward for moving a container."""
        reward = 0.0  # Base reward
        
        # Get source and destination information
        source_type = self._get_position_type_direct(source_position)
        dest_type = self._get_position_type_direct(destination_position)
        
        # Calculate distance for this move
        distance = 0.0
        if hasattr(self.terminal, 'get_distance'):
            try:
                distance = self.terminal.get_distance(source_position, destination_position)
            except:
                # If distance calculation fails, estimate based on time
                distance = time_taken * 0.5  # Rough estimate
        
        # Empty crane movement penalty
        if container is None:
            empty_move_penalty = -5.0
            distance_time_penalty = -0.05 * distance - time_taken / 60.0
            return empty_move_penalty + distance_time_penalty
        
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
        
        # DISTANCE AND TIME PENALTY
        distance_penalty = -0.02 * distance  # -0.02 per meter
        time_penalty = -min(time_taken / 120, 1.0)  # Cap at -1 for moves over 2 minutes
        distance_time_penalty = distance_penalty + time_penalty
        reward += distance_time_penalty
        
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
        
        return reward
    
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


# Alias for backward compatibility
TerminalEnvironment = OptimizedTerminalEnvironment