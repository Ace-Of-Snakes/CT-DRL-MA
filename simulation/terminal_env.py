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

# Import our custom components
from simulation.terminal_layout.CTSimulator import ContainerTerminal
from simulation.terminal_components.Container import Container, ContainerFactory
from simulation.terminal_components.Train import Train
from simulation.terminal_components.Truck import Truck
from simulation.terminal_components.Storage_Yard import StorageYard
from simulation.terminal_components.RMGCrane import RMGCrane
from simulation.terminal_components.Vehicle_Queue import VehicleQueue


class TerminalEnvironment(gym.Env):
    """
    Container Terminal Environment for RL agents.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, 
                 terminal_config_path: str = None,
                 terminal_config=None,
                 distance_matrix_path: str = None,
                 max_simulation_time: float = 86400,  # 24 hours in seconds
                 num_cranes: int = 2):
        """Initialize the terminal environment."""
        super(TerminalEnvironment, self).__init__()

        # Initialize container ID storage
        self.stored_container_ids = []

        # Load or create terminal configuration
        self.config = terminal_config or self._load_config(terminal_config_path)

        # Create terminal
        self.terminal = self._create_terminal()
        
        # Load distance matrix if available
        if distance_matrix_path and os.path.exists(distance_matrix_path):
            self.terminal.load_distance_matrix(distance_matrix_path)
        
        # Initialize environment components
        self.storage_yard = self._create_storage_yard()
        self.cranes = self._create_cranes(num_cranes)
        self.truck_queue = VehicleQueue(vehicle_type="Truck")
        self.train_queue = VehicleQueue(vehicle_type="Train")
        
        # Track current state
        self.current_simulation_time = 0.0
        self.max_simulation_time = max_simulation_time
        self.crane_available_times = [0.0] * num_cranes
        
        self.trucks_in_terminal = {}
        self.trains_in_terminal = {}
        
        # Create position mappings
        self._setup_position_mappings()
        
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

        # Initialize the environment state
        self.reset()
    
    def _load_config(self, config_path):
        """Load terminal configuration."""
        from config import TerminalConfig
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
    
    def _create_storage_yard(self):
        """Create the storage yard."""
        # Define special areas for different container types
        special_areas = {
            'reefer': [],
            'dangerous': []
        }
        
        # Add reefer areas in first and last column of each row
        for row in self.terminal.storage_row_names:
            special_areas['reefer'].append((row, 1, 1))
            special_areas['reefer'].append((row, 58, 58))
            
        # Add dangerous goods area in middle columns
        for row in self.terminal.storage_row_names:
            special_areas['dangerous'].append((row, 33, 35))
        
        return StorageYard(
            num_rows=self.terminal.num_storage_rows,
            num_bays=self.terminal.num_storage_slots_per_row,
            max_tier_height=5,
            row_names=self.terminal.storage_row_names,
            special_areas=special_areas
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
        all_positions.extend([f"{row}{i+1}" for row in self.terminal.storage_row_names 
                             for i in range(self.terminal.num_storage_slots_per_row)])
        
        self.position_to_idx = {pos: i for i, pos in enumerate(all_positions)}
        self.idx_to_position = {i: pos for i, pos in enumerate(all_positions)}
    
    def _setup_spaces(self):
        """Set up action and observation spaces."""
        num_positions = len(self.position_to_idx)
        num_cranes = len(self.cranes)
        
        # Action space: crane_id, source, destination or truck_idx, parking_spot
        self.action_space = spaces.Dict({
            'action_type': spaces.Discrete(2),  # 0: crane movement, 1: truck parking
            'crane_movement': spaces.MultiDiscrete([
                num_cranes,        # Crane index
                num_positions,     # Source position index
                num_positions      # Destination position index
            ]),
            'truck_parking': spaces.MultiDiscrete([
                10,                # Max trucks in queue to consider
                len(self.parking_spots)  # Parking spot index
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
            'current_time': spaces.Box(
                low=0,
                high=np.inf,
                shape=(1,),
                dtype=np.float32
            ),
            'yard_state': spaces.Box(
                low=0,
                high=1,
                shape=(self.terminal.num_storage_rows, self.terminal.num_storage_slots_per_row, 5),
                dtype=np.int32
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
        
        # Schedule trains and trucks
        self._schedule_trains()
        self._schedule_trucks_for_existing_containers()
        
        # Get initial observation
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
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
                    print(f"Warning: Invalid crane action replaced with valid action: "
                         f"Crane {crane_idx+1}: {self.idx_to_position[source_idx]} → {self.idx_to_position[destination_idx]}")
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
                    print(f"Warning: Invalid truck parking action replaced with valid action: "
                         f"Truck {truck_idx} → Parking spot {self.parking_spots[parking_spot_idx]}")
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
                if row != self.terminal.storage_row_names[0]:
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
        
        # Calculate the reward
        reward = self._calculate_reward(container, source_position, destination_position, time_taken)
        
        # Update crane availability time
        self.crane_available_times[crane_idx] = self.current_simulation_time + time_taken
        
        # Check if any crane is still available at the current time
        if any(t <= self.current_simulation_time for t in self.crane_available_times):
            # Some cranes still available, don't advance time
            time_advanced = 0.0
        else:
            # All cranes busy, advance to earliest available
            next_available_time = min(self.crane_available_times)
            time_advanced = next_available_time - self.current_simulation_time
            self.current_simulation_time = next_available_time
        
        # Process time advancement if needed
        if time_advanced > 0:
            self._process_time_advancement(time_advanced)
        
        # Get the next observation
        observation = self._get_observation()
        
        # Check if the episode is over
        terminated = self.current_simulation_time >= self.max_simulation_time
        truncated = False
        
        # Track last action for rendering
        self.last_action = f"Crane {crane_idx+1}: {source_position} → {destination_position}"
        if container:
            self.last_action += f" - Container: {container.container_id}"
        
        # Additional info
        info = {
            "action_type": "crane_movement",
            "time_taken": time_taken,
            "container_moved": container.container_id if container else None,
            "crane_position": crane.current_position,
            "trucks_waiting": self.truck_queue.size(),
            "trains_waiting": self.train_queue.size(),
            "current_time": self.current_simulation_time,
            "valid_action": True
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
        
        # Track last action for rendering
        self.last_action = f"Truck {truck.truck_id} → Parking {parking_spot}"
        
        # Additional info
        info = {
            "action_type": "truck_parking",
            "time_taken": time_advanced,
            "truck_id": truck.truck_id,
            "parking_spot": parking_spot,
            "trucks_waiting": self.truck_queue.size(),
            "trains_waiting": self.train_queue.size(),
            "current_time": self.current_simulation_time,
            "valid_action": True
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
    
    def _calculate_reward(self, container, source_position, destination_position, time_taken):
        """Calculate the reward for moving a container."""
        reward = 2.0  # Base reward
        
        # Get source and destination information
        source_type = self._get_position_type(source_position)
        dest_type = self._get_position_type(destination_position)
        
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
            empty_move_penalty = -1.0
            distance_time_penalty = -0.05 * distance - time_taken / 60.0
            
            print(f"Empty move: base={empty_move_penalty}, distance={distance:.2f}m, time={time_taken:.2f}s")
            print(f"Distance-time penalty: {distance_time_penalty:.2f}")
            
            return empty_move_penalty + distance_time_penalty
        
        # Determine the reward based on move type
        if source_type == 'train' and dest_type == 'truck':
            # GOLDEN MOVE: DIRECT TRAIN TO TRUCK
            move_type_reward = 30.0
            print(f"Golden move (train→truck): +{move_type_reward:.2f}")
        elif source_type == 'truck' and dest_type == 'train':
            # GOLDEN MOVE: DIRECT TRUCK TO TRAIN
            move_type_reward = 30.0
            print(f"Golden move (truck→train): +{move_type_reward:.2f}")
        elif source_type == 'storage' and (dest_type == 'truck' or dest_type == 'train'):
            # GOOD MOVE: STORAGE TO TRUCK OR TRAIN
            move_type_reward = 15.0
            print(f"Good move (storage→{dest_type}): +{move_type_reward:.2f}")
            
            # DEADLINE BONUS: Container moved before deadline
            if hasattr(container, 'departure_date') and container.departure_date:
                time_until_deadline = (container.departure_date - self.current_simulation_datetime).total_seconds()
                if time_until_deadline > 0:
                    # Scale reward based on time till deadline
                    time_factor = min(1.0, 24*3600 / max(3600, time_until_deadline))
                    deadline_bonus = 5.0 * time_factor
                    move_type_reward += deadline_bonus
                    print(f"Deadline bonus: +{deadline_bonus:.2f} (due in {time_until_deadline/3600:.1f}h)")
        elif (source_type == 'train' or source_type == 'truck') and dest_type == 'storage':
            # STANDARD MOVES: TRAIN/TRUCK TO STORAGE
            move_type_reward = 5.0
            print(f"Standard move ({source_type}→storage): +{move_type_reward:.2f}")
        elif source_type == 'storage' and dest_type == 'storage':
            # RESHUFFLING: STORAGE TO STORAGE
            move_type_reward = -0.5
            print(f"Reshuffling penalty: {move_type_reward:.2f}")
        elif source_type == 'truck' and dest_type == 'parking':
            # TRUCK PARKING: ASSIGN TRUCK TO SPOT
            move_type_reward = 6.0
            print(f"Truck parking assignment: +{move_type_reward:.2f}")
        
        # Add the move type reward
        reward += move_type_reward
        
        # DEADLINE PENALTY: Container moved after deadline
        if hasattr(container, 'departure_date') and container.departure_date:
            time_past_deadline = (self.current_simulation_datetime - container.departure_date).total_seconds()
            if time_past_deadline > 0:
                past_deadline_hours = time_past_deadline / 3600
                deadline_penalty = -min(10.0, past_deadline_hours * 0.5)  # Cap at -10
                reward += deadline_penalty
                print(f"Past deadline penalty: {deadline_penalty:.2f} ({past_deadline_hours:.1f}h late)")
        
        # PRIORITY BONUS: Based on container priority
        if hasattr(container, 'priority'):
            priority_factor = max(0, (100 - container.priority) / 100)
            priority_bonus = priority_factor * 2.0
            reward += priority_bonus
            print(f"Priority bonus: +{priority_bonus:.2f} (priority {container.priority})")
        
        # DISTANCE AND TIME PENALTY
        distance_penalty = -0.02 * distance  # -0.02 per meter
        time_penalty = -min(time_taken / 120, 1.0)  # Cap at -1 for moves over 2 minutes
        distance_time_penalty = distance_penalty + time_penalty
        reward += distance_time_penalty
        
        print(f"Distance-time penalty: {distance_time_penalty:.2f} (dist={distance:.1f}m, time={time_taken:.1f}s)")
        
        # Final reward
        final_reward = round(reward, 2)
        print(f"Total reward: {final_reward:.2f}")
        
        return final_reward
    # Add these to the TerminalEnvironment class

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
    def _process_time_advancement(self, time_advanced):
        """Process events that occur during time advancement."""
        # Update current simulation datetime
        self.current_simulation_datetime += timedelta(seconds=time_advanced)
        
        # Process vehicle arrivals based on time advancement
        self._process_vehicle_arrivals(time_advanced)
        
        # Process vehicle departures
        self._process_vehicle_departures()
    
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
    
    def _get_observation(self):
        """Get the current observation with proper action masking."""
        # Create basic observation dictionary
        observation = {
            'crane_positions': np.array([crane.current_position for crane in self.cranes], dtype=np.int32),
            'crane_available_times': np.array(self.crane_available_times, dtype=np.float32),
            'current_time': np.array([self.current_simulation_time], dtype=np.float32),
            'yard_state': self.storage_yard.get_state_representation(),
            'parking_status': np.zeros(len(self.parking_spots), dtype=np.int32),
            'rail_status': np.zeros((len(self.terminal.track_names), self.terminal.num_railslots_per_track), dtype=np.int32),
            'queue_sizes': np.array([self.truck_queue.size(), self.train_queue.size()], dtype=np.int32)
        }
        
        # Generate crane movement action mask
        crane_action_mask = np.zeros((
            len(self.cranes), 
            len(self.position_to_idx),
            len(self.position_to_idx)
        ), dtype=np.int8)
        
        # For each crane, calculate valid moves
        for i, crane in enumerate(self.cranes):
            # Skip if crane is not available yet
            if self.current_simulation_time < self.crane_available_times[i]:
                continue
                
            # Get all possible source positions where a container can be picked up
            source_positions = crane._get_source_positions(self.storage_yard, self.trucks_in_terminal, self.trains_in_terminal)
            
            # For each source, find all valid destinations
            for source_position in source_positions:
                # Get the container at this position
                container = self._get_container_at_position(source_position)
                
                if container is None:
                    continue
                    
                # Get valid destinations based on rules
                destinations = crane._get_destination_positions(source_position, container, self.storage_yard, 
                                                           self.trucks_in_terminal, self.trains_in_terminal)
                
                # Update action mask for each valid source->destination pair
                for dest_position in destinations:
                    source_idx = self.position_to_idx[source_position]
                    dest_idx = self.position_to_idx[dest_position]
                    
                    # Apply operational constraints
                    if self._is_rail_position(source_position) and self._is_rail_position(dest_position):
                        # No rail slot to rail slot movements
                        continue

                    if self._is_storage_position(source_position) and self._is_storage_position(dest_position):
                        # Extract bay numbers
                        source_bay = int(re.findall(r'\d+', source_position)[0]) - 1
                        dest_bay = int(re.findall(r'\d+', dest_position)[0]) - 1
                        
                        # Check pre-marshalling distance constraint
                        if abs(source_bay - dest_bay) > 5:
                            continue
                        
                        # Check for stacking compatibility
                        existing_container, _ = self.storage_yard.get_top_container(dest_position)
                        
                        # If there's a container at the destination and we're trying to stack on top of it
                        if existing_container is not None and container is not None:
                            # Check if stacking is safe
                            if not container.can_be_stacked_on(existing_container):
                                continue
                    
                    # Swap body/trailer placement restrictions in storage
                    if (container and container.container_type in ["Trailer", "Swap Body"] 
                        and self._is_storage_position(dest_position)):
                        # Must be placed in the row nearest to driving lane
                        dest_row = dest_position[0]
                        nearest_row = self.terminal.storage_row_names[0]
                        if dest_row != nearest_row:
                            continue
                    
                    # Update the action mask
                    crane_action_mask[i, source_idx, dest_idx] = 1
        
        # Generate truck parking action mask
        truck_parking_mask = np.zeros((10, len(self.parking_spots)), dtype=np.int8)
        
        # Check which parking spots are available
        available_spots = [i for i, spot in enumerate(self.parking_spots) 
                          if spot not in self.trucks_in_terminal]
        
        # Get trucks from the queue
        trucks_in_queue = list(self.truck_queue.vehicles.queue)
        
        # For each truck in the queue, determine valid parking spots
        for truck_idx, truck in enumerate(trucks_in_queue):
            if truck_idx >= 10:  # Only consider first 10 trucks
                break
                
            # First check: Can only park in available spots
            for spot_idx in available_spots:
                parking_spot = self.parking_spots[spot_idx]
                
                # Default: all available spots are valid
                truck_parking_mask[truck_idx, spot_idx] = 1
                
                # Apply the parallel parking rule for pickup trucks
                if truck.is_pickup_truck and truck.pickup_container_ids:
                    # Find the wagon that has the container this truck needs to pick up
                    valid_spot = False
                    for track_id, train in self.trains_in_terminal.items():
                        for i, wagon in enumerate(train.wagons):
                            for container in wagon.containers:
                                if container.container_id in truck.pickup_container_ids:
                                    # This wagon has a container the truck needs to pick up
                                    target_position = f"{track_id.lower()}_{i+1}"
                                    
                                    # Get the rail slot index
                                    slot_num = int(target_position.split('_')[1])
                                    
                                    # Get the parallel parking spots and one spot on each side
                                    parallel_spot_names = []
                                    if slot_num > 1:
                                        parallel_spot_names.append(f"p_{slot_num-1}")
                                    parallel_spot_names.append(f"p_{slot_num}")
                                    if slot_num < self.terminal.num_railslots_per_track:
                                        parallel_spot_names.append(f"p_{slot_num+1}")
                                    
                                    # Check if this parking spot is one of the valid ones
                                    if parking_spot in parallel_spot_names:
                                        valid_spot = True
                                        break
                            if valid_spot:
                                break
                        if valid_spot:
                            break
                    
                    # If this is a pickup truck and the parking spot is not valid per our rules,
                    # mask it as invalid
                    if not valid_spot:
                        truck_parking_mask[truck_idx, spot_idx] = 0
                
                # Apply the rule for delivery trucks (bringing containers)
                elif not truck.is_pickup_truck and truck.containers:
                    # For delivery trucks, check if they're bringing containers for specific wagons
                    valid_spot = False
                    for container in truck.containers:
                        if hasattr(container, 'destination_id'):
                            # Find the wagon that needs this container
                            for track_id, train in self.trains_in_terminal.items():
                                for i, wagon in enumerate(train.wagons):
                                    if container.destination_id in wagon.pickup_container_ids:
                                        # This wagon is expecting this container
                                        target_position = f"{track_id.lower()}_{i+1}"
                                        
                                        # Get the rail slot index
                                        slot_num = int(target_position.split('_')[1])
                                        
                                        # Get the parallel parking spots and one spot on each side
                                        parallel_spot_names = []
                                        if slot_num > 1:
                                            parallel_spot_names.append(f"p_{slot_num-1}")
                                        parallel_spot_names.append(f"p_{slot_num}")
                                        if slot_num < self.terminal.num_railslots_per_track:
                                            parallel_spot_names.append(f"p_{slot_num+1}")
                                        
                                        # Check if this parking spot is one of the valid ones
                                        if parking_spot in parallel_spot_names:
                                            valid_spot = True
                                            break
                                if valid_spot:
                                    break
                            if valid_spot:
                                break
                    
                    # If this is a delivery truck and the parking spot is not valid per our rules,
                    # mask it as invalid
                    if not valid_spot:
                        truck_parking_mask[truck_idx, spot_idx] = 0
        
        # Add action masks to observation
        observation['action_mask'] = {
            'crane_movement': crane_action_mask,
            'truck_parking': truck_parking_mask
        }
        
        return observation
    
    def _get_container_at_position(self, position):
        """Helper to get container at a position."""
        if self._is_storage_position(position):
            # Get top container from storage
            container, _ = self.storage_yard.get_top_container(position)
            return container
        elif self._is_truck_position(position):
            # Get container from truck
            truck = self.trucks_in_terminal.get(position)
            if truck and hasattr(truck, 'containers') and truck.containers:
                return truck.containers[0]  # Return the first container
            return None
        elif self._is_rail_position(position):
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
        """Determine the type of a position (train, truck, storage)."""
        if self._is_rail_position(position):
            return 'train'
        elif self._is_truck_position(position):
            return 'truck'
        else:
            return 'storage'
    
    def _is_storage_position(self, position: str) -> bool:
        """Check if a position is in the storage yard."""
        return position[0].isalpha() and position[1:].isdigit()
    
    def _is_truck_position(self, position: str) -> bool:
        """Check if a position is a truck parking spot."""
        return position.startswith('p_')
    
    def _is_rail_position(self, position: str) -> bool:
        """Check if a position is a rail slot."""
        return position.startswith('t') and '_' in position
    
    def render(self, mode='human'):
        """Render the terminal environment."""
        if mode == 'human':
            # Render the terminal for human viewing
            fig, ax = self.terminal.visualize(figsize=(15, 10), show_labels=True)
            
            # Draw cranes
            for i, crane in enumerate(self.cranes):
                pos = crane.current_position
                crane_color = f'C{i}'  # Different color for each crane
                
                # Convert bay, row to x, y
                x_pos = pos[0] * self.terminal.rail_slot_length + self.terminal.rail_slot_length / 2
                y_pos = 0  # Placeholder
                
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
        plt.close('all')


def test_environment(num_steps=5, render=True):
    """Test the terminal environment with random actions."""
    # Create terminal config
    from config import TerminalConfig
    config = TerminalConfig()
    
    # Create environment with the config
    env = TerminalEnvironment(terminal_config=config)
    obs, info = env.reset()
    
    print(f"Environment initialized with {len(env.cranes)} cranes")
    
    # Print all valid moves
    print("\n=== VALID MOVES BEFORE FIRST ACTION ===\n")
    for i, crane in enumerate(env.cranes):
        valid_moves = crane.get_valid_moves(env.storage_yard, env.trucks_in_terminal, env.trains_in_terminal)
        print(f"Crane {i+1} can make {len(valid_moves)} valid moves:")
        
        # Sort moves by source for better readability
        sorted_moves = sorted(valid_moves.items(), key=lambda x: x[0][0])
        
        # Print first 10 moves as sample
        for j, ((source, dest), time) in enumerate(sorted_moves[:10]):
            print(f"  {source} → {dest} (Est. time: {time:.1f}s)")
        
        # If there are more moves, show a count
        if len(valid_moves) > 10:
            print(f"  ...and {len(valid_moves) - 10} more moves\n")
    
    total_reward = 0
    for i in range(num_steps):
        # Get a random action
        action = {
            'action_type': np.random.randint(0, 2),
            'crane_movement': env.action_space['crane_movement'].sample(),
            'truck_parking': env.action_space['truck_parking'].sample()
        }
        
        # Take a step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Print step information
        print(f"\nStep {i+1}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")
        print(f"Simulation Time: {env.current_simulation_time:.2f}s")
        print(f"Crane Positions: {obs['crane_positions']}")
        print(f"Crane Available Times: {obs['crane_available_times']}")
        
        # Print valid moves after action
        if i < num_steps - 1:  # Skip on last step
            print("\n=== VALID MOVES AFTER ACTION ===\n")
            for j, crane in enumerate(env.cranes):
                valid_moves = crane.get_valid_moves(env.storage_yard, env.trucks_in_terminal, env.trains_in_terminal)
                print(f"Crane {j+1} can make {len(valid_moves)} valid moves:")
                
                # Sort and print a sample of moves
                sorted_moves = sorted(valid_moves.items(), key=lambda x: x[0][0])
                for k, ((source, dest), time) in enumerate(sorted_moves[:5]):
                    print(f"  {source} → {dest} (Est. time: {time:.1f}s)")
                
                if len(valid_moves) > 5:
                    print(f"  ...and {len(valid_moves) - 5} more moves\n")
        
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