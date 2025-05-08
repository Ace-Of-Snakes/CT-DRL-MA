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
import random

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
                 terminal_config=None,
                 distance_matrix_path: str = None,
                 max_simulation_time: float = 86400,  # 24 hours in seconds
                 num_cranes: int = 2):
        """
        Initialize the terminal environment.
        
        Args:
            terminal_config_path: Path to terminal configuration file
            terminal_config: TerminalConfig object (created if None)
            distance_matrix_path: Path to pre-calculated distance matrix
            max_simulation_time: Maximum simulation time in seconds
            num_cranes: Number of RMG cranes in the terminal
        """
        super(TerminalEnvironment, self).__init__()

        # Initialize container ID storage
        self.stored_container_ids = []

        # Load or create terminal configuration
        if terminal_config is None:
            from config import TerminalConfig
            self.config = TerminalConfig(terminal_config_path)
        else:
            self.config = terminal_config

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
        
        self.action_space = spaces.Dict({
            'action_type': spaces.Discrete(2),  # 0: crane movement, 1: truck parking
            'crane_movement': spaces.MultiDiscrete([
                num_cranes,  # Crane index
                len(all_positions),  # Source position index
                len(all_positions)   # Destination position index
            ]),
            'truck_parking': spaces.MultiDiscrete([
                10,  # Max number of trucks in queue to consider (can be adjusted)
                len(self.parking_spots)  # Parking spot index
            ])
        })
        
        # Create an action mask space
        self.action_mask_space = spaces.Dict({
            'crane_movement': spaces.Box(
                low=0,
                high=1,
                shape=(
                    num_cranes,  # Crane index
                    len(all_positions),  # Source position index
                    len(all_positions)   # Destination position index
                ),
                dtype=np.int8
            ),
            'truck_parking': spaces.Box(
                low=0,
                high=1,
                shape=(
                    10,  # Max trucks in queue
                    len(self.parking_spots)  # Parking spot index
                ),
                dtype=np.int8
            )
        })
        
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

        # Base simulation date for scheduling
        self.base_simulation_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Generate train arrival schedule for the simulation period
        self.train_schedule = self.config.generate_train_arrival_schedule(
            n_trains=20,  # Adjust based on expected terminal throughput
            base_date=self.base_simulation_date
        )

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
        # Reset container ID storage
        self.stored_container_ids = []
        # Reset time tracking
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
        
        # Schedule trains based on the KDE model
        self._schedule_trains()

        # Generate truck pickups for containers in storage
        self._schedule_trucks_for_existing_containers()

        # # Add some trucks and trains to the queues
        # self._generate_initial_vehicles()
        
        # Get initial observation
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """
        Take a step in the environment by executing the specified action.
        
        Args:
            action: Dictionary containing action details
                - action_type: 0 for crane movement, 1 for truck parking
                - crane_movement: [crane_idx, source_idx, destination_idx] for crane movements
                - truck_parking: [truck_idx, parking_spot_idx] for truck parking
                
        Returns:
            observation: Next observation
            reward: Reward for the action
            terminated: Whether the episode is over
            truncated: Whether the episode was truncated
            info: Additional information
        """
        # Get current observation with action mask
        current_obs = self._get_observation()
        
        # Determine action type
        action_type = action['action_type']
        
        if action_type == 0:  # Crane movement
            crane_idx, source_idx, destination_idx = action['crane_movement']
            
            # Check if the crane movement action is valid according to the mask
            if current_obs['action_mask']['crane_movement'][crane_idx, source_idx, destination_idx] == 0:
                # Invalid action - choose a random valid crane movement instead
                valid_actions = np.argwhere(current_obs['action_mask']['crane_movement'] == 1)
                if len(valid_actions) > 0:
                    # Choose a random valid action
                    valid_idx = np.random.randint(0, len(valid_actions))
                    crane_idx, source_idx, destination_idx = valid_actions[valid_idx]
                    # Inform about the replacement
                    print(f"Warning: Invalid crane action replaced with valid action: "
                        f"Crane {crane_idx+1}: {self.idx_to_position[source_idx]} → {self.idx_to_position[destination_idx]}")
                else:
                    # No valid crane actions - wait until next time
                    observation = current_obs
                    reward = 0  # Neutral reward for waiting
                    terminated = self.current_simulation_time >= self.max_simulation_time
                    truncated = False
                    info = {"action": "wait", "reason": "No valid crane actions available"}
                    return observation, reward, terminated, truncated, info
            
            # Execute crane movement action
            return self._execute_crane_movement(crane_idx, source_idx, destination_idx)
        
        elif action_type == 1:  # Truck parking
            truck_idx, parking_spot_idx = action['truck_parking']
            
            # Check if the truck parking action is valid according to the mask
            if current_obs['action_mask']['truck_parking'][truck_idx, parking_spot_idx] == 0:
                # Invalid action - choose a random valid truck parking instead
                valid_actions = np.argwhere(current_obs['action_mask']['truck_parking'] == 1)
                if len(valid_actions) > 0:
                    # Choose a random valid action
                    valid_idx = np.random.randint(0, len(valid_actions))
                    truck_idx, parking_spot_idx = valid_actions[valid_idx]
                    # Inform about the replacement
                    print(f"Warning: Invalid truck parking action replaced with valid action: "
                        f"Truck {truck_idx} → Parking spot {self.parking_spots[parking_spot_idx]}")
                else:
                    # No valid truck parking actions - wait until next time
                    observation = current_obs
                    reward = 0  # Neutral reward for waiting
                    terminated = self.current_simulation_time >= self.max_simulation_time
                    truncated = False
                    info = {"action": "wait", "reason": "No valid truck parking actions available"}
                    return observation, reward, terminated, truncated, info
            
            # Execute truck parking action
            return self._execute_truck_parking(truck_idx, parking_spot_idx)

    def _calculate_truck_parking_reward(self, truck, parking_spot):
        """
        Calculate reward for parking a truck.
        
        Rewards optimal placement: higher reward for placing trucks near the wagons 
        that have their containers.
        """
        # Base reward for successful truck parking
        reward = 1.0
        
        # Check if truck is empty (pickup truck)
        if truck.is_pickup_truck and truck.pickup_container_ids:
            # For pickup trucks, check if they're parked near the wagons with their containers
            optimal_placed = False
            
            # Find the wagon that has the container this truck needs to pick up
            target_position = None
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
                track_idx = int(track_id[1:]) - 1
                
                # Get the parallel parking spots and one spot on each side
                parallel_spots = []
                if 0 <= slot_num - 1 < self.terminal.num_railslots_per_track:
                    parallel_spots.append(f"p_{slot_num-1}")
                parallel_spots.append(f"p_{slot_num}")
                if slot_num + 1 <= self.terminal.num_railslots_per_track:
                    parallel_spots.append(f"p_{slot_num+1}")
                
                # Check if truck is parked in an optimal spot
                if parking_spot in parallel_spots:
                    optimal_placed = True
                    # Higher reward for exact parallel spot
                    if parking_spot == f"p_{slot_num}":
                        reward += 3.0
                    else:
                        reward += 2.0
                
                # Penalty if not optimally placed, based on distance
                if not optimal_placed:
                    parking_idx = int(parking_spot.split('_')[1])
                    distance = abs(parking_idx - slot_num)
                    distance_penalty = min(0.5 * distance, 5.0)  # Cap the penalty
                    reward -= distance_penalty
        
        # For delivery trucks (bringing containers)
        elif not truck.is_pickup_truck and truck.containers:
            # Check if truck has containers for specific wagons
            for container in truck.containers:
                if hasattr(container, 'destination_id'):
                    # Container has a specific destination wagon
                    target_position = None
                    
                    # Find the wagon that needs this container
                    for track_id, train in self.trains_in_terminal.items():
                        for i, wagon in enumerate(train.wagons):
                            if container.destination_id in wagon.pickup_container_ids:
                                target_position = f"{track_id.lower()}_{i+1}"
                                break
                        if target_position:
                            break
                    
                    # If we found a target wagon, check if truck is parked optimally
                    if target_position:
                        # Get the rail slot index
                        track_id = target_position.split('_')[0]
                        slot_num = int(target_position.split('_')[1])
                        
                        # Get the parallel parking spots and one spot on each side
                        parallel_spots = []
                        if 0 <= slot_num - 1 < self.terminal.num_railslots_per_track:
                            parallel_spots.append(f"p_{slot_num-1}")
                        parallel_spots.append(f"p_{slot_num}")
                        if slot_num + 1 <= self.terminal.num_railslots_per_track:
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
        
        return reward

    def _initialize_storage_yard(self):
        """Initialize the storage yard with random containers using probability model."""
        # Fill about 30% of the yard with random containers
        num_rows = self.terminal.num_storage_rows
        num_bays = self.terminal.num_storage_slots_per_row
        num_positions = num_rows * num_bays
        
        # Number of positions to fill
        num_to_fill = int(num_positions * 0.3)
        
        # Randomly select positions to fill
        positions_to_fill = np.random.choice(num_positions, num_to_fill, replace=False)
        
        # List to keep track of container IDs for later scheduling
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
            
            # Set appropriate departure date based on wait time
            container.departure_date = self.current_simulation_datetime + timedelta(hours=wait_time)
            
            # Respect trailer/swap body placement constraint
            if container.container_type in ["Trailer", "Swap Body"]:
                # Only place in the row nearest to driving lane
                if row != self.terminal.storage_row_names[0]:
                    # Create a different type of container
                    container = ContainerFactory.create_random(config=self.config)
                    # If still a special type, skip this position
                    if container.container_type in ["Trailer", "Swap Body"]:
                        continue
            
            # Add to storage yard
            self.storage_yard.add_container(position, container)
            
            # Randomly add a second container (20% chance)
            if np.random.random() < 0.2:
                # Create another container
                container2 = ContainerFactory.create_random(config=self.config)
                # Check stacking compatibility
                if container2.can_stack_with(container):
                    self.storage_yard.add_container(position, container2, tier=2)
                    self.stored_container_ids.append(container2.container_id)

    def _execute_crane_movement(self, crane_idx, source_idx, destination_idx):
        """Execute a crane movement action."""
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

    def _calculate_priority_from_wait_time(self, wait_time):
        """
        Calculate container priority based on wait time.
        
        Args:
            wait_time: Time in hours until pickup
            
        Returns:
            Priority value (lower is higher priority)
        """
        # Base priority starts at 100
        priority = 100
        
        # If pickup is very soon, high priority
        if wait_time < 24:  # Less than a day
            priority -= 50  # Highest priority
        elif wait_time < 48:  # Less than two days
            priority -= 30
        elif wait_time < 72:  # Less than three days
            priority -= 10
        
        # Ensure priority doesn't go below 1
        return max(1, priority)
    
    def _schedule_trains(self):
        """Schedule trains based on KDE model."""
        # Use existing train schedule to add trains to the queue
        for train_id, planned_arrival, realized_arrival in self.train_schedule:
            # Convert datetime to seconds from simulation start
            arrival_time_seconds = (realized_arrival - self.base_simulation_date).total_seconds()
            
            if arrival_time_seconds < 0:
                continue  # Skip trains scheduled before simulation start
                
            if arrival_time_seconds > self.max_simulation_time:
                continue  # Skip trains beyond simulation time
            
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
        # Make sure stored_container_ids exists and has items
        if not hasattr(self, 'stored_container_ids') or not self.stored_container_ids:
            self.stored_container_ids = []
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

    def _execute_truck_parking(self, truck_idx, parking_spot_idx):
        """Execute a truck parking action."""
        # Get the truck from the queue
        if truck_idx >= self.truck_queue.size():
            # Invalid truck index, return with no action
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
            # Parking spot already occupied, return with no action
            observation = self._get_observation()
            reward = -1  # Small penalty for invalid action
            terminated = self.current_simulation_time >= self.max_simulation_time
            truncated = False
            info = {"action_type": "truck_parking", "result": "parking_spot_occupied"}
            return observation, reward, terminated, truncated, info
        
        # Get the truck from the queue (without removing it yet)
        truck_list = list(self.truck_queue.vehicles.queue)
        truck = truck_list[truck_idx]
        
        # Now perform the actual assignment
        self.truck_queue.vehicles.queue.remove(truck)  # Remove from queue
        truck.parking_spot = parking_spot
        truck.status = "waiting"
        self.trucks_in_terminal[parking_spot] = truck
        
        # Calculate reward for the truck parking action based on proximity to relevant wagon
        reward = self._calculate_truck_parking_reward(truck, parking_spot)
        
        # Advance time slightly for the truck parking action (e.g., 1 minute = 60 seconds)
        time_advanced = 60.0
        self.current_simulation_time += time_advanced
        
        # Process any events that happen during this time advancement
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
        Get the current observation of the environment with proper action masking.
        
        Returns:
            Dictionary containing observation data with valid action mask
        """
        # Create basic observation dictionary
        observation = {
            # Crane positions and availability
            'crane_positions': np.array([crane.current_position for crane in self.cranes], dtype=np.int32),
            'crane_available_times': np.array(self.crane_available_times, dtype=np.float32),
            
            # Current simulation time - ensure this key exists
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
        
        # Generate action mask with strict rules enforcement
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
                container = self._get_container_at_position(source_position, self.storage_yard, 
                                                        self.trucks_in_terminal, self.trains_in_terminal)
                
                if container is None:
                    continue
                    
                # Get valid destinations based on our rules
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
                        
                        # Check for weight compatibility if stacking on top of existing container
                        source_container = self._get_container_at_position(source_position, self.storage_yard, 
                                                                        self.trucks_in_terminal, self.trains_in_terminal)
                        
                        # Check if there's already a container at the destination
                        existing_container, _ = self.storage_yard.get_top_container(dest_position)
                        
                        # If there's a container at the destination and we're trying to stack on top of it
                        if existing_container is not None and source_container is not None:
                            # Check if the container can be safely stacked based on weight
                            if not source_container.can_be_stacked_on(existing_container):
                                continue  # Skip this move if stacking is unsafe due to weight
                    
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
        
            # Generate action mask for truck parking
            truck_parking_mask = np.zeros((
                10,  # Max trucks in queue to consider
                len(self.parking_spots)
            ), dtype=np.int8)
            
            # Check which parking spots are available
            available_spots = [i for i, spot in enumerate(self.parking_spots) 
                            if spot not in self.trucks_in_terminal]
            
            # Get trucks from the queue
            trucks_in_queue = list(self.truck_queue.vehicles.queue)
            
            # For each truck in the queue, determine valid parking spots
            for truck_idx, truck in enumerate(trucks_in_queue):
                if truck_idx >= 10:  # Only consider first 10 trucks for masking
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
                                        track_id_str = target_position.split('_')[0]
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
                                            track_id_str = target_position.split('_')[0]
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
    
    def _calculate_reward(self, container, source_position, destination_position, time_taken):
        """
        Calculate the reward for moving a container with time and distance penalties.
        
        Args:
            container: The container that was moved
            source_position: Source position
            destination_position: Destination position
            time_taken: Time taken for the move in seconds
            
        Returns:
            Calculated reward value
        """
        # Base reward starts at zero
        reward = 0.0
        
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
                distance = time_taken * 0.5  # Rough estimate of distance
        
        # Empty crane movement penalty
        if container is None:
            # Higher penalty for empty moves
            empty_move_penalty = -5.0
            # Additional penalty based on distance and time
            distance_time_penalty = -0.05 * distance - time_taken / 60.0  # More penalty for longer moves
            
            # Log the penalty calculation for debugging
            print(f"Empty move: base={empty_move_penalty}, distance={distance:.2f}m, time={time_taken:.2f}s")
            print(f"Distance-time penalty: {distance_time_penalty:.2f}")
            
            return empty_move_penalty + distance_time_penalty
        
        # If we have a container, determine the reward based on move type
        move_type_reward = 0.0
        
        # GOLDEN MOVE: DIRECT TRAIN TO TRUCK - HIGHEST REWARD
        if source_type == 'train' and dest_type == 'truck':
            move_type_reward = 10.0  # Best possible move
            print(f"Golden move (train→truck): +{move_type_reward:.2f}")

        # GOLDEN MOVE: DIRECT TRAIN TO TRUCK - HIGHEST REWARD
        if source_type == 'truck' and dest_type == 'train':
            move_type_reward = 10.0  # Best possible move
            print(f"Golden move (truck→train): +{move_type_reward:.2f}")
        
        # GOOD MOVE: STORAGE TO TRUCK OR TRAIN - GOOD REWARD
        elif source_type == 'storage' and (dest_type == 'truck' or dest_type == 'train'):
            move_type_reward = 3.0
            print(f"Good move (storage→{dest_type}): +{move_type_reward:.2f}")
            
            # DEADLINE BONUS: Container moved before deadline
            if hasattr(container, 'departure_date') and container.departure_date:
                time_until_deadline = (container.departure_date - self.current_simulation_datetime).total_seconds()
                if time_until_deadline > 0:
                    # Scale reward based on time till deadline (higher for tighter deadlines)
                    time_factor = min(1.0, 24*3600 / max(3600, time_until_deadline))  # Cap at 24 hours
                    deadline_bonus = 5.0 * time_factor  # Maximum +5 bonus for very tight deadlines
                    move_type_reward += deadline_bonus
                    print(f"Deadline bonus: +{deadline_bonus:.2f} (due in {time_until_deadline/3600:.1f}h)")
        
        # STANDARD MOVES: TRAIN/TRUCK TO STORAGE - MODEST REWARD
        elif (source_type == 'train' or source_type == 'truck') and dest_type == 'storage':
            move_type_reward = 2.0
            print(f"Standard move ({source_type}→storage): +{move_type_reward:.2f}")
        
        # RESHUFFLING: STORAGE TO STORAGE - PENALTY
        elif source_type == 'storage' and dest_type == 'storage':
            move_type_reward = -4.0
            print(f"Reshuffling penalty: {move_type_reward:.2f}")
        
        # TRUCK PARKING: ASSIGN TRUCK TO SPOT - SMALL REWARD
        elif source_type == 'truck' and dest_type == 'parking':
            move_type_reward = 2.0
            print(f"Truck parking assignment: +{move_type_reward:.2f}")
        
        # Add the move type reward to base reward
        reward += move_type_reward
        
        # DEADLINE PENALTY: Container moved after deadline
        deadline_penalty = 0.0
        if hasattr(container, 'departure_date') and container.departure_date:
            time_past_deadline = (self.current_simulation_datetime - container.departure_date).total_seconds()
            if time_past_deadline > 0:
                # More punishment the longer past deadline
                past_deadline_hours = time_past_deadline / 3600
                deadline_penalty = -min(10.0, past_deadline_hours * 0.5)  # Cap at -10
                reward += deadline_penalty
                print(f"Past deadline penalty: {deadline_penalty:.2f} ({past_deadline_hours:.1f}h late)")
        
        # PRIORITY BONUS: Based on container priority
        priority_bonus = 0.0
        if hasattr(container, 'priority'):
            # Higher priority (lower number) gives higher reward
            priority_factor = max(0, (100 - container.priority) / 100)
            priority_bonus = priority_factor * 2.0
            reward += priority_bonus
            print(f"Priority bonus: +{priority_bonus:.2f} (priority {container.priority})")
        
        # DISTANCE AND TIME PENALTY
        distance_time_penalty = 0.0
        
        # Scale based on distance
        distance_penalty = -0.02 * distance  # -0.02 per meter
        
        # Scale based on time
        time_penalty = -min(time_taken / 120, 1.0)  # Cap at -1 for moves taking over 2 minutes
        
        distance_time_penalty = distance_penalty + time_penalty
        reward += distance_time_penalty
        
        print(f"Distance-time penalty: {distance_time_penalty:.2f} (dist={distance:.1f}m, time={time_taken:.1f}s)")
        
        # Final reward (rounded to 2 decimal places for cleaner reporting)
        final_reward = round(reward, 2)
        print(f"Total reward: {final_reward:.2f}")
        
        return final_reward
    
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
        # Update current simulation datetime
        self.current_simulation_datetime += timedelta(seconds=time_advanced)
        
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
        # Update queues with current time
        arrived_trains = self.train_queue.update(self.current_simulation_datetime)
        arrived_trucks = self.truck_queue.update(self.current_simulation_datetime)
        
        # Generate additional trucks based on KDE model if needed
        truck_arrival_probability = min(0.8, time_advanced / 3600)  # Cap at 80% per hour
        if np.random.random() < truck_arrival_probability:
            # Create a random truck
            truck = Truck()
            # Sample arrival time
            truck_hour = self.config.sample_from_kde('truck_pickup', n_samples=1)[0]
            arrival_time = self.config.hours_to_datetime(truck_hour, self.current_simulation_datetime)
            
            # Randomly decide if bringing container or picking up
            if np.random.random() < 0.5:
                # Truck bringing a container
                container = ContainerFactory.create_random(config=self.config)
                truck.add_container(container)
            else:
                # Truck coming to pick up
                # In a real system, this would be a specific container
                pickup_id = f"CONT{np.random.randint(1000, 9999)}"
                truck.add_pickup_container_id(pickup_id)
            
            self.truck_queue.schedule_arrival(truck, arrival_time)
        
        # Process arrivals from queues
        self._process_truck_arrivals()
        self._process_train_arrivals()
    
    def _process_vehicle_departures(self):
        """Process vehicle departures."""
        self._process_truck_departures()
        self._process_train_departures()
    
    def _process_truck_arrivals(self):
        """Process trucks from the queue into available parking spots."""
        pass
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
    # Create terminal config
    from config import TerminalConfig
    config = TerminalConfig()
    
    # Create environment with the config
    env = TerminalEnvironment(terminal_config=config)
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
        if hasattr(env.action_space, 'sample'):
            action = env.action_space.sample()  # Random action
        else:
            # For Dict action space
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
        
        # Track last action for rendering
        if "action_type" in info:
            if info["action_type"] == "crane_movement" and "container_moved" in info and info["container_moved"]:
                env.last_action = f"Crane moved {info['container_moved']}"
            elif info["action_type"] == "truck_parking":
                env.last_action = f"Truck {info.get('truck_id', 'unknown')} parked at {info.get('parking_spot', 'unknown')}"
        
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