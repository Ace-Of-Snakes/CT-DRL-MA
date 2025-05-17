import numpy as np
import warp as wp
import torch
from datetime import datetime, timedelta
import time
import os
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional, Any, Union

# Placeholder for kernels
_kernel_instances = {
    'crane_mask': None,
    'truck_parking_mask': None, 
    'terminal_truck_mask': None
}

class WarpTerminalEnvironment(gym.Env):
    """Container Terminal Environment using NVIDIA Warp for GPU-accelerated simulation."""
    
    def __init__(self,
                 terminal_config=None,
                 max_simulation_time=86400,  # 1 day in seconds
                 num_cranes=2,
                 num_terminal_trucks=2,
                 num_rail_tracks=6,
                 num_rail_slots_per_track=29,
                 num_storage_rows=5,
                 num_storage_bays=58,
                 max_stack_height=5,
                 num_parking_spots=29,
                 max_containers=1000,
                 max_vehicles=100,
                 device=None,
                 log_performance=False):
        """Initialize the Warp-accelerated terminal environment."""
        super(WarpTerminalEnvironment, self).__init__()
        
        # Defer warp import to avoid circular import issues
        # This is now inside a function call, not at module level
        self._init_warp(device)
        
        # Store configuration
        self.terminal_config = terminal_config
        self.max_simulation_time = max_simulation_time
        self.num_cranes = num_cranes
        self.num_terminal_trucks = num_terminal_trucks
        
        # Terminal dimensions
        self.num_rail_tracks = num_rail_tracks
        self.num_rail_slots_per_track = num_rail_slots_per_track
        self.num_storage_rows = num_storage_rows
        self.num_storage_bays = num_storage_bays
        self.max_stack_height = max_stack_height
        self.num_parking_spots = num_parking_spots
        
        # Maximum capacities
        self.max_containers = max_containers
        self.max_vehicles = max_vehicles
        
        # Performance tracking
        self.log_performance = log_performance
        self.step_times = []
        self.reset_times = []
        
        # Initialize Warp components - now after warp is fully initialized
        self._init_components()
        
        # Initialize simulation time
        self.current_simulation_time = 0.0
        self.current_simulation_datetime = datetime.now()
        
        # Initialize vehicle limits
        self.max_trucks_per_day = 20
        self.max_trains_per_day = 5
        
        # Initialize vehicle and container tracking
        self.trucks_arrived = 0
        self.trains_arrived = 0
        self.containers_moved = 0
        
        # Initialize queues for trucks and trains
        self.truck_queue = QueueWrapper(max_size=100)
        self.train_queue = QueueWrapper(max_size=20)
        
        # Initialize action and observation spaces
        self._setup_action_observation_spaces()
        
        # Set up position mappings
        self._setup_position_mappings()
        
        # Initialize vehicles and containers
        self._initialize_simulation()
        
        # Simplified rendering mode (for faster training)
        self.simplified_rendering = False
        
        # Initialize optimization flags (default to disabled)
        self.use_precomputed_events = False
        self.use_precomputed_movements = False
        self.use_precomputed_stacking = False
        
        # Optimization performance tracking
        self.optimized_step_times = []

    def _setup_action_observation_spaces(self):
        """Set up the action and observation spaces for the environment."""
        # Calculate total number of positions
        total_positions = (self.num_rail_tracks * self.num_rail_slots_per_track + 
                         self.num_parking_spots + 
                         self.num_storage_rows * self.num_storage_bays)
        
        # Define individual action spaces
        crane_movement_space = spaces.MultiDiscrete([
            self.num_cranes,           # Crane index
            total_positions,           # Source position
            total_positions            # Destination position
        ])
        
        truck_parking_space = spaces.MultiDiscrete([
            self.max_vehicles,         # Truck index
            self.num_parking_spots     # Parking spot index
        ])
        
        terminal_truck_space = spaces.MultiDiscrete([
            self.num_terminal_trucks,  # Terminal truck index
            total_positions,           # Source position
            total_positions            # Destination position
        ])
        
        # Combined action space as a Dict
        self.action_space = spaces.Dict({
            'action_type': spaces.Discrete(3),  # 0: Crane, 1: Parking, 2: Terminal Truck
            'crane_movement': crane_movement_space,
            'truck_parking': truck_parking_space,
            'terminal_truck': terminal_truck_space
        })
        
        # Observation space definition
        # Calculate observation shape for each component
        crane_positions_shape = (self.num_cranes, 2)  # (x, y) for each crane
        crane_available_times_shape = (self.num_cranes,)
        terminal_truck_available_times_shape = (self.num_terminal_trucks,)
        current_time_shape = (1,)
        
        # Yard state has multiple features per position
        yard_state_shape = (self.num_storage_rows, self.num_storage_bays, 7)  # 7 features per position
        
        # Parking and rail status
        parking_status_shape = (self.num_parking_spots,)
        rail_status_shape = (self.num_rail_tracks,)
        
        # Queue sizes
        queue_sizes_shape = (2,)  # Train and truck queues
        
        # Combine into observation space
        self.observation_space = spaces.Dict({
            'crane_positions': spaces.Box(low=-float('inf'), high=float('inf'), shape=crane_positions_shape),
            'crane_available_times': spaces.Box(low=0, high=float('inf'), shape=crane_available_times_shape),
            'terminal_truck_available_times': spaces.Box(low=0, high=float('inf'), shape=terminal_truck_available_times_shape),
            'current_time': spaces.Box(low=0, high=float('inf'), shape=current_time_shape),
            'yard_state': spaces.Box(low=0, high=1, shape=yard_state_shape),
            'parking_status': spaces.Box(low=-1, high=1, shape=parking_status_shape),
            'rail_status': spaces.Box(low=-1, high=1, shape=rail_status_shape),
            'queue_sizes': spaces.Box(low=0, high=100, shape=queue_sizes_shape),
            'action_mask': spaces.Dict({
                'crane_movement': spaces.Box(low=0, high=1, shape=crane_movement_space.nvec),
                'truck_parking': spaces.Box(low=0, high=1, shape=truck_parking_space.nvec),
                'terminal_truck': spaces.Box(low=0, high=1, shape=terminal_truck_space.nvec)
            })
        })
    
    def _setup_position_mappings(self):
        """Set up mappings between position strings and indices."""
        self.position_to_idx = {}
        self.idx_to_position = {}
        
        # Copy mappings from terminal state
        self.position_to_idx = self.terminal_state.position_to_idx.copy()
        self.idx_to_position = self.terminal_state.idx_to_position.copy()
    
    def _initialize_simulation(self):
        """Initialize the simulation with starting vehicles and containers."""
        # Clear existing vehicles and containers
        self._reset_terminal_state()
        
        # Set up initial crane positions
        self._initialize_cranes()
        
        # Generate initial containers in storage yard
        self._generate_initial_containers()
        
        # Schedule initial vehicle arrivals
        self._schedule_vehicle_arrivals()
    
    def _reset_terminal_state(self):
        """Reset all terminal state arrays."""
        # Reset container arrays
        self.terminal_state.container_positions.fill_(-1)
        self.terminal_state.container_vehicles.fill_(-1)
        self.terminal_state.container_properties[:, 6].fill_(0.0)  # Inactive containers
        
        # Reset yard arrays
        self.terminal_state.yard_container_indices.fill_(-1)
        self.terminal_state.stack_heights.fill_(0)
        
        # Reset vehicle arrays
        self.terminal_state.vehicle_positions.fill_(-1)
        self.terminal_state.vehicle_properties[:, 6].fill_(0.0)  # Inactive vehicles
        self.terminal_state.vehicle_containers.fill_(-1)
        self.terminal_state.vehicle_container_counts.fill_(0)
        self.terminal_state.parking_vehicles.fill_(-1)
        self.terminal_state.rail_track_vehicles.fill_(-1)
        
        # Reset queues
        self.terminal_state.vehicle_queues.fill_(-1)
        self.terminal_state.queue_sizes.fill_(0)
        
        # Reset crane state
        self.terminal_state.crane_properties[:, 0].fill_(0.0)  # Idle status
        self.terminal_state.crane_properties[:, 1].fill_(-1.0)  # No container loaded
        self.terminal_state.crane_properties[:, 2].fill_(0.0)  # Available time
    
    def _initialize_cranes(self):
        """Initialize cranes at their starting positions."""
        # Create a CPU-side array first
        num_cranes = self.num_cranes
        crane_positions_np = np.zeros((num_cranes, 2), dtype=np.float32)
        crane_target_positions_np = np.zeros((num_cranes, 2), dtype=np.float32)
        crane_operational_areas_np = np.zeros((num_cranes, 4), dtype=np.float32)
        
        # Set initial positions for cranes
        # Place them evenly across the terminal
        terminal_length = self.num_rail_slots_per_track * 24.0  # Approximate slot length
        for i in range(num_cranes):
            # Distribute cranes evenly along x-axis
            x_pos = (i + 1) * terminal_length / (num_cranes + 1)
            
            # Place cranes at the center of rails area
            y_pos = self.num_rail_tracks * 2.5  # Approximate position
            
            # Set crane position in NumPy array
            crane_positions_np[i, 0] = x_pos
            crane_positions_np[i, 1] = y_pos
            
            # Set crane target position to same as current position
            crane_target_positions_np[i, 0] = x_pos
            crane_target_positions_np[i, 1] = y_pos
            
            # Define operational area for crane
            crane_operational_areas_np[i, 0] = 0.0  # min_x
            crane_operational_areas_np[i, 1] = 0.0  # min_y
            crane_operational_areas_np[i, 2] = terminal_length  # max_x
            
            # If multiple cranes, divide the y-range
            if num_cranes > 1:
                area_height = self.num_storage_rows * 10.0  # Approximate
                section_height = area_height / num_cranes
                crane_operational_areas_np[i, 1] = i * section_height
                crane_operational_areas_np[i, 3] = (i + 1) * section_height
            else:
                # Single crane covers the entire terminal height
                crane_operational_areas_np[i, 3] = self.num_storage_rows * 10.0
        
        # Convert NumPy arrays to Warp arrays in one step
        self.terminal_state.crane_positions = wp.array(crane_positions_np, dtype=wp.float32, device=self.device)
        self.terminal_state.crane_target_positions = wp.array(crane_target_positions_np, dtype=wp.float32, device=self.device)
        self.terminal_state.crane_operational_areas = wp.array(crane_operational_areas_np, dtype=wp.float32, device=self.device)
    
    def _generate_initial_containers(self):
        """Generate initial containers in the storage yard."""
        # Number of initial containers to generate
        num_initial_containers = int(self.num_storage_rows * self.num_storage_bays * 0.3)  # 30% occupancy
        
        # Generate containers
        for i in range(num_initial_containers):
            # Create a random container
            container_idx = self.container_registry.create_random_container()
            
            # Find a valid position for the container
            row = np.random.randint(0, self.num_storage_rows)
            bay = np.random.randint(0, self.num_storage_bays)
            
            # Get position string
            row_letter = chr(65 + row)  # Convert row index to letter (A, B, C, ...)
            position_str = f"{row_letter}{bay+1}"
            
            # Check if we can place the container here
            if not self.storage_yard.can_accept_container(position_str, container_idx):
                # If not, find optimal location
                optimal_location = self.storage_yard.find_optimal_location(container_idx)
                if optimal_location:
                    position_str = optimal_location
                else:
                    # Skip this container if no valid position found
                    continue
            
            # Place container in yard
            self.storage_yard.add_container(container_idx, position_str)
    
    def _schedule_vehicle_arrivals(self):
        """Schedule initial vehicle arrivals."""
        # Schedule trucks
        num_trucks = np.random.randint(5, self.max_trucks_per_day + 1)
        for i in range(num_trucks):
            # Random arrival time within the first day
            arrival_time = np.random.uniform(0, self.max_simulation_time * 0.5)
            
            # Add to event queue
            self._add_event(arrival_time, 0, i)  # Event type 0 = Truck arrival
        
        # Schedule trains
        num_trains = np.random.randint(1, self.max_trains_per_day + 1)
        for i in range(num_trains):
            # Trains arrive at more specific times
            arrival_time = np.random.uniform(0, self.max_simulation_time * 0.7)
            
            # Add to event queue
            self._add_event(arrival_time, 1, i)  # Event type 1 = Train arrival
        
    @wp.kernel
    def _kernel_add_event_to_queue(event_queue: wp.array(dtype=wp.float32, ndim=2),
                                queue_idx: wp.int32,
                                time: wp.float32,
                                event_type: wp.int32,
                                event_data: wp.int32):
        """Add event to the event queue."""
        event_queue[queue_idx, 0] = time
        event_queue[queue_idx, 1] = float(event_type)  # Cast to float since the array is float32
        event_queue[queue_idx, 2] = float(event_data)  # Cast to float since the array is float32

    @wp.kernel
    def _kernel_update_queue_size(queue_size: wp.array(dtype=wp.int32, ndim=1),
                            new_size: wp.int32):
        """Update the event queue size."""
        queue_size[0] = new_size

    def _add_event(self, time, event_type, event_data):
        """Add an event to the queue at the specified time."""
        # Get current event queue size
        queue_size_np = self.terminal_state.event_queue_size.numpy()
        queue_size = int(queue_size_np[0])
        
        # Check if there's room in the queue
        if queue_size >= 1000:  # Max queue size
            return False
        
        # Add event to queue using kernel
        wp.launch(
            kernel=self._kernel_add_event_to_queue,
            dim=1,
            inputs=[
                self.terminal_state.event_queue,
                queue_size,
                float(time),
                int(event_type),
                int(event_data)
            ]
        )
        
        # Increment queue size using kernel
        wp.launch(
            kernel=self._kernel_update_queue_size,
            dim=1,
            inputs=[
                self.terminal_state.event_queue_size,
                queue_size + 1
            ]
        )
        
        return True
    
    @wp.kernel
    def _kernel_get_event_data(event_queue: wp.array(dtype=wp.float32, ndim=2),
                            event_idx: wp.int32,
                            field_idx: wp.int32,
                            result: wp.array(dtype=wp.float32, ndim=1)):
        """Get event data from the event queue."""
        result[0] = event_queue[event_idx, field_idx]

    @wp.kernel
    def _kernel_set_event_data(event_queue: wp.array(dtype=wp.float32, ndim=2),
                            event_idx: wp.int32,
                            field_idx: wp.int32,
                            value: wp.float32):
        """Set event data in the event queue."""
        event_queue[event_idx, field_idx] = value

    @wp.kernel
    def _kernel_shift_event(event_queue: wp.array(dtype=wp.float32, ndim=2),
                        src_idx: wp.int32, 
                        dst_idx: wp.int32):
        """Shift an event from source to destination index."""
        # Copy all fields (0, 1, 2)
        event_queue[dst_idx, 0] = event_queue[src_idx, 0]
        event_queue[dst_idx, 1] = event_queue[src_idx, 1]
        event_queue[dst_idx, 2] = event_queue[src_idx, 2]
    
    def _process_vehicle_arrivals(self, time_advance=0):
        """Process vehicle arrivals for the current time."""
        current_time = self.current_simulation_time
        
        # Process events from the queue
        queue_size_np = self.terminal_state.event_queue_size.numpy()
        queue_size = int(queue_size_np[0])
        events_to_remove = []
        
        # Create result array for event data
        event_data = wp.zeros(1, dtype=wp.float32, device=self.device)
        
        # First, identify events to process
        for i in range(queue_size):
            # Get event time
            wp.launch(
                kernel=self._kernel_get_event_data,
                dim=1,
                inputs=[
                    self.terminal_state.event_queue,
                    i,
                    0,  # Time field
                    event_data
                ]
            )
            event_time = float(event_data.numpy()[0])
            
            # Check if event should trigger now
            if event_time <= current_time:
                # Get event type
                wp.launch(
                    kernel=self._kernel_get_event_data,
                    dim=1,
                    inputs=[
                        self.terminal_state.event_queue,
                        i,
                        1,  # Event type field
                        event_data
                    ]
                )
                event_type = int(event_data.numpy()[0])
                
                # Get event data
                wp.launch(
                    kernel=self._kernel_get_event_data,
                    dim=1,
                    inputs=[
                        self.terminal_state.event_queue,
                        i,
                        2,  # Event data field
                        event_data
                    ]
                )
                event_data_value = int(event_data.numpy()[0])
                
                # Process the event
                if event_type == 0:  # Truck arrival
                    self._create_truck_arrival()
                elif event_type == 1:  # Train arrival
                    self._create_train_arrival()
                
                # Mark for removal
                events_to_remove.append(i)
        
        # Remove processed events (backwards to maintain indices)
        for i in sorted(events_to_remove, reverse=True):
            # Shift remaining events forward
            for j in range(i, queue_size - 1):
                # Move event j+1 to position j
                wp.launch(
                    kernel=self._kernel_shift_event,
                    dim=1,
                    inputs=[
                        self.terminal_state.event_queue,
                        j + 1,
                        j
                    ]
                )
            
            # Decrement queue size
            queue_size -= 1
        
        # Update queue size
        wp.launch(
            kernel=self._kernel_update_queue_size,
            dim=1,
            inputs=[
                self.terminal_state.event_queue_size,
                queue_size
            ]
        )
        
        # Schedule new arrivals if needed
        if time_advance > 0:
            # Probability of a new truck arrival
            if np.random.random() < 0.1 and self.trucks_arrived < self.max_trucks_per_day:
                # Schedule truck arrival
                arrival_time = current_time + np.random.uniform(0, time_advance)
                self._add_event(arrival_time, 0, 0)
            
            # Probability of a new train arrival
            if np.random.random() < 0.05 and self.trains_arrived < self.max_trains_per_day:
                # Schedule train arrival
                arrival_time = current_time + np.random.uniform(0, time_advance)
                self._add_event(arrival_time, 1, 0)
                
    @wp.kernel
    def _kernel_update_container_position(container_positions: wp.array(dtype=wp.int32, ndim=1),
                                        container_idx: wp.int32,
                                        position_idx: wp.int32):
        """Update a container's position index."""
        container_positions[container_idx] = position_idx
        
    def _parse_position(self, position_str):
        """Parse position string into row and bay indices."""
        if len(position_str) < 2 or not position_str[0].isalpha():
            return None, None
            
        row_letter = position_str[0].upper()
        
        try:
            bay_num = int(position_str[1:])
        except ValueError:
            return None, None
        
        # Get row index from letter
        if row_letter not in self.storage_yard.row_names:
            return None, None
            
        row_idx = self.storage_yard.row_names.index(row_letter)
        bay_idx = bay_num - 1  # Convert to 0-based index
        
        return row_idx, bay_idx
        
    @wp.kernel
    def _kernel_set_vehicle_properties(vehicle_properties: wp.array(dtype=wp.float32, ndim=2),
                                    vehicle_idx: wp.int32,
                                    props: wp.array(dtype=wp.float32, ndim=1)):
        """Set vehicle properties."""
        for i in range(props.shape[0]):
            vehicle_properties[vehicle_idx, i] = props[i]

    @wp.kernel
    def _kernel_set_element_vehicle_container(vehicle_containers: wp.array(dtype=wp.int32, ndim=2),
                                          vehicle_idx: wp.int32,
                                          container_slot: wp.int32,
                                          container_idx: wp.int32):
        """Set a container on a vehicle in a specific slot."""
        vehicle_containers[vehicle_idx, container_slot] = container_idx

    @wp.kernel
    def _kernel_set_element_container_vehicle(container_vehicles: wp.array(dtype=wp.int32, ndim=1),
                                          container_idx: wp.int32,
                                          vehicle_idx: wp.int32):
        """Set which vehicle a container is on."""
        container_vehicles[container_idx] = vehicle_idx

    @wp.kernel
    def _kernel_set_int_value(arr: wp.array(dtype=wp.int32, ndim=1), 
                          idx: wp.int32, 
                          value: wp.int32):
        """Set a value in an int array at the specified index."""
        arr[idx] = value

    @wp.kernel
    def _kernel_set_element_queue_size(queue_sizes: wp.array(dtype=wp.int32, ndim=1),
                                  queue_idx: wp.int32,
                                  size: wp.int32):
        """Set queue size for a specific queue."""
        queue_sizes[queue_idx] = size

    @wp.kernel
    def _kernel_set_element_vehicle_queue(vehicle_queues: wp.array(dtype=wp.int32, ndim=2),
                                      queue_idx: wp.int32,
                                      pos_idx: wp.int32,
                                      vehicle_idx: wp.int32):
        """Set vehicle in queue at specified position."""
        vehicle_queues[queue_idx, pos_idx] = vehicle_idx

    @wp.kernel
    def _kernel_get_queue_size(queue_sizes: wp.array(dtype=wp.int32, ndim=1),
                          queue_idx: wp.int32,
                          result: wp.array(dtype=wp.int32, ndim=1)):
        """Get queue size and store in result array."""
        result[0] = queue_sizes[queue_idx]

    def _create_truck_arrival(self):
        """Create a new truck arrival and add it to the queue."""
        # Find an available vehicle slot
        vehicle_idx = -1
        for i in range(self.max_vehicles):
            if self.terminal_state.vehicle_properties.numpy()[i, 6] == 0:  # Inactive
                vehicle_idx = i
                break
        
        if vehicle_idx == -1:
            return None  # No available slots
        
        # Decide if it's a pickup or delivery truck
        is_pickup = np.random.random() < 0.5
        
        # Set truck properties
        # Create a CPU array with values
        vehicle_props = np.zeros(7, dtype=np.float32)
        vehicle_props[0] = 0.0  # Truck type
        vehicle_props[1] = 1.0  # WAITING status
        vehicle_props[2] = 1.0 if is_pickup else 0.0  # Pickup flag
        vehicle_props[3] = 1.0  # Max containers
        vehicle_props[4] = self.current_simulation_time  # Arrival time
        vehicle_props[5] = 0.0  # Departure time
        vehicle_props[6] = 1.0  # Active flag
        
        # Update the whole row at once using a kernel
        wp.launch(
            kernel=self._kernel_set_vehicle_properties,
            dim=1,
            inputs=[
                self.terminal_state.vehicle_properties,
                vehicle_idx,
                wp.array(vehicle_props, dtype=wp.float32, device=self.device)
            ]
        )
        
        # If it's a delivery truck, add a container
        if not is_pickup:
            # Create a random container
            container_idx = self.container_registry.create_random_container()
            
            # Assign container to truck using kernels
            wp.launch(
                kernel=self._kernel_set_element_vehicle_container,
                dim=1,
                inputs=[
                    self.terminal_state.vehicle_containers,
                    vehicle_idx,
                    0,  # First slot
                    container_idx
                ]
            )
            
            wp.launch(
                kernel=self._kernel_set_int_value,
                dim=1,
                inputs=[
                    self.terminal_state.vehicle_container_counts,
                    vehicle_idx,
                    1  # One container
                ]
            )
            
            wp.launch(
                kernel=self._kernel_set_element_container_vehicle,
                dim=1,
                inputs=[
                    self.terminal_state.container_vehicles,
                    container_idx,
                    vehicle_idx
                ]
            )
        
        # Get current queue size
        queue_size_result = wp.zeros(1, dtype=wp.int32, device=self.device)
        wp.launch(
            kernel=self._kernel_get_queue_size,
            dim=1,
            inputs=[
                self.terminal_state.queue_sizes,
                0,  # Truck queue index
                queue_size_result
            ]
        )
        queue_size = int(queue_size_result.numpy()[0])
        
        # Add to truck queue
        wp.launch(
            kernel=self._kernel_set_element_vehicle_queue,
            dim=1,
            inputs=[
                self.terminal_state.vehicle_queues,
                0,  # Truck queue
                queue_size,
                vehicle_idx
            ]
        )
        
        # Update queue size
        wp.launch(
            kernel=self._kernel_set_element_queue_size,
            dim=1,
            inputs=[
                self.terminal_state.queue_sizes,
                0,  # Truck queue index
                queue_size + 1
            ]
        )
        
        # Update arrived count
        self.trucks_arrived += 1
        
        return vehicle_idx

    def _create_train_arrival(self):
        """Create a new train arrival and add it to the queue."""
        # Find an available vehicle slot
        vehicle_idx = -1
        vehicle_props_np = self.terminal_state.vehicle_properties.numpy()
        for i in range(self.max_vehicles):
            if vehicle_props_np[i, 6] == 0:  # Inactive
                vehicle_idx = i
                break
        
        if vehicle_idx == -1:
            return None  # No available slots
        
        # Create properties array
        train_props = np.zeros(7, dtype=np.float32)
        train_props[0] = 1.0  # Train type
        train_props[1] = 1.0  # WAITING status
        train_props[2] = 0.0  # Not a pickup train
        train_props[3] = 10.0  # Max containers (wagons)
        train_props[4] = self.current_simulation_time  # Arrival time
        train_props[5] = 0.0  # Departure time
        train_props[6] = 1.0  # Active flag
        
        # Update vehicle properties using kernel
        wp.launch(
            kernel=self._kernel_set_vehicle_properties,
            dim=1,
            inputs=[
                self.terminal_state.vehicle_properties,
                vehicle_idx,
                wp.array(train_props, dtype=wp.float32, device=self.device)
            ]
        )
        
        # Add containers to train
        num_containers = np.random.randint(3, 11)  # 3-10 containers
        container_indices = []
        
        for i in range(num_containers):
            # Create a random container
            container_idx = self.container_registry.create_random_container()
            container_indices.append(container_idx)
            
            # Assign container to train (one at a time)
            wp.launch(
                kernel=self._kernel_set_element_vehicle_container,
                dim=1,
                inputs=[
                    self.terminal_state.vehicle_containers,
                    vehicle_idx,
                    i,
                    container_idx
                ]
            )
            
            # Update container's vehicle
            wp.launch(
                kernel=self._kernel_set_element_container_vehicle,
                dim=1,
                inputs=[
                    self.terminal_state.container_vehicles,
                    container_idx,
                    vehicle_idx
                ]
            )
        
        # Update container count
        wp.launch(
            kernel=self._kernel_set_int_value,
            dim=1,
            inputs=[
                self.terminal_state.vehicle_container_counts,
                vehicle_idx,
                num_containers
            ]
        )
        
        # Add to train queue
        queue_size_np = self.terminal_state.queue_sizes.numpy()
        queue_size = int(queue_size_np[1])
        
        wp.launch(
            kernel=self._kernel_set_int_value,
            dim=1,
            inputs=[
                self.terminal_state.vehicle_queues[1],
                queue_size,
                vehicle_idx
            ]
        )
        
        # Update queue size
        wp.launch(
            kernel=self._kernel_set_int_value,
            dim=1,
            inputs=[
                self.terminal_state.queue_sizes,
                1,
                queue_size + 1
            ]
        )
        
        # Update arrived count
        self.trains_arrived += 1
        
        return vehicle_idx
    
    @wp.kernel
    def _kernel_set_float_value(arr: wp.array(dtype=wp.float32, ndim=1), 
                            idx: wp.int32, 
                            value: wp.float32):
        """Set a value in a float array at the specified index."""
        arr[idx] = value

    def _process_vehicle_departures(self):
        """Process vehicle departures for the current time."""
        current_time = self.current_simulation_time
        
        # Convert to numpy for safe indexing
        vehicle_props_np = self.terminal_state.vehicle_properties.numpy()
        vehicle_positions_np = self.terminal_state.vehicle_positions.numpy()
        
        # Check all vehicles for departures
        for i in range(self.max_vehicles):
            # Only process active vehicles
            if vehicle_props_np[i, 6] == 0:  # Inactive
                continue
                
            # Check if vehicle is in DEPARTING status and ready to depart
            if (vehicle_props_np[i, 1] == 3 and       # DEPARTING status
                vehicle_props_np[i, 5] > 0 and        # Has departure time
                vehicle_props_np[i, 5] <= current_time):  # Time to depart
                
                # Get vehicle type and position
                vehicle_type = int(vehicle_props_np[i, 0])
                position_idx = int(vehicle_positions_np[i])
                
                # Remove from parking/rail using kernel
                if vehicle_type == 0:  # Truck
                    # Find parking spot from numpy arrays
                    parking_vehicles_np = self.terminal_state.parking_vehicles.numpy()
                    for spot in range(self.num_parking_spots):
                        if parking_vehicles_np[spot] == i:
                            # Clear the spot with kernel
                            wp.launch(
                                kernel=self._kernel_set_int_value,
                                dim=1,
                                inputs=[self.terminal_state.parking_vehicles, spot, -1]
                            )
                            break
                else:  # Train
                    # Find rail track from numpy arrays
                    rail_track_vehicles_np = self.terminal_state.rail_track_vehicles.numpy()
                    for track in range(self.num_rail_tracks):
                        if rail_track_vehicles_np[track] == i:
                            # Clear the track with kernel
                            wp.launch(
                                kernel=self._kernel_set_int_value,
                                dim=1,
                                inputs=[self.terminal_state.rail_track_vehicles, track, -1]
                            )
                            break
                
                # Mark vehicle as inactive with kernel
                zero_props = np.zeros(7, dtype=np.float32)
                wp.launch(
                    kernel=self._kernel_set_vehicle_properties,
                    dim=1,
                    inputs=[
                        self.terminal_state.vehicle_properties,
                        i,
                        wp.array(zero_props, dtype=wp.float32, device=self.device)
                    ]
                )
                
                # Clear vehicle position with kernel
                wp.launch(
                    kernel=self._kernel_set_int_value,
                    dim=1,
                    inputs=[self.terminal_state.vehicle_positions, i, -1]
                )
                
                # Get vehicle container info
                vehicle_containers_np = self.terminal_state.vehicle_containers.numpy()
                
                # Remove any remaining containers
                for j in range(10):  # Assuming max 10 containers per vehicle
                    container_idx = vehicle_containers_np[i, j]
                    if container_idx >= 0:
                        # Mark container as inactive with kernel
                        wp.launch(
                            kernel=self._kernel_set_float_value,
                            dim=1,
                            inputs=[self.terminal_state.container_properties, container_idx * 8 + 6, 0.0]  # Index 6 is active flag
                        )
                        
                        # Clear container's vehicle with kernel
                        wp.launch(
                            kernel=self._kernel_set_int_value,
                            dim=1,
                            inputs=[self.terminal_state.container_vehicles, container_idx, -1]
                        )
                        
                        # Clear vehicle's container with kernel
                        wp.launch(
                            kernel=self._kernel_set_int_value,
                            dim=1,
                            inputs=[self.terminal_state.vehicle_containers, i * 10 + j, -1]  # Assuming shape [vehicle_idx, container_slot]
                        )
                
                # Reset container count with kernel
                wp.launch(
                    kernel=self._kernel_set_int_value,
                    dim=1,
                    inputs=[self.terminal_state.vehicle_container_counts, i, 0]
                )
                
                # Schedule new arrivals if needed
                if vehicle_type == 0 and self.trucks_arrived < self.max_trucks_per_day:
                    # Schedule new truck arrival
                    arrival_time = current_time + np.random.uniform(300, 1800)  # 5-30 minutes
                    self._add_event(arrival_time, 0, 0)
                elif vehicle_type == 1 and self.trains_arrived < self.max_trains_per_day:
                    # Schedule new train arrival
                    arrival_time = current_time + np.random.uniform(3600, 7200)  # 1-2 hours
                    self._add_event(arrival_time, 1, 0)
    
    @wp.kernel
    def _kernel_set_element_crane_properties(properties: wp.array(dtype=wp.float32, ndim=2),
                                        crane_idx: wp.int32,
                                        prop_idx: wp.int32,
                                        value: wp.float32):
        """Set a value in crane properties array."""
        properties[crane_idx, prop_idx] = value

    def _execute_crane_movement(self, crane_action):
        """Execute a crane movement action."""
        crane_idx, src_idx, dst_idx = crane_action
        
        # Convert indices to position strings
        src_pos = self.idx_to_position.get(src_idx, None)
        dst_pos = self.idx_to_position.get(dst_idx, None)
        
        if src_pos is None or dst_pos is None:
            # Invalid position indices
            return 0.0
        
        # Check if crane is available
        crane_props = self.terminal_state.crane_properties.numpy()
        crane_available_time = float(crane_props[crane_idx, 2])
        if crane_available_time > self.current_simulation_time:
            return 0.0
        
        # Get container at source position
        container_idx = -1
        if src_pos[0].isalpha() and src_pos[0].upper() in self.storage_yard.row_names:
            # Storage position
            container_idx, _ = self.storage_yard.get_top_container(src_pos)
            if container_idx is None:
                container_idx = -1
        else:
            # Rail or parking position
            # This would normally check vehicles at these positions
            pass
        
        if container_idx < 0:
            # No container at source position
            return 0.0
        
        # Check if container can be placed at destination
        can_place = False
        if dst_pos[0].isalpha() and dst_pos[0].upper() in self.storage_yard.row_names:
            # Storage position
            can_place = self.storage_yard.can_accept_container(dst_pos, container_idx)
        else:
            # Rail or parking position
            # This would normally check vehicle capacity
            can_place = True
        
        if not can_place:
            # Cannot place container at destination
            return 0.0
        
        container_props = self.terminal_state.container_properties.numpy()
        # Get container type from properties (index 0 is the type code)
        container_type = int(container_props[container_idx, 0])
        
        # Get stack height if needed (only needed for storage destinations)
        stack_height = 0.0
        if dst_pos[0].isalpha() and dst_pos[0].upper() in self.storage_yard.row_names:
            row, bay = self._parse_position(dst_pos)
            if row is not None and bay is not None:
                stack_heights_np = self.terminal_state.stack_heights.numpy()
                stack_height = float(stack_heights_np[row, bay])
        
        # Calculate movement time
        movement_time = self.movement_calculator.calculate_movement_time(
            src_pos, dst_pos, crane_idx, container_type=container_type, stack_height=stack_height
        )
        
        # Update crane position and time using kernel
        wp.launch(
            kernel=self._kernel_set_element_crane_properties,
            dim=1,
            inputs=[
                self.terminal_state.crane_properties,
                crane_idx,
                2,  # Index for available time
                self.current_simulation_time + movement_time
            ]
        )
        
        # Update crane position for visualization
        dst_coords = self._get_position_coordinates(dst_pos)
        wp.launch(
            kernel=self._kernel_set_element_crane_properties,
            dim=1,
            inputs=[
                self.terminal_state.crane_positions,
                crane_idx,
                0,  # X coordinate
                dst_coords[0]
            ]
        )
        
        wp.launch(
            kernel=self._kernel_set_element_crane_properties,
            dim=1,
            inputs=[
                self.terminal_state.crane_positions,
                crane_idx,
                1,  # Y coordinate
                dst_coords[1]
            ]
        )
        
        # Remove container from source
        if src_pos[0].isalpha() and src_pos[0].upper() in self.storage_yard.row_names:
            # Storage position
            self.storage_yard.remove_container(src_pos)
        
        # Place container at destination
        if dst_pos[0].isalpha() and dst_pos[0].upper() in self.storage_yard.row_names:
            # Storage position
            self.storage_yard.add_container(container_idx, dst_pos)
        
        # Update container position
        dst_position_idx = self.position_to_idx.get(dst_pos, -1)
        wp.launch(
            kernel=self._kernel_update_container_position,
            dim=1,
            inputs=[self.terminal_state.container_positions, container_idx, dst_position_idx]
        )
        
        # Increment containers moved
        self.containers_moved += 1
        
        # Reward proportional to movement efficiency (inverse of time)
        reward = 10.0 / (1.0 + movement_time / 60.0)  # Normalize to ~0-10 range
        
        return reward
    
    def _execute_truck_parking(self, truck_action):
        """Execute a truck parking action."""
        # Not fully implemented - placeholder for the action
        return 0.0
    
    def _execute_terminal_truck(self, terminal_truck_action):
        """Execute a terminal truck action."""
        # Not fully implemented - placeholder for the action
        return 0.0
    
    def _advance_time(self, seconds):
        """Advance simulation time by the specified number of seconds."""
        # Update simulation time
        self.current_simulation_time += seconds
        self.current_simulation_datetime += timedelta(seconds=seconds)
        
        # Process vehicle arrivals and departures
        self._process_vehicle_arrivals(seconds)
        self._process_vehicle_departures()
    
    def _get_observation(self):
        """Get the current observation."""
        # Create observation dictionary
        
        # Get crane positions and available times
        crane_positions = self.terminal_state.crane_positions.numpy()
        crane_props = self.terminal_state.crane_properties.numpy()
        crane_available_times = np.array([
            max(0, float(crane_props[i, 2]) - self.current_simulation_time)
            for i in range(self.num_cranes)
        ])
        
        # Get terminal truck available times (placeholder)
        terminal_truck_available_times = np.zeros(self.num_terminal_trucks)
        
        # Current time as normalized value
        current_time = np.array([self.current_simulation_time / self.max_simulation_time])
        
        # Get yard state from storage yard
        yard_state = self.storage_yard.get_yard_state()
        
        # Get parking status
        parking_status = np.array([
            float(self.terminal_state.parking_vehicles.numpy()[i]) >= 0
            for i in range(self.num_parking_spots)
        ])
        
        # Get rail status
        rail_status = np.array([
            float(self.terminal_state.rail_track_vehicles.numpy()[i]) >= 0
            for i in range(self.num_rail_tracks)
        ])
        
        # Get queue sizes
        queue_sizes = self.terminal_state.queue_sizes.numpy()
        
        # Generate action masks
        action_mask = self._generate_action_masks()
        
        # Combine into observation
        observation = {
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
        
        return observation
        
    def _init_warp(self, device=None):
        """Safely initialize Warp and register kernels."""
        # Import warp here, not at module level
        
        # Set device
        self.device = device if device else ("cuda" if wp.get_cuda_device_count() > 0 else "cpu")
        print(f"Initializing WarpTerminalEnvironment on device: {self.device}")
        
        # Register kernels only once globally
        global _kernel_instances
        if _kernel_instances['crane_mask'] is None:
            # Define kernel implementations as local functions
            # This avoids the circular reference issue
            
            def kernel_generate_crane_mask(crane_positions: wp.array(dtype=wp.float32, ndim=2),
                                        crane_properties: wp.array(dtype=wp.float32, ndim=2),
                                        yard_container_indices: wp.array(dtype=wp.int32, ndim=3),
                                        stack_heights: wp.array(dtype=wp.int32, ndim=2),
                                        rail_track_vehicles: wp.array(dtype=wp.int32, ndim=1),
                                        parking_vehicles: wp.array(dtype=wp.int32, ndim=1),
                                        container_positions: wp.array(dtype=wp.int32, ndim=1),
                                        current_time: wp.float32,
                                        num_cranes: wp.int32,
                                        num_positions: wp.int32,
                                        crane_mask: wp.array(dtype=wp.int32, ndim=3)):
                # Get thread indices
                crane_idx, src_idx, dst_idx = wp.tid()
                
                # Check bounds
                if (crane_idx >= num_cranes or 
                    src_idx >= num_positions or 
                    dst_idx >= num_positions):
                    return
                
                # Default: invalid action
                crane_mask[crane_idx, src_idx, dst_idx] = 0
                
                # Check if crane is available
                crane_available_time = crane_properties[crane_idx, 2]
                if crane_available_time > current_time:
                    return
                    
                # Check if source position has a container
                source_has_container = bool(False)
                for container_idx in range(container_positions.shape[0]):
                    if container_positions[container_idx] == src_idx:
                        source_has_container = bool(True)
                        break
                        
                if not source_has_container:
                    return
                    
                # Mark as valid action
                crane_mask[crane_idx, src_idx, dst_idx] = 1
            
            def kernel_generate_truck_parking_mask(parking_vehicles: wp.array(dtype=wp.int32, ndim=1),
                                                vehicle_properties: wp.array(dtype=wp.float32, ndim=2),
                                                num_vehicles: wp.int32,
                                                num_parking_spots: wp.int32,
                                                truck_parking_mask: wp.array(dtype=wp.int32, ndim=2)):
                # Get thread indices
                truck_idx, spot_idx = wp.tid()
                
                # Check bounds
                if truck_idx >= num_vehicles or spot_idx >= num_parking_spots:
                    return
                
                # Default: invalid action
                truck_parking_mask[truck_idx, spot_idx] = 0
                
                # Check if truck is active
                if vehicle_properties[truck_idx, 6] <= 0:
                    return
                
                # Check if truck is ready to park (status = WAITING)
                if vehicle_properties[truck_idx, 1] != 1:  # WAITING status
                    return
                
                # Check if parking spot is available
                if parking_vehicles[spot_idx] >= 0:
                    return
                
                # Mark as valid action
                truck_parking_mask[truck_idx, spot_idx] = 1
            
            def kernel_generate_terminal_truck_mask(terminal_truck_positions: wp.array(dtype=wp.int32, ndim=1),
                                                container_positions: wp.array(dtype=wp.int32, ndim=1),
                                                num_terminal_trucks: wp.int32,
                                                num_positions: wp.int32,
                                                terminal_truck_mask: wp.array(dtype=wp.int32, ndim=3)):
                # Get thread indices
                truck_idx, src_idx, dst_idx = wp.tid()
                
                # Check bounds
                if (truck_idx >= num_terminal_trucks or
                    src_idx >= num_positions or
                    dst_idx >= num_positions):
                    return
                
                # Default: invalid action
                terminal_truck_mask[truck_idx, src_idx, dst_idx] = 0
                
                # For now, just allow all terminal truck movements (simplified)
                terminal_truck_mask[truck_idx, src_idx, dst_idx] = 1
            
            # Register the kernels
            _kernel_instances['crane_mask'] = wp.kernel(kernel_generate_crane_mask)
            _kernel_instances['truck_parking_mask'] = wp.kernel(kernel_generate_truck_parking_mask)
            _kernel_instances['terminal_truck_mask'] = wp.kernel(kernel_generate_terminal_truck_mask)
    
    def _init_components(self):
        """Initialize Warp components."""
        # Import Warp components here to avoid circular imports
        from simulation.warp_components.WarpTerminalState import WarpTerminalState
        self.terminal_state = WarpTerminalState(
            num_rail_tracks=self.num_rail_tracks,
            num_rail_slots_per_track=self.num_rail_slots_per_track,
            num_storage_rows=self.num_storage_rows,
            num_storage_bays=self.num_storage_bays,
            max_stack_height=self.max_stack_height,
            num_parking_spots=self.num_parking_spots,
            max_containers=self.max_containers,
            max_vehicles=self.max_vehicles,
            num_cranes=self.num_cranes,
            device=self.device
        )
        
        from simulation.warp_components.WarpContainerRegistry import WarpContainerRegistry
        self.container_registry = WarpContainerRegistry(
            terminal_state=self.terminal_state,
            max_containers=self.max_containers,
            device=self.device
        )
        
        from simulation.warp_components.WarpMovementCalculator import WarpMovementCalculator
        self.movement_calculator = WarpMovementCalculator(
            terminal_state=self.terminal_state,
            device=self.device
        )
        
        from simulation.warp_components.WarpStackingKernels import WarpStackingKernels
        self.stacking_kernels = WarpStackingKernels(
            terminal_state=self.terminal_state,
            container_registry=self.container_registry,
            device=self.device
        )
        
        from simulation.warp_components.WarpStorageYard import WarpStorageYard
        self.storage_yard = WarpStorageYard(
            terminal_state=self.terminal_state,
            container_registry=self.container_registry,
            stacking_kernels=self.stacking_kernels,
            device=self.device
        )

    def _generate_action_masks(self):
        """Generate action masks for the current state."""
        global _kernel_instances
        
        # Calculate total number of positions
        total_positions = len(self.position_to_idx)
        
        # Create arrays for action masks
        crane_mask = np.zeros((self.num_cranes, total_positions, total_positions), dtype=np.int32)
        truck_parking_mask = np.zeros((self.max_vehicles, self.num_parking_spots), dtype=np.int32)
        terminal_truck_mask = np.zeros((self.num_terminal_trucks, total_positions, total_positions), dtype=np.int32)
        
        # Create Warp arrays on device
        crane_mask_device = wp.array(crane_mask, dtype=wp.int32, device=self.device)
        truck_mask_device = wp.array(truck_parking_mask, dtype=wp.int32, device=self.device)
        terminal_truck_mask_device = wp.array(terminal_truck_mask, dtype=wp.int32, device=self.device)
        
        # Launch kernels using the registered kernels
        wp.launch(
            kernel=_kernel_instances['crane_mask'],
            dim=[self.num_cranes, total_positions, total_positions],
            inputs=[
                self.terminal_state.crane_positions,
                self.terminal_state.crane_properties,
                self.terminal_state.yard_container_indices,
                self.terminal_state.stack_heights,
                self.terminal_state.rail_track_vehicles,
                self.terminal_state.parking_vehicles,
                self.terminal_state.container_positions,
                float(self.current_simulation_time),
                self.num_cranes,
                total_positions,
                crane_mask_device
            ]
        )
        
        wp.launch(
            kernel=_kernel_instances['truck_parking_mask'],
            dim=[self.max_vehicles, self.num_parking_spots],
            inputs=[
                self.terminal_state.parking_vehicles,
                self.terminal_state.vehicle_properties,
                self.max_vehicles,
                self.num_parking_spots,
                truck_mask_device
            ]
        )
        
        wp.launch(
            kernel=_kernel_instances['terminal_truck_mask'],
            dim=[self.num_terminal_trucks, total_positions, total_positions],
            inputs=[
                self.terminal_state.vehicle_positions,
                self.terminal_state.container_positions,
                self.num_terminal_trucks,
                total_positions,
                terminal_truck_mask_device
            ]
        )
        
        # Convert back to numpy arrays
        crane_mask = crane_mask_device.numpy()
        truck_parking_mask = truck_mask_device.numpy()
        terminal_truck_mask = terminal_truck_mask_device.numpy()
        
        # Combine into a dictionary
        action_mask = {
            'crane_movement': crane_mask,
            'truck_parking': truck_parking_mask,
            'terminal_truck': terminal_truck_mask
        }
        
        return action_mask

    @wp.kernel
    def _kernel_set_simulation_time(simulation_time: wp.array(dtype=wp.float32, ndim=1),
                                time_value: wp.float32):
        """Set the simulation time."""
        simulation_time[0] = time_value

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state and initialize optimizations.
        
        Args:
            seed: Random seed
            options: Dictionary of options including:
                - 'optimize': Whether to use optimizations (default: True)
                - 'precompute_events': Whether to precompute events (default: True)
                - 'precompute_movements': Whether to precompute movements (default: True)
                - 'precompute_stacking': Whether to precompute stacking (default: True)
                - 'save_tables': Whether to save precomputed tables (default: False)
        
        Returns:
            Tuple of (observation, info)
        """
        start_time = time.time()
        
        # Reset random seed
        if seed is not None:
            np.random.seed(seed)
            # Use kernel to set simulation param
            wp.launch(
                kernel=self._kernel_set_float_value,
                dim=1,
                inputs=[self.terminal_state.simulation_params, 2, float(seed)]
            )
        
        # Reset simulation time
        self.current_simulation_time = 0.0
        self.current_simulation_datetime = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
        # Use kernel to set simulation time
        wp.launch(
            kernel=self._kernel_set_float_value,
            dim=1,
            inputs=[self.terminal_state.simulation_time, 0, 0.0]
        )
        
        # Reset all terminal state arrays
        self._reset_terminal_state()
        
        # Reset tracking counters
        self.trucks_arrived = 0
        self.trains_arrived = 0
        self.containers_moved = 0
        
        # Reinitialize the simulation
        self._initialize_simulation()
        
        # Initialize optimizations if requested
        use_optimizations = True
        if options is not None and 'optimize' in options:
            use_optimizations = options['optimize']
        
        if use_optimizations:
            # Extract optimization options
            precompute_events = True
            precompute_movements = True
            precompute_stacking = True
            save_tables = False
            
            if options is not None:
                if 'precompute_events' in options:
                    precompute_events = options['precompute_events']
                if 'precompute_movements' in options:
                    precompute_movements = options['precompute_movements']
                if 'precompute_stacking' in options:
                    precompute_stacking = options['precompute_stacking']
                if 'save_tables' in options:
                    save_tables = options['save_tables']
            
            # Initialize optimizations
            self.initialize_optimizations(
                precompute_events=precompute_events,
                precompute_movements=precompute_movements,
                precompute_stacking=precompute_stacking,
                save_tables=save_tables
            )
            
            # Replace step with optimized step
            if not hasattr(self, 'original_step'):
                self.original_step = self.step
                self.step = self.optimized_step
        
        # Get initial observation
        observation = self._get_observation()
        
        # Reset performance tracking
        if self.log_performance:
            self.step_times = []
        
        # Create optimized step times array
        self.optimized_step_times = []
        
        # Track reset time
        reset_time = time.time() - start_time
        if self.log_performance:
            self.reset_times.append(reset_time)
        
        return observation, {}
    
    def step(self, action):
        """Take a step in the environment using the provided action."""
        start_time = time.time()
        
        # Extract action components
        action_type = action['action_type']
        
        # Execute the action based on type
        if action_type == 0:  # Crane movement
            reward = self._execute_crane_movement(action['crane_movement'])
        elif action_type == 1:  # Truck parking
            reward = self._execute_truck_parking(action['truck_parking'])
        elif action_type == 2:  # Terminal truck
            reward = self._execute_terminal_truck(action['terminal_truck'])
        else:
            # Invalid action type
            reward = 0.0
        
        # Advance simulation time if no action was taken
        if reward == 0:
            self._advance_time(300)  # 5 minutes
        
        # Update simulation time in terminal state using kernel
        wp.launch(
            kernel=self._kernel_set_simulation_time,
            dim=1,
            inputs=[
                self.terminal_state.simulation_time,
                self.current_simulation_time
            ]
        )
        
        # Process arrivals and departures for the current time
        self._process_vehicle_arrivals()
        self._process_vehicle_departures()
        
        # Check if episode is done
        terminated = False
        truncated = self.current_simulation_time >= self.max_simulation_time
        
        # Get new observation
        observation = self._get_observation()
        
        # Create info dictionary
        info = {
            'simulation_time': self.current_simulation_time,
            'simulation_datetime': self.current_simulation_datetime,
            'trucks_handled': self.trucks_arrived,
            'trains_handled': self.trains_arrived,
            'containers_moved': self.containers_moved
        }
        
        # Track step time
        step_time = time.time() - start_time
        if self.log_performance:
            self.step_times.append(step_time)
        
        return observation, reward, terminated, truncated, info
    
    def create_wait_action(self):
        """Create a valid wait action (no-op)."""
        return {
            'action_type': 0,  # Crane movement
            'crane_movement': np.array([0, 0, 0]),
            'truck_parking': np.array([0, 0]),
            'terminal_truck': np.array([0, 0, 0])
        }
    
    def _get_position_coordinates(self, position_str):
        """Get coordinates for a position string."""
        # Try to get from terminal state if available
        position_idx = self.position_to_idx.get(position_str, -1)
        
        if position_idx == -1:
            # Position not found, return origin
            return (0.0, 0.0)
        
        # Determine position type and get coordinates
        if position_str.startswith('t') and '_' in position_str:
            # Rail position
            track_id = position_str.split('_')[0]
            slot_num = int(position_str.split('_')[1])
            track_num = int(track_id[1:])
            
            # Get from rail positions if available
            if hasattr(self.terminal_state, 'rail_positions'):
                return tuple(self.terminal_state.rail_positions[track_num-1, slot_num-1].numpy())
            else:
                # Approximate positions
                return ((slot_num - 1) * 24.0, (track_num - 1) * 5.0)
                
        elif position_str.startswith('p_'):
            # Parking position
            spot_num = int(position_str.split('_')[1])
            
            # Get from parking positions if available
            if hasattr(self.terminal_state, 'parking_positions'):
                return tuple(self.terminal_state.parking_positions[spot_num-1].numpy())
            else:
                # Approximate positions
                return ((spot_num - 1) * 24.0, self.num_rail_tracks * 5.0 + 5.0)
                
        else:
            # Storage position
            row_letter = position_str[0]
            bay_num = int(position_str[1:])
            row_idx = ord(row_letter.upper()) - ord('A')
            
            # Get from storage positions if available
            if hasattr(self.terminal_state, 'storage_positions'):
                return tuple(self.terminal_state.storage_positions[row_idx, bay_num-1].numpy())
            else:
                # Approximate positions
                return ((bay_num - 1) * 12.0, self.num_rail_tracks * 5.0 + 10.0 + row_idx * 10.0)
    
    def _get_position_type_code(self, position_str):
        """Get numeric code for position type (0=rail, 1=truck, 2=storage)."""
        if position_str.startswith('t') and '_' in position_str:
            return 0  # Rail
        elif position_str.startswith('p_'):
            return 1  # Truck
        else:
            return 2  # Storage

    def render(self, mode='human'):
        """Render the environment."""
        if self.simplified_rendering:
            # Simplified rendering for faster training
            return None
        
        # Create figure for rendering
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw terminal layout
        self._draw_terminal_layout(ax)
        
        # Draw containers in storage yard
        self._draw_storage_yard(ax)
        
        # Draw vehicles (trucks and trains)
        self._draw_vehicles(ax)
        
        # Draw cranes
        self._draw_cranes(ax)
        
        # Add simulation info
        ax.set_title(f"Container Terminal Simulation - Time: {self.current_simulation_datetime.strftime('%H:%M:%S')}")
        ax.set_xlabel("Length (m)")
        ax.set_ylabel("Width (m)")
        
        # Set axis limits
        terminal_length = self.num_rail_slots_per_track * 24.0
        terminal_width = (self.num_rail_tracks * 5.0 + 
                         10.0 +  # Parking and driving lane
                         self.num_storage_rows * 10.0)
        ax.set_xlim(-10, terminal_length + 10)
        ax.set_ylim(-10, terminal_width + 10)
        
        plt.tight_layout()
        plt.show()
        plt.close()
    
    def _draw_terminal_layout(self, ax):
        """Draw the terminal layout."""
        # Draw rail tracks
        rail_y = 0
        for i in range(self.num_rail_tracks):
            rail_y = i * 5.0
            ax.add_patch(plt.Rectangle((0, rail_y - 1), self.num_rail_slots_per_track * 24.0, 2, 
                                       color='gray', alpha=0.5))
            ax.text(-5, rail_y, f"T{i+1}", fontsize=8, ha='right', va='center')
        
        # Draw parking area
        parking_y = self.num_rail_tracks * 5.0 + 5.0
        ax.add_patch(plt.Rectangle((0, parking_y - 2), self.num_rail_slots_per_track * 24.0, 4, 
                                   color='lightgray', alpha=0.3))
        ax.text(-5, parking_y, "Parking", fontsize=8, ha='right', va='center')
        
        # Draw driving lane
        lane_y = parking_y + 4
        ax.add_patch(plt.Rectangle((0, lane_y), self.num_rail_slots_per_track * 24.0, 6, 
                                   color='darkgray', alpha=0.2))
        ax.text(-5, lane_y + 3, "Lane", fontsize=8, ha='right', va='center')
        
        # Draw storage area
        storage_y = lane_y + 6
        for i in range(self.num_storage_rows):
            row_y = storage_y + i * 10.0
            row_letter = chr(65 + i)
            ax.text(-5, row_y + 5, row_letter, fontsize=8, ha='right', va='center')
            
            for j in range(self.num_storage_bays):
                bay_x = j * 12.0
                ax.add_patch(plt.Rectangle((bay_x, row_y), 10, 10, 
                                           fill=False, edgecolor='k', alpha=0.2))
                if j % 5 == 0:  # Label every 5 bays
                    ax.text(bay_x + 5, row_y - 2, str(j+1), fontsize=6, ha='center')
    
    def _draw_storage_yard(self, ax):
        """Draw containers in the storage yard."""
        # Get stack heights
        stack_heights = self.terminal_state.stack_heights.numpy()
        
        # Draw each container
        storage_y_base = (self.num_rail_tracks * 5.0 + 10.0 + 6.0)  # Base y-position for storage
        
        for row in range(self.num_storage_rows):
            row_y = storage_y_base + row * 10.0
            row_letter = chr(65 + row)
            
            for bay in range(self.num_storage_bays):
                bay_x = bay * 12.0
                height = stack_heights[row, bay]
                
                if height > 0:
                    # Draw containers in this stack
                    for tier in range(height):
                        container_idx = self.terminal_state.yard_container_indices[row, bay, tier].numpy()
                        if container_idx >= 0:
                            # Get container type for coloring
                            container_props = self.terminal_state.container_properties[container_idx].numpy()
                            container_type = int(container_props[0])
                            goods_type = int(container_props[1])
                            
                            # Choose color based on type
                            if goods_type == 1:  # Reefer
                                color = 'blue'
                            elif goods_type == 2:  # Dangerous
                                color = 'red'
                            else:  # Regular
                                color = ['lightblue', 'lightgreen', 'orange', 'yellow', 'purple', 'cyan'][container_type % 6]
                            
                            # Draw container
                            container_y = row_y + tier * 2  # Stack containers vertically
                            ax.add_patch(plt.Rectangle((bay_x + 1, container_y + 1), 8, 1.5, 
                                                      color=color, alpha=0.7))
                            
                            # Add container ID text (only for top container)
                            if tier == height - 1:
                                container_id = f"{row_letter}{bay+1}:{container_idx}"
                                ax.text(bay_x + 5, container_y + 2, container_id, fontsize=6, ha='center')
    
    def _draw_vehicles(self, ax):
        """Draw vehicles (trucks and trains)."""
        # Draw trucks in parking
        parking_y = self.num_rail_tracks * 5.0 + 5.0
        
        for spot in range(self.num_parking_spots):
            truck_idx = self.terminal_state.parking_vehicles[spot].numpy()
            if truck_idx >= 0:
                spot_x = spot * 24.0
                ax.add_patch(plt.Rectangle((spot_x + 2, parking_y - 1), 20, 2, 
                                          color='green', alpha=0.7))
                
                # Show container on truck if present
                if self.terminal_state.vehicle_container_counts[truck_idx] > 0:
                    container_idx = self.terminal_state.vehicle_containers[truck_idx, 0].numpy()
                    if container_idx >= 0:
                        ax.add_patch(plt.Rectangle((spot_x + 5, parking_y - 0.5), 14, 1, 
                                                  color='orange', alpha=0.7))
        
        # Draw trains on rail tracks
        for track in range(self.num_rail_tracks):
            train_idx = self.terminal_state.rail_track_vehicles[track].numpy()
            if train_idx >= 0:
                track_y = track * 5.0
                
                # Draw locomotive
                ax.add_patch(plt.Rectangle((0, track_y - 1), 10, 2, 
                                          color='brown', alpha=0.7))
                
                # Draw wagons with containers
                container_count = self.terminal_state.vehicle_container_counts[train_idx].numpy()
                for i in range(container_count):
                    wagon_x = 10 + i * 24.0
                    ax.add_patch(plt.Rectangle((wagon_x, track_y - 1), 20, 2, 
                                              color='gray', alpha=0.7))
                    
                    container_idx = self.terminal_state.vehicle_containers[train_idx, i].numpy()
                    if container_idx >= 0:
                        ax.add_patch(plt.Rectangle((wagon_x + 2, track_y - 0.5), 16, 1, 
                                                  color='blue', alpha=0.7))
    
    def _draw_cranes(self, ax):
        """Draw cranes at their current positions."""
        for i in range(self.num_cranes):
            crane_pos = self.terminal_state.crane_positions[i].numpy()
            x, y = crane_pos
            
            # Draw crane structure
            ax.add_patch(plt.Rectangle((x - 10, 0), 20, y + 10, 
                                      fill=False, edgecolor='red', linewidth=1.5))
            
            # Draw trolley
            ax.add_patch(plt.Rectangle((x - 2, y - 2), 4, 4, 
                                      color='red', alpha=0.7))
            
            # Check if crane is carrying a container
            container_idx = int(self.terminal_state.crane_properties[i, 1].numpy())
            if container_idx >= 0:
                ax.add_patch(plt.Rectangle((x - 4, y - 4), 8, 2, 
                                          color='yellow', alpha=0.7))
    
    def set_vehicle_limits(self, max_trucks=None, max_trains=None):
        """Set limits for vehicle arrivals per day."""
        if max_trucks is not None:
            self.max_trucks_per_day = max_trucks
        
        if max_trains is not None:
            self.max_trains_per_day = max_trains
    
    def set_simplified_rendering(self, simplified=True):
        """Set simplified rendering mode for faster training."""
        self.simplified_rendering = simplified
    
    def print_performance_stats(self):
        """Print performance statistics for the environment."""
        if not self.log_performance:
            print("Performance logging not enabled.")
            return
        
        print("\nEnvironment Performance Statistics:")
        
        if self.step_times:
            avg_step = sum(self.step_times) / len(self.step_times) * 1000  # Convert to ms
            min_step = min(self.step_times) * 1000
            max_step = max(self.step_times) * 1000
            print(f"  Step time: Avg: {avg_step:.2f}ms, Min: {min_step:.2f}ms, Max: {max_step:.2f}ms")
        
        if self.reset_times:
            avg_reset = sum(self.reset_times) / len(self.reset_times) * 1000  # Convert to ms
            print(f"  Reset time: {avg_reset:.2f}ms average")
        
        # Print component-specific statistics
        if hasattr(self.movement_calculator, 'print_performance_stats'):
            self.movement_calculator.print_performance_stats()
        
        if hasattr(self.storage_yard, 'print_performance_stats'):
            self.storage_yard.print_performance_stats()
        
        if hasattr(self.stacking_kernels, 'print_performance_stats'):
            self.stacking_kernels.print_performance_stats()
            
        # Print optimization statistics if available
        if hasattr(self, 'optimized_step_times') and self.optimized_step_times:
            self.print_optimization_stats()
    
    #
    # OPTIMIZATION METHODS - NEW CODE FOR MEMORY-PERFORMANCE TRADEOFF
    #
    
    def initialize_optimizations(self, 
                               precompute_events=True, 
                               precompute_movements=True,
                               precompute_stacking=True,
                               save_tables=False,
                               tables_path="./precomputed/"):
        """
        Initialize all optimization components to trade memory for speed.
        
        Args:
            precompute_events: Whether to precompute vehicle events
            precompute_movements: Whether to precompute movement times
            precompute_stacking: Whether to precompute stacking compatibility
            save_tables: Whether to save precomputed tables to disk
            tables_path: Path to save/load precomputed tables
        """
        optimization_start = time.time()
        
        # Store optimization flags
        self.use_precomputed_events = precompute_events
        self.use_precomputed_movements = precompute_movements
        self.use_precomputed_stacking = precompute_stacking
        
        # Ensure directory exists if saving tables
        if save_tables:
            import os
            os.makedirs(tables_path, exist_ok=True)
        
        # Track memory usage
        initial_memory = 0
        if self.device == 'cuda' and torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        
        if precompute_movements:
            print("Precomputing movement lookup tables...")
            self._initialize_movement_lookup_tables(save_tables, tables_path)
        
        if precompute_stacking:
            print("Precomputing stacking compatibility matrix...")
            self._initialize_stacking_compatibility(save_tables, tables_path)
        
        if precompute_events:
            print("Precomputing vehicle arrival events...")
            self._initialize_event_precomputation(save_tables, tables_path)
        
        # Calculate memory usage
        final_memory = 0
        if self.device == 'cuda' and torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            
        optimization_time = time.time() - optimization_start
        
        print(f"Optimizations initialized in {optimization_time:.2f} seconds")
        print(f"Memory usage: {final_memory - initial_memory:.2f} MB")

    def _initialize_movement_lookup_tables(self, save_tables=False, tables_path="./precomputed/"):
        """Initialize movement lookup tables."""
        movement_start = time.time()
        
        # Get position information
        num_positions = len(self.position_to_idx)
        
        # Define container types to precompute
        container_types = ["TEU", "FEU", "HQ", "Trailer", "Swap Body"]
        type_indices = {t: i for i, t in enumerate(container_types)}
        
        # Define stack heights to precompute (0-5 containers)
        max_stack_heights = 6
        
        # Try to load from file first
        table_path = os.path.join(tables_path, "movement_lookup.npy")
        if os.path.exists(table_path):
            try:
                print(f"Loading movement lookup tables from {table_path}...")
                self.movement_lookup_tables = np.load(table_path)
                self.movement_lookup_config = {
                    'position_to_idx': self.position_to_idx,
                    'container_types': container_types,
                    'type_indices': type_indices,
                    'max_stack_heights': max_stack_heights
                }
                
                # Convert to appropriate shape if needed
                if self.movement_lookup_tables.shape != (num_positions, num_positions, len(container_types), max_stack_heights):
                    print("Loaded table has incorrect shape, recomputing...")
                    self.movement_lookup_tables = None
            except Exception as e:
                print(f"Error loading movement tables: {e}")
                self.movement_lookup_tables = None
        
        # If not loaded, compute from scratch
        if not hasattr(self, 'movement_lookup_tables') or self.movement_lookup_tables is None:
            print("Computing movement lookup tables from scratch...")
            
            # Create numpy array for lookup tables - we use numpy arrays for better indexing
            self.movement_lookup_tables = np.zeros(
                (num_positions, num_positions, len(container_types), max_stack_heights), 
                dtype=np.float32
            )
            
            self.movement_lookup_config = {
                'position_to_idx': self.position_to_idx,
                'container_types': container_types,
                'type_indices': type_indices,
                'max_stack_heights': max_stack_heights
            }
            
            # Process positions in batches
            batch_size = 50  # Adjust based on memory constraints
            for src_start in range(0, num_positions, batch_size):
                src_end = min(src_start + batch_size, num_positions)
                print(f"Processing sources {src_start}-{src_end} of {num_positions}...")
                
                for src_idx in range(src_start, src_end):
                    src_pos = self.idx_to_position.get(src_idx)
                    if src_pos is None:
                        continue
                    
                    # Get position type for source
                    src_type = self._get_position_type_code(src_pos)
                    
                    for dst_idx in range(num_positions):
                        dst_pos = self.idx_to_position.get(dst_idx)
                        if dst_pos is None or src_pos == dst_pos:
                            continue
                        
                        # Get position type for destination
                        dst_type = self._get_position_type_code(dst_pos)
                        
                        # Compute for each container type and stack height
                        for type_idx, container_type in enumerate(container_types):
                            for stack_idx in range(max_stack_heights):
                                # Calculate stack height in meters
                                stack_height = stack_idx * 2.59  # Standard container height
                                
                                try:
                                    # Use existing movement calculator
                                    movement_time = self.movement_calculator.calculate_movement_time(
                                        src_pos, dst_pos, 0, 
                                        container_type=type_idx,  # Pass type index directly
                                        stack_height=stack_height
                                    )
                                    
                                    # Store in numpy array
                                    self.movement_lookup_tables[src_idx, dst_idx, type_idx, stack_idx] = movement_time
                                except Exception as e:
                                    # Use a default time if calculation fails
                                    self.movement_lookup_tables[src_idx, dst_idx, type_idx, stack_idx] = 100.0
            
            # Save to file if requested
            if save_tables:
                try:
                    print(f"Saving movement lookup tables to {table_path}...")
                    np.save(table_path, self.movement_lookup_tables)
                except Exception as e:
                    print(f"Error saving movement tables: {e}")
        
        movement_time = time.time() - movement_start
        print(f"Movement lookup tables initialized in {movement_time:.2f} seconds")
        print(f"Table shape: {self.movement_lookup_tables.shape}, Memory: {self.movement_lookup_tables.nbytes / (1024*1024):.2f} MB")
        
        # Set up movement calculator to use lookup tables
        if hasattr(self.movement_calculator, 'use_lookup_tables'):
            self.movement_calculator.use_lookup_tables(
                self.movement_lookup_tables,
                container_types,
                max_stack_heights
            )

    def _initialize_stacking_compatibility(self, save_tables=False, tables_path="./precomputed/"):
        """Initialize stacking compatibility matrix."""
        stacking_start = time.time()
        
        # Calculate matrix size
        max_containers = self.terminal_state.max_containers
        
        # Try to load from file first
        matrix_path = os.path.join(tables_path, "stacking_matrix.npy")
        if os.path.exists(matrix_path):
            try:
                print(f"Loading stacking compatibility matrix from {matrix_path}...")
                self.stacking_compatibility_matrix = np.load(matrix_path)
                
                # Verify shape
                if self.stacking_compatibility_matrix.shape != (max_containers, max_containers):
                    print("Loaded matrix has incorrect shape, recomputing...")
                    self.stacking_compatibility_matrix = None
            except Exception as e:
                print(f"Error loading stacking matrix: {e}")
                self.stacking_compatibility_matrix = None
        
        # If not loaded, compute from scratch
        if not hasattr(self, 'stacking_compatibility_matrix') or self.stacking_compatibility_matrix is None:
            print("Computing stacking compatibility matrix from scratch...")
            
            # Create numpy array for compatibility matrix
            self.stacking_compatibility_matrix = np.zeros((max_containers, max_containers), dtype=np.int32)
            
            # Precompute for each container pair
            for upper_idx in range(max_containers):
                # Skip inactive containers
                container_props = self.terminal_state.container_properties.numpy()
                if container_props[upper_idx, 6] <= 0:  # Check active flag
                    continue
                    
                for lower_idx in range(max_containers):
                    # Skip inactive containers and self-stacking
                    if container_props[lower_idx, 6] <= 0 or upper_idx == lower_idx:
                        continue
                    
                    # Get container properties
                    upper_type = int(container_props[upper_idx, 0])
                    upper_goods = int(container_props[upper_idx, 1])
                    upper_weight = float(container_props[upper_idx, 3])
                    upper_compatibility = int(container_props[upper_idx, 5])
                    
                    lower_type = int(container_props[lower_idx, 0])
                    lower_goods = int(container_props[lower_idx, 1])
                    lower_weight = float(container_props[lower_idx, 3])
                    lower_stackable = bool(container_props[lower_idx, 4])
                    lower_compatibility = int(container_props[lower_idx, 5])
                    
                    # Check stacking rules
                    
                    # Can't stack on non-stackable container
                    if not lower_stackable:
                        continue
                    
                    # Check compatibility
                    can_stack = True
                    
                    # None compatibility - can't stack
                    if lower_compatibility == 0 or upper_compatibility == 0:
                        can_stack = False
                    
                    # Self compatibility - must be same type and goods
                    elif lower_compatibility == 1 or upper_compatibility == 1:
                        if lower_type != upper_type or lower_goods != upper_goods:
                            can_stack = False
                    
                    # Size compatibility - must be same size
                    elif lower_compatibility == 2 or upper_compatibility == 2:
                        if lower_type != upper_type:
                            can_stack = False
                    
                    # Weight constraint - container above should be lighter
                    if upper_weight > lower_weight:
                        can_stack = False
                    
                    # Update matrix
                    if can_stack:
                        self.stacking_compatibility_matrix[upper_idx, lower_idx] = 1
            
            # Save to file if requested
            if save_tables:
                try:
                    print(f"Saving stacking compatibility matrix to {matrix_path}...")
                    np.save(matrix_path, self.stacking_compatibility_matrix)
                except Exception as e:
                    print(f"Error saving stacking matrix: {e}")
        
        stacking_time = time.time() - stacking_start
        print(f"Stacking compatibility matrix initialized in {stacking_time:.2f} seconds")
        print(f"Matrix shape: {self.stacking_compatibility_matrix.shape}, Memory: {self.stacking_compatibility_matrix.nbytes / (1024*1024):.2f} MB")
        
        # Set up stacking kernels to use compatibility matrix
        if hasattr(self.stacking_kernels, 'use_compatibility_matrix'):
            self.stacking_kernels.use_compatibility_matrix(self.stacking_compatibility_matrix)

    def _initialize_event_precomputation(self, save_tables=False, tables_path="./precomputed/"):
        """Initialize event precomputation."""
        event_start = time.time()
        
        # Calculate maximum events
        simulation_days = self.max_simulation_time / 86400
        events_per_day = (self.max_trucks_per_day + self.max_trains_per_day) * 2  # Arrive & depart
        max_events = int(simulation_days * events_per_day)
        
        # Try to load from file first
        events_path = os.path.join(tables_path, "precomputed_events.npy")
        if os.path.exists(events_path):
            try:
                print(f"Loading precomputed events from {events_path}...")
                events_data = np.load(events_path, allow_pickle=True).item()
                
                self.precomputed_events = events_data['events']
                self.event_count = events_data['count']
                self.current_event_idx = 0
                
                # Verify configuration matches
                if (events_data['max_simulation_time'] != self.max_simulation_time or
                    events_data['max_trucks_per_day'] != self.max_trucks_per_day or
                    events_data['max_trains_per_day'] != self.max_trains_per_day):
                    print("Loaded events have different configuration, recomputing...")
                    self.precomputed_events = None
            except Exception as e:
                print(f"Error loading precomputed events: {e}")
                self.precomputed_events = None
        
        # If not loaded, compute from scratch
        if not hasattr(self, 'precomputed_events') or self.precomputed_events is None:
            print("Computing events from scratch...")
            
            # Create numpy array for events [time, type, data]
            self.precomputed_events = np.zeros((max_events, 3), dtype=np.float32)
            self.event_count = 0
            self.current_event_idx = 0
            
            # Generate all vehicle arrivals for entire simulation
            for day in range(int(simulation_days)):
                day_start = day * 86400
                
                # Generate truck arrivals
                num_trucks = np.random.randint(5, self.max_trucks_per_day + 1)
                for i in range(num_trucks):
                    arrival_time = day_start + np.random.uniform(0, 86400)
                    self._add_precomputed_event(arrival_time, 0, i)  # 0 = truck arrival
                
                # Generate train arrivals
                num_trains = np.random.randint(1, self.max_trains_per_day + 1)
                for i in range(num_trains):
                    arrival_time = day_start + np.random.uniform(0, 86400)
                    self._add_precomputed_event(arrival_time, 1, i)  # 1 = train arrival
            
            # Sort events by time
            event_indices = np.argsort(self.precomputed_events[:self.event_count, 0])
            self.precomputed_events[:self.event_count] = self.precomputed_events[event_indices]
            
            # Save to file if requested
            if save_tables:
                try:
                    print(f"Saving precomputed events to {events_path}...")
                    events_data = {
                        'events': self.precomputed_events,
                        'count': self.event_count,
                        'max_simulation_time': self.max_simulation_time,
                        'max_trucks_per_day': self.max_trucks_per_day,
                        'max_trains_per_day': self.max_trains_per_day
                    }
                    np.save(events_path, events_data)
                except Exception as e:
                    print(f"Error saving precomputed events: {e}")
        
        event_time = time.time() - event_start
        print(f"Event precomputation initialized in {event_time:.2f} seconds")
        print(f"Precomputed {self.event_count} events, Memory: {self.precomputed_events.nbytes / (1024*1024):.2f} MB")

    def _add_precomputed_event(self, time, event_type, event_data):
        """Add an event to the precomputed events array."""
        if not hasattr(self, 'event_count') or self.event_count >= len(self.precomputed_events):
            return False
        
        # Add event to array
        self.precomputed_events[self.event_count, 0] = time
        self.precomputed_events[self.event_count, 1] = event_type
        self.precomputed_events[self.event_count, 2] = event_data
        
        self.event_count += 1
        return True

    def get_movement_time(self, src_pos, dst_pos, container_type_idx=2, stack_height=0.0):
        """
        Get movement time from lookup table if available.
        
        Args:
            src_pos: Source position string
            dst_pos: Destination position string
            container_type_idx: Container type index (0-4)
            stack_height: Height of container stack
            
        Returns:
            Movement time in seconds
        """
        if not hasattr(self, 'use_precomputed_movements') or not self.use_precomputed_movements:
            # Use original calculation
            return self.movement_calculator.calculate_movement_time(
                src_pos, dst_pos, 0, container_type=container_type_idx, stack_height=stack_height
            )
        
        if not hasattr(self, 'movement_lookup_tables') or self.movement_lookup_tables is None:
            # Lookup tables not initialized
            return self.movement_calculator.calculate_movement_time(
                src_pos, dst_pos, 0, container_type=container_type_idx, stack_height=stack_height
            )
        
        # Get indices
        src_idx = self.position_to_idx.get(src_pos, -1)
        dst_idx = self.position_to_idx.get(dst_pos, -1)
        
        if src_idx == -1 or dst_idx == -1:
            # Invalid positions
            return self.movement_calculator.calculate_movement_time(
                src_pos, dst_pos, 0, container_type=container_type_idx, stack_height=stack_height
            )
        
        # Convert stack height to index
        container_height = 2.59  # Standard container height
        stack_idx = min(int(stack_height / container_height), self.movement_lookup_config['max_stack_heights'] - 1)
        
        # Convert container type string to index if needed
        if isinstance(container_type_idx, str):
            type_idx = self.movement_lookup_config['type_indices'].get(container_type_idx, 0)
        else:
            type_idx = container_type_idx
        
        # Get time from lookup table
        return self.movement_lookup_tables[src_idx, dst_idx, type_idx, stack_idx]

    def can_stack(self, upper_container_idx, lower_container_idx):
        """
        Check if upper container can be stacked on lower container.
        
        Args:
            upper_container_idx: Index of container to be placed on top
            lower_container_idx: Index of container at the bottom
            
        Returns:
            True if stacking is valid, False otherwise
        """
        if not hasattr(self, 'use_precomputed_stacking') or not self.use_precomputed_stacking:
            # Use original validation
            return self.stacking_kernels.can_place_at(upper_container_idx, lower_container_idx)
        
        if not hasattr(self, 'stacking_compatibility_matrix') or self.stacking_compatibility_matrix is None:
            # Matrix not initialized
            return self.stacking_kernels.can_place_at(upper_container_idx, lower_container_idx)
        
        # Validate indices
        if (upper_container_idx < 0 or upper_container_idx >= len(self.stacking_compatibility_matrix) or
            lower_container_idx < 0 or lower_container_idx >= len(self.stacking_compatibility_matrix)):
            return False
        
        # Check compatibility
        return bool(self.stacking_compatibility_matrix[upper_container_idx, lower_container_idx])

    def fast_forward_to_next_event(self):
        """
        Fast forward to the next vehicle arrival event.
        
        Returns:
            Time advanced in seconds
        """
        if not hasattr(self, 'use_precomputed_events') or not self.use_precomputed_events:
            # Use standard time advance
            self._advance_time(300)  # 5 minutes
            return 300.0
        
        if not hasattr(self, 'precomputed_events') or self.precomputed_events is None:
            # Events not precomputed
            self._advance_time(300)  # 5 minutes
            return 300.0
        
        # Get current time
        current_time = self.current_simulation_time
        
        # Find next event
        next_event_idx = -1
        for i in range(self.event_count):
            if self.precomputed_events[i, 0] > current_time:
                next_event_idx = i
                break
        
        if next_event_idx == -1:
            # No more events
            self._advance_time(300)  # 5 minutes
            return 300.0
        
        # Get event info
        next_time = self.precomputed_events[next_event_idx, 0]
        event_type = int(self.precomputed_events[next_event_idx, 1])
        event_data = int(self.precomputed_events[next_event_idx, 2])
        
        # Calculate time delta
        time_delta = next_time - current_time
        
        # Don't advance more than 30 minutes at once
        max_advance = 1800.0  # 30 minutes
        if time_delta > max_advance:
            self._advance_time(max_advance)
            return max_advance
        
        # Update simulation time
        self._advance_time(time_delta)
        
        # Process the event
        if event_type == 0:  # Truck arrival
            self._create_truck_arrival()
        elif event_type == 1:  # Train arrival
            self._create_train_arrival()
        
        return time_delta

    def optimized_step(self, action):
        """
        Optimized step method using precomputed data.
        
        Args:
            action: Action dictionary
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Start timing
        step_start = time.time()
        
        # Execute action
        action_type = action['action_type']
        
        if action_type == 0:  # Crane movement
            reward = self._execute_optimized_crane_movement(action['crane_movement'])
        elif action_type == 1:  # Truck parking
            reward = self._execute_truck_parking(action['truck_parking'])
        elif action_type == 2:  # Terminal truck
            reward = self._execute_terminal_truck(action['terminal_truck'])
        else:
            # Invalid action type
            reward = 0.0
        
        # If no action was taken, advance simulation time
        if reward == 0:
            if hasattr(self, 'use_precomputed_events') and self.use_precomputed_events:
                # Use optimized time advancement
                self.fast_forward_to_next_event()
            else:
                # Use standard time advancement
                self._advance_time(300)  # 5 minutes
        
        # Update simulation time in terminal state
        current_time = self.current_simulation_time
        wp.launch(
            kernel=self._kernel_set_simulation_time,
            dim=1,
            inputs=[
                self.terminal_state.simulation_time,
                float(current_time)
            ]
        )
        
        # Process vehicle arrivals and departures
        if not hasattr(self, 'use_precomputed_events') or not self.use_precomputed_events:
            # Use regular processing
            self._process_vehicle_arrivals()
            self._process_vehicle_departures()
        
        # Check if episode is done
        terminated = False
        truncated = current_time >= self.max_simulation_time
        
        # Get new observation
        observation = self._get_observation()
        
        # Create info dictionary with performance metrics
        step_time = time.time() - step_start
        info = {
            'simulation_time': current_time,
            'simulation_datetime': self.current_simulation_datetime,
            'trucks_handled': self.trucks_arrived,
            'trains_handled': self.trains_arrived,
            'containers_moved': self.containers_moved,
            'step_compute_time': step_time
        }
        
        # Track step time
        if hasattr(self, 'optimized_step_times'):
            self.optimized_step_times.append(step_time)
        else:
            self.optimized_step_times = [step_time]
        
        return observation, reward, terminated, truncated, info

    def _execute_optimized_crane_movement(self, crane_action):
        """
        Execute a crane movement with optimized lookups.
        
        Args:
            crane_action: Crane movement action [crane_idx, src_idx, dst_idx]
            
        Returns:
            Reward for the action
        """
        crane_idx, src_idx, dst_idx = crane_action
        
        # Convert indices to position strings
        src_pos = self.idx_to_position.get(src_idx, None)
        dst_pos = self.idx_to_position.get(dst_idx, None)
        
        if src_pos is None or dst_pos is None:
            # Invalid position indices
            return 0.0
        
        # Check if crane is available
        crane_props_np = self.terminal_state.crane_properties.numpy()
        crane_available_time = float(crane_props_np[crane_idx, 2])
        if crane_available_time > self.current_simulation_time:
            return 0.0
        
        # Get container at source position
        container_idx = -1
        if src_pos[0].isalpha() and src_pos[0].upper() in self.storage_yard.row_names:
            # Storage position
            container_idx, _ = self.storage_yard.get_top_container(src_pos)
            if container_idx is None:
                container_idx = -1
        else:
            # Rail or parking position
            # This would normally check vehicles at these positions
            pass
        
        if container_idx < 0:
            # No container at source position
            return 0.0
        
        # Check if container can be placed at destination
        can_place = False
        if dst_pos[0].isalpha() and dst_pos[0].upper() in self.storage_yard.row_names:
            # Storage position - use optimized stacking check if available
            if hasattr(self, 'use_precomputed_stacking') and self.use_precomputed_stacking:
                row, bay = self._parse_position(dst_pos)
                if row is not None and bay is not None:
                    stack_heights_np = self.terminal_state.stack_heights.numpy()
                    height = int(stack_heights_np[row, bay])
                    
                    if height > 0:
                        # Get top container in destination stack
                        yard_indices_np = self.terminal_state.yard_container_indices.numpy()
                        top_container_idx = int(yard_indices_np[row, bay, height - 1])
                        
                        if top_container_idx >= 0:
                            # Check if can stack using optimized lookup
                            can_place = self.can_stack(container_idx, top_container_idx)
                        else:
                            can_place = True  # No container in stack
                    else:
                        can_place = True  # Empty stack
            else:
                # Use original validation
                can_place = self.storage_yard.can_accept_container(dst_pos, container_idx)
        else:
            # Rail or parking position
            can_place = True  # Assume we can place
        
        if not can_place:
            # Cannot place container at destination
            return 0.0
        
        # Get container type for movement time calculation
        container_props_np = self.terminal_state.container_properties.numpy()
        container_type_idx = int(container_props_np[container_idx, 0])
        
        # Get stack height if needed
        stack_height = 0.0
        if dst_pos[0].isalpha() and dst_pos[0].upper() in self.storage_yard.row_names:
            row, bay = self._parse_position(dst_pos)
            if row is not None and bay is not None:
                stack_heights_np = self.terminal_state.stack_heights.numpy()
                stack_height = float(stack_heights_np[row, bay])
        
        # Get movement time using optimized lookup
        movement_time = self.get_movement_time(src_pos, dst_pos, container_type_idx, stack_height)
        
        # Update crane position and time
        wp.launch(
            kernel=self._kernel_set_element_crane_properties,
            dim=1,
            inputs=[
                self.terminal_state.crane_properties,
                crane_idx,
                2,  # Index for available time
                float(self.current_simulation_time + movement_time)
            ]
        )
        
        # Update crane position for visualization
        dst_coords = self._get_position_coordinates(dst_pos)
        wp.launch(
            kernel=self._kernel_set_element_crane_properties,
            dim=1,
            inputs=[
                self.terminal_state.crane_positions,
                crane_idx,
                0,  # X coordinate
                float(dst_coords[0])
            ]
        )
        
        wp.launch(
            kernel=self._kernel_set_element_crane_properties,
            dim=1,
            inputs=[
                self.terminal_state.crane_positions,
                crane_idx,
                1,  # Y coordinate
                float(dst_coords[1])
            ]
        )
        
        # Remove container from source
        if src_pos[0].isalpha() and src_pos[0].upper() in self.storage_yard.row_names:
            # Storage position
            self.storage_yard.remove_container(src_pos)
        
        # Place container at destination
        if dst_pos[0].isalpha() and dst_pos[0].upper() in self.storage_yard.row_names:
            # Storage position
            self.storage_yard.add_container(container_idx, dst_pos)
        
        # Update container position
        dst_position_idx = self.position_to_idx.get(dst_pos, -1)
        wp.launch(
            kernel=self._kernel_update_container_position,
            dim=1,
            inputs=[self.terminal_state.container_positions, container_idx, dst_position_idx]
        )
        
        # Increment containers moved
        self.containers_moved += 1
        
        # Reward proportional to movement efficiency (inverse of time)
        reward = 10.0 / (1.0 + movement_time / 60.0)  # Normalize to ~0-10 range
        
        return reward

    def print_optimization_stats(self):
        """Print statistics about optimization performance."""
        print("\nOptimization Performance Statistics:")
        
        # Memory usage
        memory_usage = 0.0
        
        if hasattr(self, 'movement_lookup_tables') and self.movement_lookup_tables is not None:
            movement_memory = self.movement_lookup_tables.nbytes / (1024 * 1024)  # MB
            memory_usage += movement_memory
            print(f"  Movement lookup tables: {movement_memory:.2f} MB")
        
        if hasattr(self, 'stacking_compatibility_matrix') and self.stacking_compatibility_matrix is not None:
            stacking_memory = self.stacking_compatibility_matrix.nbytes / (1024 * 1024)  # MB
            memory_usage += stacking_memory
            print(f"  Stacking compatibility matrix: {stacking_memory:.2f} MB")
        
        if hasattr(self, 'precomputed_events') and self.precomputed_events is not None:
            event_memory = self.precomputed_events.nbytes / (1024 * 1024)  # MB
            memory_usage += event_memory
            print(f"  Precomputed events: {event_memory:.2f} MB")
        
        print(f"  Total memory usage: {memory_usage:.2f} MB")
        
        # Step time performance
        if hasattr(self, 'step_times') and self.step_times and hasattr(self, 'optimized_step_times') and self.optimized_step_times:
            original_avg = sum(self.step_times) / len(self.step_times) * 1000  # ms
            optimized_avg = sum(self.optimized_step_times) / len(self.optimized_step_times) * 1000  # ms
            
            print(f"\nStep time performance:")
            print(f"  Original implementation: {original_avg:.2f}ms average")
            print(f"  Optimized implementation: {optimized_avg:.2f}ms average")
            
            if optimized_avg > 0:
                speedup = original_avg / optimized_avg
                print(f"  Speedup: {speedup:.2f}x faster")
        
        # Container movement stats
        print(f"\nContainer movements: {self.containers_moved}")


class QueueWrapper:
    """Wrapper for queue operations to simplify interface."""
    
    def __init__(self, max_size=100):
        """Initialize empty queue with max size."""
        self.items = []
        self.max_size = max_size
    
    def add(self, item):
        """Add item to queue."""
        if len(self.items) < self.max_size:
            self.items.append(item)
            return True
        return False
    
    def get(self):
        """Get and remove first item from queue."""
        if self.items:
            return self.items.pop(0)
        return None
    
    def size(self):
        """Return current queue size."""
        return len(self.items)
    
    def is_empty(self):
        """Check if queue is empty."""
        return len(self.items) == 0
    
    def clear(self):
        """Clear the queue."""
        self.items = []