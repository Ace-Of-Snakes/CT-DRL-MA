import numpy as np
import warp as wp
from datetime import datetime, timedelta
import time
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional, Any, Union

# DO NOT import warp at module level

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
    
    # def _register_kernels(self):
    #     """Register Warp kernels for environment operations."""
    #     # Register kernels for generating action masks
    #     wp.register_kernel(self._kernel_generate_crane_mask)
    #     wp.register_kernel(self._kernel_generate_truck_parking_mask)
    #     wp.register_kernel(self._kernel_generate_terminal_truck_mask)
    
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
    # Add these kernels to WarpTerminalEnvironment class

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

    # 3. Fix the _create_truck_arrival function
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
    
    @wp.kernel
    def _kernel_set_vehicle_container(vehicle_containers: wp.array(dtype=wp.int32, ndim=2),
                                    vehicle_idx: wp.int32,
                                    container_slot: wp.int32,
                                    container_idx: wp.int32):
        """Set a container on a vehicle in a specific slot."""
        vehicle_containers[vehicle_idx, container_slot] = container_idx

    @wp.kernel
    def _kernel_set_container_vehicle(container_vehicles: wp.array(dtype=wp.int32, ndim=1),
                                container_idx: wp.int32,
                                vehicle_idx: wp.int32):
        """Set which vehicle a container is on."""
        container_vehicles[container_idx] = vehicle_idx


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
                kernel=self._kernel_set_vehicle_container,
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
                kernel=self._kernel_set_container_vehicle,
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
    
# 1. First, add these essential kernel functions to properly handle array operations

    @wp.kernel
    def _kernel_set_element_crane_properties(properties: wp.array(dtype=wp.float32, ndim=2),
                                        crane_idx: wp.int32,
                                        prop_idx: wp.int32,
                                        value: wp.float32):
        """Set a value in crane properties array."""
        properties[crane_idx, prop_idx] = value

    @wp.kernel
    def _kernel_set_element_container_vehicle(container_vehicles: wp.array(dtype=wp.int32, ndim=1),
                                        container_idx: wp.int32,
                                        vehicle_idx: wp.int32):
        """Set vehicle index for a container."""
        container_vehicles[container_idx] = vehicle_idx

    @wp.kernel
    def _kernel_set_element_vehicle_container(vehicle_containers: wp.array(dtype=wp.int32, ndim=2),
                                        vehicle_idx: wp.int32,
                                        slot_idx: wp.int32,
                                        container_idx: wp.int32):
        """Set container index for a vehicle slot."""
        vehicle_containers[vehicle_idx, slot_idx] = container_idx

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
    def _kernel_set_simulation_time(simulation_time: wp.array(dtype=wp.float32, ndim=1),
                                time_value: wp.float32):
        """Set the simulation time."""
        simulation_time[0] = time_value

    @wp.kernel
    def _kernel_get_queue_size(queue_sizes: wp.array(dtype=wp.int32, ndim=1),
                            queue_idx: wp.int32,
                            result: wp.array(dtype=wp.int32, ndim=1)):
        """Get queue size and store in result array."""
        result[0] = queue_sizes[queue_idx]

# 2. Fix the _execute_crane_movement function
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
                # src_idx = wp.tid()[1]
                # dst_idx = wp.tid()[2]
                
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
                # spot_idx = wp.tid(1)
                
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
                # src_idx = wp.tid(1)
                # dst_idx = wp.tid(2)
                
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
    def _kernel_set_float_value(arr: wp.array(dtype=wp.float32, ndim=1), 
                            idx: wp.int32, 
                            value: wp.float32):
        """Set a value in a float array at the specified index."""
        arr[idx] = value

    @wp.kernel
    def _kernel_set_int_value(arr: wp.array(dtype=wp.int32, ndim=1), 
                            idx: wp.int32, 
                            value: wp.int32):
        """Set a value in an int array at the specified index."""
        arr[idx] = value

    @wp.kernel
    def _kernel_add_event_to_queue(event_queue: wp.array(dtype=wp.float32, ndim=2),
                                queue_idx: wp.int32,
                                time: wp.float32,
                                event_type: wp.int32,
                                event_data: wp.int32):
        """Add event to the event queue."""
        event_queue[queue_idx, 0] = time
        event_queue[queue_idx, 1] = float(event_type)
        event_queue[queue_idx, 2] = float(event_data)

    @wp.kernel
    def _kernel_update_queue_size(queue_size: wp.array(dtype=wp.int32, ndim=1),
                            new_size: wp.int32):
        """Update the event queue size."""
        queue_size[0] = new_size

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
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
        
        # Get initial observation
        observation = self._get_observation()
        
        # Reset performance tracking
        if self.log_performance:
            self.step_times = []
        
        # Track reset time
        reset_time = time.time() - start_time
        if self.log_performance:
            self.reset_times.append(reset_time)
        
        return observation, {}
    
    # 4. Fix the step method to correctly update simulation time
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