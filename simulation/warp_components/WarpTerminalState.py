import warp as wp
import numpy as np
from typing import Dict, Tuple, List, Optional

class WarpTerminalState:
    """
    Core state representation for container terminal simulation using NVIDIA Warp.
    Manages all terminal data structures as warp arrays for efficient GPU simulation.
    """
    def __init__(self, 
             num_rail_tracks: int = 6,
             num_rail_slots_per_track: int = 29,
             num_storage_rows: int = 5,
             num_storage_bays: int = 58,
             max_stack_height: int = 5,
             num_parking_spots: int = 29,
             max_containers: int = 10000,
             max_vehicles: int = 100,
             num_cranes: int = 2,
             device: str = None):
        """
        Initialize the warp-accelerated terminal state.
        
        Args:
            num_rail_tracks: Number of rail tracks in the terminal
            num_rail_slots_per_track: Number of slots per rail track
            num_storage_rows: Number of rows in storage yard
            num_storage_bays: Number of bays per row in storage yard
            max_stack_height: Maximum height of container stacks
            num_parking_spots: Number of truck parking spots
            max_containers: Maximum number of containers that can be in the system
            max_vehicles: Maximum number of vehicles that can be in the system
            num_cranes: Number of RMG cranes
            device: Computation device (if None, will use CUDA if available)
        """
        # Set device
        self.device = device if device else ("cuda" if wp.get_cuda_device_count() > 0 else "cpu")
        print(f"Initializing WarpTerminalState on device: {self.device}")
        
        # Store configuration
        self.num_rail_tracks = num_rail_tracks
        self.num_rail_slots_per_track = num_rail_slots_per_track
        self.num_storage_rows = num_storage_rows
        self.num_storage_bays = num_storage_bays
        self.max_stack_height = max_stack_height
        self.num_parking_spots = num_parking_spots
        self.max_containers = max_containers
        self.max_vehicles = max_vehicles
        self.num_cranes = num_cranes
        
        # Generate row names (A, B, C, ...)
        self.row_names = [chr(65 + i) for i in range(num_storage_rows)]
        
        # Initialize terminal layout arrays
        self._initialize_layout_arrays()
        
        # Initialize container arrays
        self._initialize_container_arrays()
        
        # Initialize vehicle arrays
        self._initialize_vehicle_arrays()
        
        # Initialize crane arrays
        self._initialize_crane_arrays()
        
        # Initialize simulation state arrays
        self._initialize_simulation_state()
        
        # Create position mapping for faster lookups
        self._create_position_mapping()
        
        # Register warp kernels
        # self._register_kernels()

    def _initialize_layout_arrays(self):
        """Initialize arrays for terminal layout positions."""
        # Create position arrays for various terminal elements
        # Each position is an (x, y) coordinate in meters
        
        # Rail positions [track, slot, 2]
        self.rail_positions = wp.zeros(
            shape=(self.num_rail_tracks, self.num_rail_slots_per_track, 2), 
            dtype=wp.float32, 
            device=self.device
        )
        
        # Parking positions [spot, 2]
        self.parking_positions = wp.zeros(
            shape=(self.num_parking_spots, 2),
            dtype=wp.float32,
            device=self.device
        )
        
        # Storage positions [row, bay, 2]
        self.storage_positions = wp.zeros(
            shape=(self.num_storage_rows, self.num_storage_bays, 2),
            dtype=wp.float32,
            device=self.device
        )
        
        # Special area masks for storage
        # Each mask indicates which positions are part of a special area
        self.special_area_masks = {
            'reefer': wp.zeros(
                shape=(self.num_storage_rows, self.num_storage_bays),
                dtype=wp.bool,
                device=self.device
            ),
            'dangerous': wp.zeros(
                shape=(self.num_storage_rows, self.num_storage_bays),
                dtype=wp.bool,
                device=self.device
            ),
            'trailer': wp.zeros(
                shape=(self.num_storage_rows, self.num_storage_bays),
                dtype=wp.bool,
                device=self.device
            ),
            'swap_body': wp.zeros(
                shape=(self.num_storage_rows, self.num_storage_bays),
                dtype=wp.bool,
                device=self.device
            )
        }
        
        # Distance matrix for efficient lookup
        total_positions = (self.num_rail_tracks * self.num_rail_slots_per_track + 
                        self.num_parking_spots + 
                        self.num_storage_rows * self.num_storage_bays)
        
        self.distance_matrix = wp.zeros(
            shape=(total_positions, total_positions),
            dtype=wp.float32,
            device=self.device
        )

    def _initialize_container_arrays(self):
        """Initialize arrays for managing containers."""
        # Container registry - stores all container data
        # We use a structure-of-arrays approach for better GPU performance
        
        # Container indices (sparse mapping of container IDs to indices)
        self.container_indices = {}  # Dictionary for CPU-side lookup
        
        # Container positions (container_idx -> position_idx)
        # -1 means the container is not currently placed
        self.container_positions = wp.zeros(
            shape=(self.max_containers,),
            dtype=wp.int32,
            device=self.device
        )
        self.container_positions.fill_(-1)  # Initialize to "not placed"
        
        # Which vehicle the container is on (if any)
        # -1 means not on any vehicle
        self.container_vehicles = wp.zeros(
            shape=(self.max_containers,),
            dtype=wp.int32,
            device=self.device
        )
        self.container_vehicles.fill_(-1)  # Initialize to "not on vehicle"
        
        # Container properties - packed efficiently for GPU
        # Properties:
        # [0]: Type code (0=TEU, 1=FEU, 2=HQ, 3=Trailer, 4=Swap Body)
        # [1]: Goods type (0=Regular, 1=Reefer, 2=Dangerous)
        # [2]: Priority (0-100)
        # [3]: Weight (kg)
        # [4]: Is stackable (0/1)
        # [5]: Stack compatibility (0=none, 1=self, 2=size)
        # [6]: Active flag (0/1) - 1 if the container is active in the simulation
        # [7]: Departure time (seconds from simulation start)
        self.container_properties = wp.zeros(
            shape=(self.max_containers, 8),
            dtype=wp.float32,
            device=self.device
        )
        
        # Container dimensions [length, width, height]
        self.container_dimensions = wp.zeros(
            shape=(self.max_containers, 3),
            dtype=wp.float32,
            device=self.device
        )
        
        # Storage yard state - direct representation of stacks
        # Format: [row, bay, tier]
        # Value: container index (or -1 if empty)
        self.yard_container_indices = wp.zeros(
            shape=(self.num_storage_rows, self.num_storage_bays, self.max_stack_height),
            dtype=wp.int32,
            device=self.device
        )
        self.yard_container_indices.fill_(-1)  # Initialize to empty
        
        # Current stack heights - for efficient lookup
        # Format: [row, bay]
        # Value: current height (0 = empty, 1 = one container, etc.)
        self.stack_heights = wp.zeros(
            shape=(self.num_storage_rows, self.num_storage_bays),
            dtype=wp.int32,
            device=self.device
        )

    def _initialize_vehicle_arrays(self):
        """Initialize arrays for managing vehicles (trucks and trains)."""
        # Vehicle state arrays
        
        # Vehicle positions (vehicle_idx -> position_idx)
        # -1 means the vehicle is not currently placed
        self.vehicle_positions = wp.zeros(
            shape=(self.max_vehicles,),
            dtype=wp.int32,
            device=self.device
        )
        self.vehicle_positions.fill_(-1)  # Initialize to "not placed"
        
        # Vehicle properties
        # [0]: Type (0=Truck, 1=Train, 2=Terminal Truck)
        # [1]: Status (0=Arriving, 1=Waiting, 2=Loading, 3=Departing, 4=Departed)
        # [2]: Is pickup vehicle (0/1)
        # [3]: Max containers
        # [4]: Arrival time (seconds from simulation start)
        # [5]: Departure time (seconds from simulation start) - 0 if not scheduled
        # [6]: Active flag (0/1) - 1 if the vehicle is active in the simulation
        self.vehicle_properties = wp.zeros(
            shape=(self.max_vehicles, 7),
            dtype=wp.float32,
            device=self.device
        )
        
        # Container indices on each vehicle
        # Format: [vehicle_idx, container_slot]
        # Value: container index (or -1 if empty)
        self.vehicle_containers = wp.zeros(
            shape=(self.max_vehicles, 10),  # Assuming max 10 containers per vehicle
            dtype=wp.int32,
            device=self.device
        )
        self.vehicle_containers.fill_(-1)  # Initialize to empty
        
        # Vehicle container counts
        self.vehicle_container_counts = wp.zeros(
            shape=(self.max_vehicles,),
            dtype=wp.int32,
            device=self.device
        )
        
        # Parking spot mapping
        # Format: [spot]
        # Value: vehicle index (or -1 if empty)
        self.parking_vehicles = wp.zeros(
            shape=(self.num_parking_spots,),
            dtype=wp.int32,
            device=self.device
        )
        self.parking_vehicles.fill_(-1)  # Initialize to empty
        
        # Rail track mapping
        # Format: [track]
        # Value: vehicle (train) index (or -1 if empty)
        self.rail_track_vehicles = wp.zeros(
            shape=(self.num_rail_tracks,),
            dtype=wp.int32,
            device=self.device
        )
        self.rail_track_vehicles.fill_(-1)  # Initialize to empty
        
        # Vehicle queues (implemented as circular buffers for GPU)
        # Format: [queue_type, queue_position]
        # Value: vehicle index (or -1 if empty)
        self.vehicle_queues = wp.zeros(
            shape=(2, 100),  # 2 queues (truck, train), max 100 vehicles each
            dtype=wp.int32,
            device=self.device
        )
        self.vehicle_queues.fill_(-1)  # Initialize to empty
        
        # Queue size counters
        self.queue_sizes = wp.zeros(
            shape=(2,),  # 2 counters (truck, train)
            dtype=wp.int32,
            device=self.device
        )

    def _initialize_crane_arrays(self):
        """Initialize arrays for managing RMG cranes."""
        # Crane positions [crane_idx, 2] - (x, y) coordinates
        self.crane_positions = wp.zeros(
            shape=(self.num_cranes, 2),
            dtype=wp.float32,
            device=self.device
        )
        
        # Crane target positions [crane_idx, 2] - (x, y) coordinates
        self.crane_target_positions = wp.zeros(
            shape=(self.num_cranes, 2),
            dtype=wp.float32,
            device=self.device
        )
        
        # Crane operational areas [crane_idx, 4] - (min_x, min_y, max_x, max_y)
        self.crane_operational_areas = wp.zeros(
            shape=(self.num_cranes, 4),
            dtype=wp.float32,
            device=self.device
        )
        
        # Crane properties
        # [0]: Status (0=Idle, 1=Moving, 2=Lifting)
        # [1]: Current load (container index, -1 if none)
        # [2]: Available time (seconds from simulation start)
        # [3]: Speed (meters per second)
        self.crane_properties = wp.zeros(
            shape=(self.num_cranes, 4),
            dtype=wp.float32,
            device=self.device
        )
        
        # Set default speeds (m/s)
        for i in range(self.num_cranes):
            self.crane_properties[i, 3] = 2.0  # 2 m/s default speed

    def _initialize_simulation_state(self):
        """Initialize arrays for simulation state."""
        # Current simulation time (seconds)
        self.simulation_time = wp.zeros(
            shape=(1,),
            dtype=wp.float32,
            device=self.device
        )
        
        # Simulation parameters
        # [0]: Max simulation time (seconds)
        # [1]: Start timestamp (for datetime calculations)
        # [2]: Random seed
        self.simulation_params = wp.zeros(
            shape=(3,),
            dtype=wp.float32,
            device=self.device
        )
        
        # Event queue (for time-based events)
        # Format: [event_idx, 3]
        # [0]: Event time (seconds from simulation start)
        # [1]: Event type (0=Vehicle Arrival, 1=Vehicle Departure, etc.)
        # [2]: Event data (vehicle index, etc.)
        self.event_queue = wp.zeros(
            shape=(1000, 3),  # Max 1000 events
            dtype=wp.float32,
            device=self.device
        )
        
        # Event queue size
        self.event_queue_size = wp.zeros(
            shape=(1,),
            dtype=wp.int32,
            device=self.device
        )

    def _create_position_mapping(self):
        """Create mappings between position strings and indices."""
        # This is a CPU-side mapping to help interface with the existing codebase
        self.position_to_idx = {}
        self.idx_to_position = {}
        
        # Add rail positions
        idx = 0
        for track in range(self.num_rail_tracks):
            track_name = f"T{track+1}"
            for slot in range(self.num_rail_slots_per_track):
                position = f"{track_name.lower()}_{slot+1}"
                self.position_to_idx[position] = idx
                self.idx_to_position[idx] = position
                idx += 1
        
        # Add parking positions
        for spot in range(self.num_parking_spots):
            position = f"p_{spot+1}"
            self.position_to_idx[position] = idx
            self.idx_to_position[idx] = position
            idx += 1
        
        # Add storage positions
        for row in range(self.num_storage_rows):
            row_name = self.row_names[row]
            for bay in range(self.num_storage_bays):
                position = f"{row_name}{bay+1}"
                self.position_to_idx[position] = idx
                self.idx_to_position[idx] = position
                idx += 1

    # def _register_kernels(self):
    #     """Register warp kernels for simulation operations."""
    #     # We'll implement and register various kernels for operations
    #     # These will be defined in separate files and imported here
    #     pass

    def get_container_properties(self, container_idx):
        """Get container properties as a dictionary."""
        if container_idx < 0 or container_idx >= self.max_containers:
            return None
            
        # Extract properties from the GPU array
        props = self.container_properties[container_idx].numpy()
        dims = self.container_dimensions[container_idx].numpy()
        
        # Map numeric codes to human-readable values
        type_codes = ["TEU", "FEU", "HQ", "Trailer", "Swap Body"]
        goods_codes = ["Regular", "Reefer", "Dangerous"]
        stack_codes = ["none", "self", "size"]
        
        # Create a dictionary of properties
        return {
            "type": type_codes[int(props[0])],
            "goods_type": goods_codes[int(props[1])],
            "priority": int(props[2]),
            "weight": props[3],
            "is_stackable": bool(props[4]),
            "stack_compatibility": stack_codes[int(props[5])],
            "active": bool(props[6]),
            "departure_time": props[7],
            "dimensions": {
                "length": dims[0],
                "width": dims[1],
                "height": dims[2]
            }
        }

    def _initialize_crane_arrays(self):
        """Initialize arrays for managing RMG cranes."""
        # Crane positions [crane_idx, 2] - (x, y) coordinates
        self.crane_positions = wp.zeros(
            shape=(self.num_cranes, 2),
            dtype=wp.float32,
            device=self.device
        )
        
        # Crane target positions [crane_idx, 2] - (x, y) coordinates
        self.crane_target_positions = wp.zeros(
            shape=(self.num_cranes, 2),
            dtype=wp.float32,
            device=self.device
        )
        
        # Crane operational areas [crane_idx, 4] - (min_x, min_y, max_x, max_y)
        self.crane_operational_areas = wp.zeros(
            shape=(self.num_cranes, 4),
            dtype=wp.float32,
            device=self.device
        )
        
        # Create a CPU-side array with default values first
        crane_props = np.zeros((self.num_cranes, 4), dtype=np.float32)
        
        # Set default values on the CPU array
        for i in range(self.num_cranes):
            # [0]: Status (0=Idle, 1=Moving, 2=Lifting)
            crane_props[i, 0] = 0.0  # Idle status
            # [1]: Current load (container index, -1 if none)
            crane_props[i, 1] = -1.0  # No container loaded
            # [2]: Available time (seconds from simulation start)
            crane_props[i, 2] = 0.0  # Available immediately
            # [3]: Speed (meters per second)
            crane_props[i, 3] = 2.0  # 2 m/s default speed
        
        # Create Warp array from the prepared numpy array
        self.crane_properties = wp.array(crane_props, dtype=wp.float32, device=self.device)

    def add_container(self, container_id, properties, dimensions):
        """Add a container to the registry."""
        # Find an empty slot in the registry
        container_idx = -1
        for i in range(self.max_containers):
            if not bool(self.container_properties[i, 6]):  # Check active flag
                container_idx = i
                break
        
        if container_idx == -1:
            print("Warning: Container registry full")
            return -1
        
        # Add container to the registry
        self.container_indices[container_id] = container_idx
        
        # Set properties
        self.container_properties[container_idx, 0] = properties["type_code"]
        self.container_properties[container_idx, 1] = properties["goods_code"]
        self.container_properties[container_idx, 2] = properties["priority"]
        self.container_properties[container_idx, 3] = properties["weight"]
        self.container_properties[container_idx, 4] = 1.0 if properties["is_stackable"] else 0.0
        self.container_properties[container_idx, 5] = properties["stack_code"]
        self.container_properties[container_idx, 6] = 1.0  # Set active flag
        self.container_properties[container_idx, 7] = properties.get("departure_time", 0.0)
        
        # Set dimensions
        self.container_dimensions[container_idx, 0] = dimensions["length"]
        self.container_dimensions[container_idx, 1] = dimensions["width"]
        self.container_dimensions[container_idx, 2] = dimensions["height"]
        
        return container_idx

    def calculate_distance_matrix(self):
        """Calculate distances between all positions using a warp kernel."""
        # This will be implemented with a warp kernel for parallelization
        # For now, placeholder implementation
        pass

    def to_cpu(self):
        """Move all warp arrays to CPU for debugging or serialization."""
        # We only need to create CPU-side copies when explicitly requested
        pass

    def to_device(self, device):
        """Move all warp arrays to the specified device."""
        if self.device == device:
            return
            
        self.device = device
        # Warp handles device transfers differently than PyTorch
        # This would need to be implemented according to Warp's API
        pass