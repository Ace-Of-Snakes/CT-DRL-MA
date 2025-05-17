import warp as wp
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import time
import os

# Standalone function kernels for reuse across instances
@wp.kernel
def kernel_calculate_terminal_distances(positions: wp.array(dtype=wp.float32, ndim=2),
                                     distance_matrix: wp.array(dtype=wp.float32, ndim=2),
                                     num_positions: wp.int32):
    """
    Kernel to calculate distances between all positions.
    """
    # Get position indices from thread ID
    i, j = wp.tid()
    
    # Check bounds
    if i >= num_positions or j >= num_positions:
        return
    
    # Calculate Euclidean distance
    dx = positions[i, 0] - positions[j, 0]
    dy = positions[i, 1] - positions[j, 1]
    distance = wp.sqrt(dx*dx + dy*dy)
    
    # Store in distance matrix
    distance_matrix[i, j] = distance


@wp.func
def calculate_travel_time(distance: wp.float32, max_speed: wp.float32, acceleration: wp.float32) -> float:
    """
    Calculate travel time with acceleration and deceleration.
    
    Args:
        distance: Distance to travel in meters
        max_speed: Maximum speed in meters per second
        acceleration: Acceleration/deceleration in meters per second squared
        
    Returns:
        Time in seconds needed for the movement
    """
    # No movement needed
    if distance <= 0.0:
        return 0.0
    
    # Calculate the distance needed to reach max speed
    accel_distance = 0.5 * max_speed * max_speed / acceleration
    
    # If we can't reach max speed (distance is too short)
    if distance <= 2.0 * accel_distance:
        # Time to accelerate and then immediately decelerate
        peak_speed = wp.sqrt(acceleration * distance)
        return 2.0 * peak_speed / acceleration
    else:
        # Time to accelerate + time at max speed + time to decelerate
        accel_time = max_speed / acceleration
        constant_speed_distance = distance - 2.0 * accel_distance
        constant_speed_time = constant_speed_distance / max_speed
        return 2.0 * accel_time + constant_speed_time


@wp.kernel
def kernel_calculate_travel_time(distance: wp.float32,
                               max_speed: wp.float32,
                               acceleration: wp.float32,
                               time: wp.array(dtype=wp.float32, ndim=1)):
    """
    Kernel to calculate travel time with acceleration and deceleration.
    
    Args:
        distance: Distance to travel in meters
        max_speed: Maximum speed in meters per second
        acceleration: Acceleration/deceleration in meters per second squared
        time: Output time in seconds
    """
    # No movement needed
    if distance <= 0:
        time[0] = 0.0
        return
    
    # Calculate the distance needed to reach max speed
    accel_distance = 0.5 * max_speed**2.0 / acceleration
    
    # If we can't reach max speed (distance is too short)
    if distance <= 2.0 * accel_distance:
        # Time to accelerate and then immediately decelerate
        peak_speed = wp.sqrt(acceleration * distance)
        time[0] = 2.0 * peak_speed / acceleration
    else:
        # Time to accelerate + time at max speed + time to decelerate
        accel_time = max_speed / acceleration
        constant_speed_distance = distance - 2.0 * accel_distance
        constant_speed_time = constant_speed_distance / max_speed
        time[0] = 2.0 * accel_time + constant_speed_time


@wp.kernel
def kernel_calculate_movement_time(src_position: wp.array(dtype=wp.float32, ndim=1),
                                dst_position: wp.array(dtype=wp.float32, ndim=1),
                                crane_position: wp.array(dtype=wp.float32, ndim=1),
                                src_type: wp.int32,
                                dst_type: wp.int32,
                                container_type: wp.int32,
                                stack_height: wp.float32,
                                trolley_speed: wp.float32,
                                hoisting_speed: wp.float32,
                                gantry_speed: wp.float32,
                                trolley_acceleration: wp.float32,
                                hoisting_acceleration: wp.float32,
                                gantry_acceleration: wp.float32,
                                max_height: wp.float32,
                                ground_vehicle_height: wp.float32,
                                container_heights: wp.array(dtype=wp.float32, ndim=1),
                                time_result: wp.array(dtype=wp.float32, ndim=1)):
    """
    Kernel to calculate time needed for a container movement.
    
    Args:
        src_position: Source position [x, y]
        dst_position: Destination position [x, y]
        crane_position: Current crane position [x, y]
        src_type: Source position type (0=rail, 1=truck, 2=storage)
        dst_type: Destination position type (0=rail, 1=truck, 2=storage)
        container_type: Container type index
        stack_height: Height of container stack at destination (or source if from storage)
        trolley_speed: Maximum trolley speed in m/s
        hoisting_speed: Maximum hoisting speed in m/s
        gantry_speed: Maximum gantry speed in m/s
        trolley_acceleration: Trolley acceleration in m/s²
        hoisting_acceleration: Hoisting acceleration in m/s²
        gantry_acceleration: Gantry acceleration in m/s²
        max_height: Maximum height of the RMG crane in meters
        ground_vehicle_height: Height of vehicles from ground in meters
        container_heights: Array of container heights by type
        time_result: Output array for time calculation result
    """
    # Initialize result
    time_result[0] = 0.0
    
    # Get container height based on type
    container_height = container_heights[container_type]
    
    # Calculate distance components
    crane_to_src_x = abs(crane_position[0] - src_position[0])
    crane_to_src_y = abs(crane_position[1] - src_position[1])
    src_to_dst_x = abs(src_position[0] - dst_position[0])
    src_to_dst_y = abs(src_position[1] - dst_position[1])
    
    # Calculate time components directly without auxiliary arrays
    # 1. Time to move crane to source
    gantry_to_src_time = calculate_travel_time(crane_to_src_y, gantry_speed, gantry_acceleration)
    trolley_to_src_time = calculate_travel_time(crane_to_src_x, trolley_speed, trolley_acceleration)
    
    # Total time to source (max of gantry and trolley, as they happen in parallel)
    time_to_src = wp.max(gantry_to_src_time, trolley_to_src_time)
    
    # 2. Time for source to destination movement
    gantry_time = calculate_travel_time(src_to_dst_y, gantry_speed, gantry_acceleration)
    trolley_time = calculate_travel_time(src_to_dst_x, trolley_speed, trolley_acceleration)
    
    # 3. Calculate vertical movement times
    vertical_distance_up = 0.0
    vertical_distance_down = 0.0
    
    # Determine vertical distances based on source and destination types
    if src_type == 0 and dst_type == 2:  # Rail to storage
        vertical_distance_up = max_height - (ground_vehicle_height + container_height)
        vertical_distance_down = max_height - stack_height
    elif src_type == 2 and dst_type == 0:  # Storage to rail
        vertical_distance_up = max_height - stack_height
        vertical_distance_down = max_height - (ground_vehicle_height + container_height)
    elif src_type == 1 and dst_type == 2:  # Truck to storage
        vertical_distance_up = max_height - (ground_vehicle_height + container_height)
        vertical_distance_down = max_height - stack_height
    elif src_type == 2 and dst_type == 1:  # Storage to truck
        vertical_distance_up = max_height - stack_height
        vertical_distance_down = max_height - (ground_vehicle_height + container_height)
    elif (src_type == 0 and dst_type == 1) or (src_type == 1 and dst_type == 0):  # Rail/truck direct moves
        vertical_distance_up = max_height - (ground_vehicle_height + container_height)
        vertical_distance_down = max_height - (ground_vehicle_height + container_height)
    elif src_type == 2 and dst_type == 2:  # Storage to storage (reshuffling)
        vertical_distance_up = max_height - stack_height
        vertical_distance_down = max_height - 0.0  # Assuming destination has no stack
    
    # Calculate vertical times directly
    vertical_up_time = calculate_travel_time(vertical_distance_up, hoisting_speed, hoisting_acceleration)
    vertical_down_time = calculate_travel_time(vertical_distance_down, hoisting_speed, hoisting_acceleration)
    
    # 4. Calculate total time with proper sequencing of operations
    movement_time = 0.0
    
    # In most RMG operations, the sequence is:
    # - Move crane to position (if needed)
    # - Move gantry to position (if needed)
    # - Lift container
    # - Move trolley to destination (can happen while lifting)
    # - Lower container
    
    # Lifting and trolley movement happen in parallel after gantry is positioned
    if gantry_time > 0.1:  # Significant gantry movement
        # Gantry moves first, then trolley and lifting happen together
        time_after_gantry = wp.max(vertical_up_time, trolley_time)
        movement_time = gantry_time + time_after_gantry + vertical_down_time
    else:
        # No gantry movement, lifting and trolley can happen simultaneously
        movement_time = wp.max(vertical_up_time, trolley_time) + vertical_down_time
    
    # Add time for attaching/detaching the container (fixed time)
    attach_detach_time = 10.0  # seconds
    movement_time += attach_detach_time
    
    # Total time is time to source + movement time
    time_result[0] = time_to_src + movement_time


@wp.kernel
def kernel_batch_movement_times(src_positions: wp.array(dtype=wp.float32, ndim=2),
                              dst_positions: wp.array(dtype=wp.float32, ndim=2),
                              src_types: wp.array(dtype=wp.int32, ndim=1),
                              dst_types: wp.array(dtype=wp.int32, ndim=1),
                              container_types: wp.array(dtype=wp.int32, ndim=1),
                              stack_heights: wp.array(dtype=wp.float32, ndim=1),
                              crane_positions: wp.array(dtype=wp.float32, ndim=2),
                              crane_indices: wp.array(dtype=wp.int32, ndim=1),
                              trolley_speed: wp.float32,
                              hoisting_speed: wp.float32,
                              gantry_speed: wp.float32,
                              trolley_acceleration: wp.float32,
                              hoisting_acceleration: wp.float32,
                              gantry_acceleration: wp.float32,
                              max_height: wp.float32,
                              ground_vehicle_height: wp.float32,
                              container_heights: wp.array(dtype=wp.float32, ndim=1),
                              time_results: wp.array(dtype=wp.float32, ndim=1)):
    """
    Kernel to calculate movement times for a batch of movements.
    
    Args:
        src_positions: Source positions [batch_size, 2]
        dst_positions: Destination positions [batch_size, 2]
        src_types: Source position types [batch_size]
        dst_types: Destination position types [batch_size]
        container_types: Container type indices [batch_size]
        stack_heights: Stack heights at destinations [batch_size]
        crane_positions: Crane positions [num_cranes, 2]
        crane_indices: Indices of cranes to use for each movement [batch_size]
        trolley_speed: Maximum trolley speed in m/s
        hoisting_speed: Maximum hoisting speed in m/s
        gantry_speed: Maximum gantry speed in m/s
        trolley_acceleration: Trolley acceleration in m/s²
        hoisting_acceleration: Hoisting acceleration in m/s²
        gantry_acceleration: Gantry acceleration in m/s²
        max_height: Maximum height of the RMG crane in meters
        ground_vehicle_height: Height of vehicles from ground in meters
        container_heights: Array of container heights by type
        time_results: Output array for time calculation results [batch_size]
    """
    # Get batch index from thread ID
    idx = wp.tid()
    
    # Check if within batch size
    if idx >= len(src_positions):
        return
    
    # Get crane index
    crane_idx = crane_indices[idx]
    
    # Extract positions for this movement
    src_pos = src_positions[idx]
    dst_pos = dst_positions[idx]
    crane_pos = crane_positions[crane_idx]
    
    # Use inline calculation for better performance
    # Get container height based on type
    container_height = container_heights[container_types[idx]]
    
    # Calculate distance components
    crane_to_src_x = abs(crane_pos[0] - src_pos[0])
    crane_to_src_y = abs(crane_pos[1] - src_pos[1])
    src_to_dst_x = abs(src_pos[0] - dst_pos[0])
    src_to_dst_y = abs(src_pos[1] - dst_pos[1])
    
    # Time to source
    gantry_to_src_time = calculate_travel_time(crane_to_src_y, gantry_speed, gantry_acceleration)
    trolley_to_src_time = calculate_travel_time(crane_to_src_x, trolley_speed, trolley_acceleration)
    time_to_src = wp.max(gantry_to_src_time, trolley_to_src_time)
    
    # Time for movement
    gantry_time = calculate_travel_time(src_to_dst_y, gantry_speed, gantry_acceleration)
    trolley_time = calculate_travel_time(src_to_dst_x, trolley_speed, trolley_acceleration)
    
    # Vertical distances
    vertical_distance_up = 0.0
    vertical_distance_down = 0.0
    
    # Determine vertical distances based on source and destination types
    if src_types[idx] == 0 and dst_types[idx] == 2:  # Rail to storage
        vertical_distance_up = max_height - (ground_vehicle_height + container_height)
        vertical_distance_down = max_height - stack_heights[idx]
    elif src_types[idx] == 2 and dst_types[idx] == 0:  # Storage to rail
        vertical_distance_up = max_height - stack_heights[idx]
        vertical_distance_down = max_height - (ground_vehicle_height + container_height)
    elif src_types[idx] == 1 and dst_types[idx] == 2:  # Truck to storage
        vertical_distance_up = max_height - (ground_vehicle_height + container_height)
        vertical_distance_down = max_height - stack_heights[idx]
    elif src_types[idx] == 2 and dst_types[idx] == 1:  # Storage to truck
        vertical_distance_up = max_height - stack_heights[idx]
        vertical_distance_down = max_height - (ground_vehicle_height + container_height)
    elif (src_types[idx] == 0 and dst_types[idx] == 1) or (src_types[idx] == 1 and dst_types[idx] == 0):  # Rail/truck
        vertical_distance_up = max_height - (ground_vehicle_height + container_height)
        vertical_distance_down = max_height - (ground_vehicle_height + container_height)
    elif src_types[idx] == 2 and dst_types[idx] == 2:  # Storage to storage
        vertical_distance_up = max_height - stack_heights[idx]
        vertical_distance_down = max_height - 0.0  # Assuming destination has no stack
    
    vertical_up_time = calculate_travel_time(vertical_distance_up, hoisting_speed, hoisting_acceleration)
    vertical_down_time = calculate_travel_time(vertical_distance_down, hoisting_speed, hoisting_acceleration)
    
    # Calculate total time
    movement_time = 0.0
    if gantry_time > 0.1:
        time_after_gantry = wp.max(vertical_up_time, trolley_time)
        movement_time = gantry_time + time_after_gantry + vertical_down_time
    else:
        movement_time = wp.max(vertical_up_time, trolley_time) + vertical_down_time
    
    # Add time for attaching/detaching
    movement_time += 10.0  # seconds for attaching/detaching
    
    # Set result
    time_results[idx] = time_to_src + movement_time


class WarpMovementCalculator:
    """
    GPU-accelerated movement calculator for terminal simulation using NVIDIA Warp.
    Handles crane movement physics, distance calculations, and travel time estimation.
    
    Provides two computation modes:
    1. Standard mode: Calculates movement times dynamically based on physics
    2. Lookup mode: Uses precomputed lookup tables for faster performance
    """
    
    def __init__(self, 
                 terminal_state,
                 trolley_speed: float = 70.0,       # m/min (Liebherr specs)
                 hoisting_speed: float = 28.0,      # m/min with load
                 gantry_speed: float = 130.0,       # m/min
                 trolley_acceleration: float = 0.3, # m/s²
                 hoisting_acceleration: float = 0.2, # m/s²
                 gantry_acceleration: float = 0.1,  # m/s²
                 max_height: float = 20.0,          # meters
                 ground_vehicle_height: float = 1.5, # meters
                 device: str = None):
        """
        Initialize the movement calculator.
        
        Args:
            terminal_state: Reference to the WarpTerminalState object
            trolley_speed: Maximum trolley speed in meters per minute
            hoisting_speed: Maximum hoisting speed in meters per minute
            gantry_speed: Maximum gantry speed in meters per minute
            trolley_acceleration: Trolley acceleration in meters per second squared
            hoisting_acceleration: Hoisting acceleration in meters per second squared
            gantry_acceleration: Gantry acceleration in meters per second squared
            max_height: Maximum height of the RMG crane in meters
            ground_vehicle_height: Height of vehicles from ground in meters
            device: Computation device (if None, will use the terminal_state's device)
        """
        self.terminal_state = terminal_state
        self.device = device if device else terminal_state.device
        
        # Convert speeds from m/min to m/s for easier calculations
        self.trolley_speed = trolley_speed / 60.0  # m/s
        self.hoisting_speed = hoisting_speed / 60.0  # m/s
        self.gantry_speed = gantry_speed / 60.0  # m/s
        
        self.trolley_acceleration = trolley_acceleration
        self.hoisting_acceleration = hoisting_acceleration
        self.gantry_acceleration = gantry_acceleration
        
        self.max_height = max_height
        self.ground_vehicle_height = ground_vehicle_height
        
        # Container heights for different types - create with values in one step
        container_heights_np = np.array([2.59, 2.59, 2.59, 2.59, 4.0, 3.0], dtype=np.float32)
        self.container_heights = wp.array(container_heights_np, dtype=wp.float32, device=self.device)
        
        # Performance tracking
        self.calculation_times = []
        self.batch_calculation_times = []
        self.lookup_times = []
        
        # Lookup table initialization (set later)
        self.lookup_tables = None
        self.lookup_container_types = None
        self.lookup_type_indices = None
        self.lookup_stack_heights = None
        self.use_lookup = False
        
        # Initialize distance matrix
        self._initialize_distance_matrix()
        
        print(f"WarpMovementCalculator initialized on device: {self.device}")
    
    def _initialize_distance_matrix(self) -> None:
        """Initialize the distance matrix for the terminal."""
        # Distance matrix is stored in the terminal state
        # but we need to calculate the values
        self.calculate_distance_matrix()
    
    def calculate_distance_matrix(self) -> None:
        """Calculate distances between all positions in the terminal."""
        # Get all positions
        num_positions = len(self.terminal_state.position_to_idx)
        
        # Create a CPU-side array first
        positions_np = np.zeros((num_positions, 2), dtype=np.float32)
        
        # Fill positions array on CPU
        for pos_str, pos_idx in self.terminal_state.position_to_idx.items():
            # Determine position type and get coordinates
            coords = self._get_position_coordinates(pos_str)
            positions_np[pos_idx, 0] = coords[0]
            positions_np[pos_idx, 1] = coords[1]
        
        # Create warp array from the filled numpy array
        positions = wp.array(positions_np, dtype=wp.float32, device=self.device)
        
        # Calculate distances
        wp.launch(
            kernel=kernel_calculate_terminal_distances,
            dim=[num_positions, num_positions],
            inputs=[
                positions,
                self.terminal_state.distance_matrix,
                num_positions
            ]
        )
    
    def use_lookup_tables(self, 
                        lookup_tables: np.ndarray, 
                        container_types: List[str],
                        stack_heights: int) -> None:
        """
        Configure the movement calculator to use precomputed lookup tables.
        
        Args:
            lookup_tables: Numpy array of precomputed movement times
            container_types: List of container types in the lookup table
            stack_heights: Number of stack heights in the lookup table
        """
        self.lookup_tables = lookup_tables
        self.lookup_container_types = container_types
        self.lookup_type_indices = {t: i for i, t in enumerate(container_types)}
        self.lookup_stack_heights = stack_heights
        self.use_lookup = True
        
        print(f"Movement calculator configured to use lookup tables")
        print(f"Table shape: {lookup_tables.shape}")
        print(f"Memory usage: {lookup_tables.nbytes / (1024*1024):.2f} MB")
    
    def calculate_movement_time(self,
                           src_position_str: str,
                           dst_position_str: str,
                           crane_idx: int = 0,
                           container_type: Union[int, str] = 2,
                           stack_height: float = 0.0) -> float:
        """
        Calculate the time needed for a container movement, using lookup if available.
        
        Args:
            src_position_str: Source position string
            dst_position_str: Destination position string
            crane_idx: Index of crane to use
            container_type: Container type index or string
            stack_height: Height of container stack
            
        Returns:
            Time in seconds needed for the movement
        """
        # Use lookup table if available and enabled
        if self.use_lookup and self.lookup_tables is not None:
            lookup_start_time = time.time()
            
            # Get position indices
            src_idx = self.terminal_state.position_to_idx.get(src_position_str, -1)
            dst_idx = self.terminal_state.position_to_idx.get(dst_position_str, -1)
            
            if src_idx >= 0 and dst_idx >= 0:
                # Convert container type to index if needed
                if isinstance(container_type, str) and self.lookup_type_indices is not None:
                    type_idx = self.lookup_type_indices.get(container_type, 0)
                else:
                    type_idx = container_type if isinstance(container_type, int) else 2
                
                # Convert stack height to index
                container_height = 2.59  # Standard container height
                stack_idx = min(int(stack_height / container_height), self.lookup_stack_heights - 1)
                
                # Get time from lookup table
                try:
                    result = self.lookup_tables[src_idx, dst_idx, type_idx, stack_idx]
                    self.lookup_times.append(time.time() - lookup_start_time)
                    return float(result)
                except (IndexError, TypeError):
                    # Fall back to calculation if lookup fails
                    pass
        
        # Perform original calculation if lookup not available or failed
        return self._calculate_movement_time_physics(src_position_str, dst_position_str,
                                              crane_idx, container_type, stack_height)
    
    def _calculate_movement_time_physics(self,
                                    src_position_str: str,
                                    dst_position_str: str,
                                    crane_idx: int = 0,
                                    container_type: Union[int, str] = 2,
                                    stack_height: float = 0.0) -> float:
        """
        Calculate movement time using physics simulation.
        
        Args:
            src_position_str: Source position string
            dst_position_str: Destination position string
            crane_idx: Index of crane to use
            container_type: Container type index or string
            stack_height: Height of container stack
            
        Returns:
            Time in seconds needed for the movement
        """
        start_time = time.time()
        
        # Get position indices
        if src_position_str not in self.terminal_state.position_to_idx or dst_position_str not in self.terminal_state.position_to_idx:
            print(f"Warning: Invalid position strings: {src_position_str}, {dst_position_str}")
            return 0.0
            
        # Get position types
        src_type = self._get_position_type_code(src_position_str)
        dst_type = self._get_position_type_code(dst_position_str)
        
        # Get position coordinates
        src_coords = self._get_position_coordinates(src_position_str)
        dst_coords = self._get_position_coordinates(dst_position_str)
        
        # Get current crane position - FIXED by using numpy first
        crane_positions_np = self.terminal_state.crane_positions.numpy()
        if crane_idx >= len(crane_positions_np):
            print(f"Warning: Invalid crane index: {crane_idx}")
            return 0.0
            
        crane_pos = crane_positions_np[crane_idx]
        
        # Ensure container_type is an integer
        if isinstance(container_type, str):
            # Map string to index if needed
            type_map = {"TEU": 0, "FEU": 2, "HQ": 3, "Trailer": 4, "Swap Body": 5}
            container_type_idx = type_map.get(container_type, 2)  # Default to FEU (2)
        else:
            container_type_idx = container_type
        
        # Create arrays on device
        src_pos = wp.array(src_coords, dtype=wp.float32, device=self.device)
        dst_pos = wp.array(dst_coords, dtype=wp.float32, device=self.device)
        crane_position = wp.array(crane_pos, dtype=wp.float32, device=self.device)
        time_result = wp.zeros(1, dtype=wp.float32, device=self.device)
        
        # Calculate movement time
        wp.launch(
            kernel=kernel_calculate_movement_time,
            dim=1,
            inputs=[
                src_pos,
                dst_pos,
                crane_position,
                src_type,
                dst_type,
                container_type_idx,
                stack_height,
                self.trolley_speed,
                self.hoisting_speed,
                self.gantry_speed,
                self.trolley_acceleration,
                self.hoisting_acceleration,
                self.gantry_acceleration,
                self.max_height,
                self.ground_vehicle_height,
                self.container_heights,
                time_result
            ]
        )
        
        result = float(time_result.numpy()[0])
        
        # Track performance
        self.calculation_times.append(time.time() - start_time)
        
        return result
    
    def batch_calculate_movement_times(self,
                                     src_position_strs: List[str],
                                     dst_position_strs: List[str],
                                     crane_indices: List[int] = None,
                                     container_types: List[int] = None,
                                     stack_heights: List[float] = None) -> List[float]:
        """
        Calculate movement times for a batch of movements efficiently.
        
        Args:
            src_position_strs: List of source position strings
            dst_position_strs: List of destination position strings
            crane_indices: List of crane indices to use (default: [0, 0, ...])
            container_types: List of container type indices (default: [2, 2, ...])
            stack_heights: List of stack heights (default: [0.0, 0.0, ...])
            
        Returns:
            List of movement times in seconds
        """
        start_time = time.time()
        
        # Check input lengths
        batch_size = len(src_position_strs)
        if len(dst_position_strs) != batch_size:
            raise ValueError("Source and destination lists must have the same length")
        
        # Default arguments if not provided
        if crane_indices is None:
            crane_indices = [0] * batch_size
        elif len(crane_indices) != batch_size:
            raise ValueError("Crane indices list must have the same length as positions")
            
        if container_types is None:
            container_types = [2] * batch_size  # Default to FEU (type code 2)
        elif len(container_types) != batch_size:
            raise ValueError("Container types list must have the same length as positions")
            
        if stack_heights is None:
            stack_heights = [0.0] * batch_size
        elif len(stack_heights) != batch_size:
            raise ValueError("Stack heights list must have the same length as positions")
        
        # If using lookup tables and all parameters are valid
        if self.use_lookup and self.lookup_tables is not None:
            results = []
            for i in range(batch_size):
                # Get movement time using lookup
                try:
                    time_value = self.calculate_movement_time(
                        src_position_strs[i], 
                        dst_position_strs[i],
                        crane_indices[i],
                        container_types[i],
                        stack_heights[i]
                    )
                    results.append(time_value)
                except Exception as e:
                    # Fallback to default value on error
                    print(f"Error in batch lookup: {e}")
                    results.append(100.0)  # Default fallback time
            
            self.batch_calculation_times.append(time.time() - start_time)
            return results
        
        # Otherwise use physics calculation with GPU batch processing
        try:
            # Prepare input arrays
            src_positions = wp.zeros((batch_size, 2), dtype=wp.float32, device=self.device)
            dst_positions = wp.zeros((batch_size, 2), dtype=wp.float32, device=self.device)
            src_types = wp.zeros(batch_size, dtype=wp.int32, device=self.device)
            dst_types = wp.zeros(batch_size, dtype=wp.int32, device=self.device)
            container_types_array = wp.zeros(batch_size, dtype=wp.int32, device=self.device)
            stack_heights_array = wp.zeros(batch_size, dtype=wp.float32, device=self.device)
            crane_indices_array = wp.zeros(batch_size, dtype=wp.int32, device=self.device)
            
            # Fill input arrays
            for i in range(batch_size):
                # Get position coordinates
                src_coords = self._get_position_coordinates(src_position_strs[i])
                dst_coords = self._get_position_coordinates(dst_position_strs[i])
                
                # Get position types
                src_type = self._get_position_type_code(src_position_strs[i])
                dst_type = self._get_position_type_code(dst_position_strs[i])
                
                # Set values in arrays using kernels
                # Since we can't access arrays directly, use separate kernels
                # Instead, fill numpy arrays and transfer to device
                src_positions.numpy()[i, 0] = src_coords[0]
                src_positions.numpy()[i, 1] = src_coords[1]
                dst_positions.numpy()[i, 0] = dst_coords[0]
                dst_positions.numpy()[i, 1] = dst_coords[1]
                src_types.numpy()[i] = src_type
                dst_types.numpy()[i] = dst_type
                container_types_array.numpy()[i] = container_types[i]
                stack_heights_array.numpy()[i] = stack_heights[i]
                crane_indices_array.numpy()[i] = crane_indices[i]
            
            # Prepare output array
            time_results = wp.zeros(batch_size, dtype=wp.float32, device=self.device)
            
            # Calculate movement times
            wp.launch(
                kernel=kernel_batch_movement_times,
                dim=batch_size,
                inputs=[
                    src_positions,
                    dst_positions,
                    src_types,
                    dst_types,
                    container_types_array,
                    stack_heights_array,
                    self.terminal_state.crane_positions,
                    crane_indices_array,
                    self.trolley_speed,
                    self.hoisting_speed,
                    self.gantry_speed,
                    self.trolley_acceleration,
                    self.hoisting_acceleration,
                    self.gantry_acceleration,
                    self.max_height,
                    self.ground_vehicle_height,
                    self.container_heights,
                    time_results
                ]
            )
            
            # Convert to Python list
            results = time_results.numpy().tolist()
            
            # Track performance
            self.batch_calculation_times.append(time.time() - start_time)
            
            return results
        
        except Exception as e:
            print(f"Error in batch calculation: {e}")
            # Fallback to sequential calculation
            results = []
            for i in range(batch_size):
                try:
                    time_value = self._calculate_movement_time_physics(
                        src_position_strs[i], 
                        dst_position_strs[i],
                        crane_indices[i],
                        container_types[i],
                        stack_heights[i]
                    )
                    results.append(time_value)
                except Exception as e2:
                    print(f"Error in fallback calculation: {e2}")
                    results.append(100.0)  # Default fallback time
            
            self.batch_calculation_times.append(time.time() - start_time)
            return results
    
    def generate_lookup_tables(self, 
                             container_types: List[str] = None,
                             max_stack_heights: int = 6,
                             save_path: str = None) -> np.ndarray:
        """
        Generate comprehensive lookup tables for all movement combinations.
        
        Args:
            container_types: List of container types to precompute (default: ["TEU", "FEU", "HQ", "Trailer", "Swap Body"])
            max_stack_heights: Number of stack heights to precompute (default: 6)
            save_path: Path to save the lookup tables (if None, won't save)
            
        Returns:
            Numpy array of precomputed movement times
        """
        generation_start = time.time()
        
        # Default container types if not provided
        if container_types is None:
            container_types = ["TEU", "FEU", "HQ", "Trailer", "Swap Body"]
        
        # Get position information
        num_positions = len(self.terminal_state.position_to_idx)
        
        # Create lookup table arrays
        lookup_tables = np.zeros(
            (num_positions, num_positions, len(container_types), max_stack_heights), 
            dtype=np.float32
        )
        
        # Map string container types to indices
        type_indices = {t: i for i, t in enumerate(container_types)}
        
        # Process positions in batches to avoid memory issues
        batch_size = 50  # Process 50 source positions at a time
        num_batches = (num_positions + batch_size - 1) // batch_size
        
        print(f"Generating movement lookup tables for {num_positions} positions...")
        print(f"Container types: {container_types}")
        print(f"Stack heights: {max_stack_heights}")
        print(f"Estimated memory usage: {lookup_tables.nbytes / (1024*1024):.2f} MB")
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_positions)
            print(f"Processing batch {batch_idx+1}/{num_batches} (positions {start_idx}-{end_idx})...")
            
            # Get source positions for this batch
            for src_idx in range(start_idx, end_idx):
                src_pos = self.terminal_state.idx_to_position.get(src_idx)
                if src_pos is None:
                    continue
                
                # Process all destinations for this source
                for dst_idx in range(num_positions):
                    dst_pos = self.terminal_state.idx_to_position.get(dst_idx)
                    if dst_pos is None or src_pos == dst_pos:
                        continue
                    
                    # Process each container type
                    for type_str, type_idx in type_indices.items():
                        # Convert to numeric type index
                        if type_str == "TEU":
                            numeric_type = 0
                        elif type_str == "FEU":
                            numeric_type = 2
                        elif type_str == "HQ":
                            numeric_type = 3
                        elif type_str == "Trailer":
                            numeric_type = 4
                        elif type_str == "Swap Body":
                            numeric_type = 5
                        else:
                            numeric_type = 2  # Default to FEU
                        
                        # Process each stack height
                        for stack_idx in range(max_stack_heights):
                            # Calculate stack height in meters
                            stack_height = stack_idx * 2.59  # Standard container height
                            
                            try:
                                # Calculate the movement time using physics
                                movement_time = self._calculate_movement_time_physics(
                                    src_pos, dst_pos, 0, numeric_type, stack_height
                                )
                                
                                # Store in lookup table
                                lookup_tables[src_idx, dst_idx, type_idx, stack_idx] = movement_time
                            except Exception as e:
                                # Use a default time if calculation fails
                                print(f"Error calculating {src_pos} to {dst_pos} (type {type_str}, height {stack_height}): {e}")
                                lookup_tables[src_idx, dst_idx, type_idx, stack_idx] = 100.0
        
        # Save tables if path provided
        if save_path is not None:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                
                print(f"Saving lookup tables to {save_path}...")
                np.save(save_path, lookup_tables)
                
                # Save metadata
                metadata_path = os.path.splitext(save_path)[0] + "_metadata.npz"
                np.savez(
                    metadata_path,
                    container_types=container_types,
                    max_stack_heights=max_stack_heights,
                    position_to_idx=list(self.terminal_state.position_to_idx.items()),
                    num_positions=num_positions
                )
                print(f"Saved metadata to {metadata_path}")
            except Exception as e:
                print(f"Error saving lookup tables: {e}")
        
        generation_time = time.time() - generation_start
        print(f"Generated lookup tables in {generation_time:.2f} seconds")
        
        # Configure the calculator to use these tables
        self.use_lookup_tables(lookup_tables, container_types, max_stack_heights)
        
        return lookup_tables
    
    def load_lookup_tables(self, file_path: str) -> bool:
        """
        Load precomputed lookup tables from a file.
        
        Args:
            file_path: Path to the lookup table file
            
        Returns:
            True if tables were loaded successfully, False otherwise
        """
        try:
            # Load the lookup tables
            lookup_tables = np.load(file_path)
            
            # Load metadata if available
            metadata_path = os.path.splitext(file_path)[0] + "_metadata.npz"
            if os.path.exists(metadata_path):
                metadata = np.load(metadata_path, allow_pickle=True)
                container_types = metadata['container_types'].tolist()
                max_stack_heights = int(metadata['max_stack_heights'])
            else:
                # Default metadata if not available
                container_types = ["TEU", "FEU", "HQ", "Trailer", "Swap Body"]
                max_stack_heights = lookup_tables.shape[3]
            
            # Configure to use the loaded tables
            self.use_lookup_tables(lookup_tables, container_types, max_stack_heights)
            
            print(f"Loaded lookup tables from {file_path}")
            return True
            
        except Exception as e:
            print(f"Error loading lookup tables: {e}")
            return False
    
    def _get_position_type_code(self, position_str: str) -> int:
        """
        Convert position string to position type code.
        
        Args:
            position_str: Position string
            
        Returns:
            Position type code (0=rail, 1=truck, 2=storage)
        """
        if position_str.startswith('t') and '_' in position_str:
            return 0  # Rail
        elif position_str.startswith('p_'):
            return 1  # Truck
        else:
            return 2  # Storage
    
    def _get_position_coordinates(self, position_str: str) -> Tuple[float, float]:
        """
        Get coordinates for a position string.
        
        Args:
            position_str: Position string
            
        Returns:
            Tuple of (x, y) coordinates
        """
        # Try to get position from terminal state
        position_idx = self.terminal_state.position_to_idx.get(position_str, -1)
        
        if position_idx == -1:
            # Position not found, return origin
            return (0.0, 0.0)
        
        # Determine position type
        position_type = self._get_position_type_code(position_str)
        
        if position_type == 0:  # Rail
            # Parse rail position
            track_id = position_str.split('_')[0]
            slot_num = int(position_str.split('_')[1])
            track_num = int(track_id[1:])
            
            # Try to get from rail positions
            if hasattr(self.terminal_state, 'rail_positions'):
                coords = self.terminal_state.rail_positions[track_num-1, slot_num-1].numpy()
                return (float(coords[0]), float(coords[1]))
            else:
                # Approximate positions
                return ((slot_num - 1) * 24.0, (track_num - 1) * 5.0)
                
        elif position_type == 1:  # Truck
            # Parse truck position
            spot_num = int(position_str.split('_')[1])
            
            # Try to get from parking positions
            if hasattr(self.terminal_state, 'parking_positions'):
                coords = self.terminal_state.parking_positions[spot_num-1].numpy()
                return (float(coords[0]), float(coords[1]))
            else:
                # Approximate positions
                return ((spot_num - 1) * 24.0, self.terminal_state.num_rail_tracks * 5.0 + 5.0)
                
        else:  # Storage
            # Parse storage position
            row_letter = position_str[0]
            bay_num = int(position_str[1:])
            row_idx = ord(row_letter) - ord('A')
            
            # Try to get from storage positions
            if hasattr(self.terminal_state, 'storage_positions'):
                coords = self.terminal_state.storage_positions[row_idx, bay_num-1].numpy()
                return (float(coords[0]), float(coords[1]))
            else:
                # Approximate positions
                return ((bay_num - 1) * 12.0, self.terminal_state.num_rail_tracks * 5.0 + 10.0 + row_idx * 10.0)
    
    def print_performance_stats(self) -> None:
        """Print performance statistics for movement calculations."""
        print("\nMovement Calculator Performance Statistics:")
        
        if self.calculation_times:
            avg_calc = sum(self.calculation_times) / len(self.calculation_times) * 1000
            print(f"  Single physics calculation: {avg_calc:.2f}ms average")
        
        if self.lookup_times:
            avg_lookup = sum(self.lookup_times) / len(self.lookup_times) * 1000
            print(f"  Lookup table access: {avg_lookup:.2f}ms average")
            
            if self.calculation_times:
                speedup = avg_calc / avg_lookup if avg_lookup > 0 else 0
                print(f"  Lookup speedup: {speedup:.2f}x faster")
        
        if self.batch_calculation_times:
            avg_batch = sum(self.batch_calculation_times) / len(self.batch_calculation_times) * 1000
            print(f"  Batch movement calculation: {avg_batch:.2f}ms average")
        
        # Memory usage
        if hasattr(self, 'lookup_tables') and self.lookup_tables is not None:
            memory_usage = self.lookup_tables.nbytes / (1024 * 1024)  # MB
            print(f"  Lookup table memory usage: {memory_usage:.2f} MB")