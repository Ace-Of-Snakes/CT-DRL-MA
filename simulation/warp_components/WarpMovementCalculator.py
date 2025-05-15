import warp as wp
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time

# Define standalone kernels outside the class to avoid the 'self' parameter issue
@wp.kernel
def kernel_calculate_terminal_distances(positions: wp.array(dtype=wp.float32, ndim=2),
                                     distance_matrix: wp.array(dtype=wp.float32, ndim=2),
                                     num_positions: wp.int32):
    """
    Kernel to calculate distances between all positions.
    """
    # Get position indices from thread ID
    i, j = wp.tid()  # Use tid.x for the first dimension
    
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
def kernel_calculate_travel_time_inline(distance: wp.float32,
                                      max_speed: wp.float32,
                                      acceleration: wp.float32,
                                      time_result: wp.array(dtype=wp.float32)):
    """
    Inlined travel time calculation (to avoid nested kernel calls)
    """
    # No movement needed
    if distance <= 0:
        time_result[0] = 0.0
        return
    
    # Calculate the distance needed to reach max speed
    accel_distance = 0.5 * max_speed**2.0 / acceleration
    
    # If we can't reach max speed (distance is too short)
    if distance <= 2.0 * accel_distance:
        # Time to accelerate and then immediately decelerate
        peak_speed = wp.sqrt(acceleration * distance)
        time_result[0] = 2.0 * peak_speed / acceleration
    else:
        # Time to accelerate + time at max speed + time to decelerate
        accel_time = max_speed / acceleration
        constant_speed_distance = distance - 2.0 * accel_distance
        constant_speed_time = constant_speed_distance / max_speed
        time_result[0] = 2.0 * accel_time + constant_speed_time

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
    
    # Create single-element arrays for the temporary time result
    # Warp doesn't allow creating arrays with wp.zeros in kernels
    
    # Calculate movement time directly by calling movement time kernel
    # This is also not ideal in Warp as kernels calling kernels can be problematic
    
    # Instead, inline the calculation logic directly:
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
        
        # Initialize distance matrix
        self._initialize_distance_matrix()
        
        print(f"WarpMovementCalculator initialized on device: {self.device}")
    
    def _initialize_distance_matrix(self):
        """Initialize the distance matrix for the terminal."""
        # Distance matrix is stored in the terminal state
        # but we need to calculate the values
        self.calculate_distance_matrix()
    
    def calculate_distance_matrix(self):
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
    
    def calculate_movement_time(self,
                            src_position_str: str,
                            dst_position_str: str,
                            crane_idx: int = 0,
                            container_type: int = 2,  # FEU is default
                            stack_height: float = 0.0) -> float:
        """Calculate the time needed for a container movement."""
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
        
        # Get current crane position
        crane_pos = np.array(self.terminal_state.crane_positions[crane_idx].numpy())
        
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
                container_type,
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
        
        result = float(time_result[0])
        
        # Track performance
        self.calculation_times.append(time.time() - start_time)
        
        return result
    
    def batch_calculate_movement_times(self,
                                    src_position_strs: List[str],
                                    dst_position_strs: List[str],
                                    crane_indices: List[int] = None,
                                    container_types: List[int] = None,
                                    stack_heights: List[float] = None) -> List[float]:
        """Calculate movement times for a batch of movements."""
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
            
            # Set position coordinates
            src_positions[i, 0] = src_coords[0]
            src_positions[i, 1] = src_coords[1]
            dst_positions[i, 0] = dst_coords[0]
            dst_positions[i, 1] = dst_coords[1]
            
            # Set types
            src_types[i] = self._get_position_type_code(src_position_strs[i])
            dst_types[i] = self._get_position_type_code(dst_position_strs[i])
            
            # Set other properties
            container_types_array[i] = container_types[i]
            stack_heights_array[i] = stack_heights[i]
            crane_indices_array[i] = crane_indices[i]
        
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
    
    def _get_position_type_code(self, position_str: str) -> int:
        """Convert position string to position type code (0=rail, 1=truck, 2=storage)."""
        if position_str.startswith('t') and '_' in position_str:
            return 0  # Rail
        elif position_str.startswith('p_'):
            return 1  # Truck
        else:
            return 2  # Storage
    
    def _get_position_coordinates(self, position_str: str) -> Tuple[float, float]:
        """Get coordinates for a position string."""
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
    
    def print_performance_stats(self):
        """Print performance statistics for movement calculations."""
        if not self.calculation_times and not self.batch_calculation_times:
            print("No performance data available.")
            return
        
        print("\nMovement Calculator Performance Statistics:")
        
        if self.calculation_times:
            avg_calc = sum(self.calculation_times) / len(self.calculation_times) * 1000
            print(f"  Single movement calculation: {avg_calc:.2f}ms average")
        
        if self.batch_calculation_times:
            avg_batch = sum(self.batch_calculation_times) / len(self.batch_calculation_times) * 1000
            print(f"  Batch movement calculation: {avg_batch:.2f}ms average")