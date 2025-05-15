import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import time

class GPURMGCrane:
    """
    GPU-accelerated implementation of Rail Mounted Gantry (RMG) Crane.
    Uses PyTorch tensors for efficient calculations on GPU.
    """
    
    def __init__(self, 
                 crane_id: str,
                 terminal: Any,
                 start_bay: int,
                 end_bay: int,
                 current_position: Tuple[int, int] = (0, 0),
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize a GPU-accelerated RMG crane.
        
        Args:
            crane_id: Unique identifier for the crane
            terminal: Terminal object reference
            start_bay: Starting bay index of crane's operational area
            end_bay: Ending bay index of crane's operational area
            current_position: Initial position as (bay, row)
            device: Device to use for tensors ('cuda' for GPU or 'cpu')
        """
        self.crane_id = crane_id
        self.terminal = terminal
        self.start_bay = start_bay
        self.end_bay = end_bay
        self.current_position = current_position
        self.device = device
        
        # Store previous position for movement calculations
        self.previous_position = current_position
        
        # Crane specifications (from original implementation)
        self.trolley_speed = 70.0 / 60.0  # m/s (converted from m/min)
        self.hoisting_speed = 28.0 / 60.0  # m/s (converted from m/min)
        self.gantry_speed = 130.0 / 60.0  # m/s (converted from m/min)
        self.trolley_acceleration = 0.3  # m/s²
        self.hoisting_acceleration = 0.2  # m/s²
        self.gantry_acceleration = 0.1  # m/s²
        self.max_height = 20.0  # meters
        self.ground_vehicle_height = 1.5  # meters
        
        # Create tensor for crane's operational area
        self._create_operational_area_mask()
        
        # Container height lookup
        self.container_heights = {
            "TWEU": 2.59,
            "THEU": 2.59,
            "FEU": 2.59,
            "FFEU": 2.59,
            "default": 2.59
        }
        
        # Movement time cache for repeated operations
        self.movement_time_cache = {}
        
        print(f"Initialized GPU-accelerated RMG Crane {self.crane_id} on {device}")
    
    def _create_operational_area_mask(self):
        """Create a tensor mask for this crane's operational area."""
        # Get dimensions from terminal if available
        if hasattr(self.terminal, 'num_storage_slots_per_row') and hasattr(self.terminal, 'num_storage_rows'):
            num_bays = self.terminal.num_storage_slots_per_row
            num_rows = self.terminal.num_storage_rows
        else:
            # Default values if not available
            num_bays = 100
            num_rows = 10
        
        # Create operational area mask for storage yard
        self.operational_area_mask = torch.zeros((num_rows, num_bays), dtype=torch.bool, device=self.device)
        self.operational_area_mask[:, self.start_bay:self.end_bay+1] = True
    
    def reset(self, position: Tuple[int, int] = None):
        """Reset the crane to initial position."""
        if position is None:
            self.current_position = (self.start_bay, 0)
        else:
            self.current_position = position
        
        self.previous_position = self.current_position
        self.movement_time_cache = {}
        
        # Reset operational area mask (in case device has changed)
        self._create_operational_area_mask()
    
    def move_container(self, 
                      source_position: str, 
                      destination_position: str,
                      storage_yard: Any,
                      trucks_in_terminal: Dict[str, Any] = None,
                      trains_in_terminal: Dict[str, Any] = None) -> Tuple[Optional[Any], float]:
        """
        Move a container from source to destination position.
        
        Args:
            source_position: Source position string
            destination_position: Destination position string
            storage_yard: Storage yard object
            trucks_in_terminal: Dictionary of trucks at positions
            trains_in_terminal: Dictionary of trains at positions
            
        Returns:
            Tuple of (container moved, time taken)
        """
        # Check if both positions are in this crane's operational area
        if not (self._is_position_in_crane_area(source_position, storage_yard) and 
                self._is_position_in_crane_area(destination_position, storage_yard)):
            return None, 0
        
        # Get the container at the source position
        container, _ = storage_yard.get_top_container(source_position)
        
        if container is None:
            return None, 0
        
        # Get position indices
        src_indices = storage_yard.position_to_indices.get(source_position, (None, None))
        dst_indices = storage_yard.position_to_indices.get(destination_position, (None, None))
        
        if src_indices[0] is None or dst_indices[0] is None:
            return None, 0
        
        # Check if the destination can accept this container
        if not storage_yard.can_accept_container(destination_position, container):
            return None, 0
        
        # Calculate movement time
        time_taken = self._calculate_movement_time(
            src_indices, dst_indices, 
            getattr(container, 'container_type', 'default')
        )
        
        # Remove the container from source
        removed_container = storage_yard.remove_container(source_position)
        
        if removed_container is None:
            return None, 0
        
        # Place the container at destination
        success = storage_yard.add_container(destination_position, removed_container)
        
        if success:
            # Update crane position
            self.previous_position = self.current_position
            self.current_position = (dst_indices[1], dst_indices[0])  # (bay, row)
            
            return removed_container, time_taken
        else:
            # If placement failed, put the container back
            storage_yard.add_container(source_position, removed_container)
            return None, 0
    
    def _calculate_movement_time(self, 
                              source_indices: Tuple[int, int],
                              destination_indices: Tuple[int, int],
                              container_type: str = 'default') -> float:
        """
        Calculate time needed for a container movement.
        
        Args:
            source_indices: (row_idx, bay_idx) of source position
            destination_indices: (row_idx, bay_idx) of destination position
            container_type: Type of container for height calculation
            
        Returns:
            Time in seconds for the movement
        """
        # Check cache first
        cache_key = (source_indices, destination_indices, container_type)
        if cache_key in self.movement_time_cache:
            return self.movement_time_cache[cache_key]
        
        # Unpack indices
        src_row_idx, src_bay_idx = source_indices
        dst_row_idx, dst_bay_idx = destination_indices
        
        # Try to use terminal distances if available
        if hasattr(self.terminal, 'positions'):
            # Look for position strings that match these indices
            source_position = None
            dest_position = None
            
            # Check if storage yard has indices_to_position method
            if hasattr(self.terminal, 'storage_yard') and hasattr(self.terminal.storage_yard, 'indices_to_position'):
                source_position = self.terminal.storage_yard.indices_to_position(src_row_idx, src_bay_idx)
                dest_position = self.terminal.storage_yard.indices_to_position(dst_row_idx, dst_bay_idx)
            
            # Or try to construct position strings directly
            if source_position is None and hasattr(self.terminal, 'storage_row_names'):
                try:
                    src_row = self.terminal.storage_row_names[src_row_idx]
                    dst_row = self.terminal.storage_row_names[dst_row_idx]
                    source_position = f"{src_row}{src_bay_idx+1}"
                    dest_position = f"{dst_row}{dst_bay_idx+1}"
                except IndexError:
                    # Fallback to simplified calculation if row names don't match
                    pass
            
            # If we have valid position strings, try to use terminal distances
            if (source_position and dest_position and 
                source_position in self.terminal.positions and 
                dest_position in self.terminal.positions):
                # Use terminal-based calculation
                src_pos = self.terminal.positions[source_position]
                dst_pos = self.terminal.positions[dest_position]
                
                # Calculate distances
                gantry_distance = abs(src_pos[1] - dst_pos[1])
                trolley_distance = abs(src_pos[0] - dst_pos[0])
                
                # Calculate times
                gantry_time = self._calculate_travel_time(
                    gantry_distance, self.gantry_speed, self.gantry_acceleration)
                trolley_time = self._calculate_travel_time(
                    trolley_distance, self.trolley_speed, self.trolley_acceleration)
                
                # Simplified vertical movement time (constant)
                vertical_time = 30.0  # seconds
                
                # Calculate total time
                if gantry_distance > 1.0:  # Significant gantry movement
                    total_time = gantry_time + max(trolley_time, vertical_time)
                else:
                    total_time = max(trolley_time, vertical_time)
                
                # Add container handling time
                total_time += 10.0  # seconds for attaching/detaching
                
                # Add time from previous position to source
                prev_bay, prev_row = self.previous_position
                distance_to_source = np.sqrt(
                    ((src_bay_idx - prev_bay) * 6)**2 + ((src_row_idx - prev_row) * 12)**2
                )
                time_to_source = distance_to_source / max(self.gantry_speed, 1.0)
                total_time += time_to_source
                
                # Save to cache and return
                self.movement_time_cache[cache_key] = total_time
                return total_time
        
        # If terminal distances not available, use simplified calculation
        # Convert to approximate physical distances
        bay_distance = abs(src_bay_idx - dst_bay_idx) * 6  # 6 meters per bay (approx)
        row_distance = abs(src_row_idx - dst_row_idx) * 12  # 12 meters per row (approx)
        
        # Calculate component movement times
        trolley_time = self._calculate_travel_time(bay_distance, self.trolley_speed, self.trolley_acceleration)
        gantry_time = self._calculate_travel_time(row_distance, self.gantry_speed, self.gantry_acceleration)
        
        # Simplified vertical time
        vertical_time = 30.0  # seconds
        
        # Calculate total time based on movement sequence
        if row_distance > 1.0:  # Significant gantry movement
            total_time = gantry_time + max(trolley_time, vertical_time)
        else:
            total_time = max(trolley_time, vertical_time)
        
        # Add container handling time
        total_time += 10.0  # seconds for attaching/detaching
        
        # Add time from previous position to source
        prev_bay, prev_row = self.previous_position
        distance_to_source = np.sqrt(
            ((src_bay_idx - prev_bay) * 6)**2 + ((src_row_idx - prev_row) * 12)**2
        )
        time_to_source = distance_to_source / max(self.gantry_speed, 1.0)
        total_time += time_to_source
        
        # Cache and return
        self.movement_time_cache[cache_key] = total_time
        return total_time
    
    def _calculate_travel_time(self, distance: float, max_speed: float, acceleration: float) -> float:
        """
        Calculate travel time with acceleration and deceleration.
        
        Args:
            distance: Distance to travel in meters
            max_speed: Maximum speed in meters per second
            acceleration: Acceleration/deceleration in meters per second squared
            
        Returns:
            Time in seconds
        """
        if distance <= 0:
            return 0.0
        
        # Calculate the distance needed to reach max speed
        accel_distance = 0.5 * max_speed**2 / acceleration
        
        # If we can't reach max speed (distance too short)
        if distance <= 2 * accel_distance:
            # Time to accelerate and then immediately decelerate
            peak_speed = np.sqrt(acceleration * distance)
            time = 2 * peak_speed / acceleration
        else:
            # Time to accelerate + time at max speed + time to decelerate
            accel_time = max_speed / acceleration
            constant_speed_distance = distance - 2 * accel_distance
            constant_speed_time = constant_speed_distance / max_speed
            time = 2 * accel_time + constant_speed_time
        
        return time
    
    def _is_position_in_crane_area(self, position: str, storage_yard: Any) -> bool:
        """Check if a position is within this crane's operational area."""
        # If not a storage position, handle differently
        if not position[0].isalpha() or not position[1:].isdigit():
            # For rail and truck positions, check bay alignment
            if '_' in position:  # Rail position (e.g., "t1_5")
                parts = position.split('_')
                if len(parts) == 2 and parts[1].isdigit():
                    slot = int(parts[1])
                    # Approximate check - assume slots align with bays
                    return self.start_bay <= slot - 1 <= self.end_bay
            return True  # Default to True for non-storage positions
        
        # For storage positions, check using position_to_indices
        if hasattr(storage_yard, 'position_to_indices'):
            indices = storage_yard.position_to_indices.get(position)
            if indices:
                row_idx, bay_idx = indices
                return self.start_bay <= bay_idx <= self.end_bay
        
        # Fallback - extract bay number directly
        try:
            bay = int(position[1:]) - 1  # Convert to 0-based
            return self.start_bay <= bay <= self.end_bay
        except:
            return False
    
    def to_device(self, device: str):
        """Move all tensors to specified device."""
        self.device = device
        self.operational_area_mask = self.operational_area_mask.to(device)
        self.movement_time_cache = {}  # Clear cache
    
    def to_cpu(self):
        """Move all tensors to CPU."""
        self.to_device('cpu')
    
    def __str__(self):
        return (f"GPURMGCrane {self.crane_id}: position={self.current_position}, "
                f"operational_area=[{self.start_bay}-{self.end_bay}], device={self.device}")