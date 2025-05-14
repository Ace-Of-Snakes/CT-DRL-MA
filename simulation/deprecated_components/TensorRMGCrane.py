import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
import re


class TensorRMGCrane:
    """
    Tensor-based implementation of Rail Mounted Gantry (RMG) Crane.
    Uses tensor operations for fast calculation of valid moves.
    """
    
    def __init__(self, 
                 crane_id: str,
                 terminal: Any,
                 start_bay: int,
                 end_bay: int,
                 current_position: Tuple[int, int] = (0, 0),
                 device: str = None):
        """Initialize a new RMG crane with tensor support."""
        self.crane_id = crane_id
        self.terminal = terminal
        self.start_bay = start_bay
        self.end_bay = end_bay
        self.current_position = current_position
        
        # Set device for tensors
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Crane specifications (same as original)
        self.trolley_speed = 70.0 / 60.0  # m/s (converted from m/min)
        self.hoisting_speed = 28.0 / 60.0  # m/s (converted from m/min)
        self.gantry_speed = 130.0 / 60.0  # m/s (converted from m/min)
        self.trolley_acceleration = 0.3  # m/s²
        self.hoisting_acceleration = 0.2  # m/s²
        self.gantry_acceleration = 0.1  # m/s²
        self.max_height = 20.0  # meters
        self.container_heights = {
            "TWEU": 2.59,
            "THEU": 2.59,
            "FEU": 2.59,
            "FFEU": 2.59,
            "default": 2.59
        }
        self.ground_vehicle_height = 1.5  # meters
        
        # Initialize operational area tensor
        self._create_operational_area_mask()
        
        # Movement time cache
        self.movement_time_cache = {}

    def _create_operational_area_mask(self):
        """Create a tensor mask for this crane's operational area."""
        # Assume num_bays and num_rows from terminal if available
        if hasattr(self.terminal, 'num_bays') and hasattr(self.terminal, 'num_rows'):
            num_bays = self.terminal.num_bays
            num_rows = self.terminal.num_rows
        else:
            # Fallback to reasonable defaults
            num_bays = 100
            num_rows = 10
        
        # Create operational area mask for storage yard
        self.operational_area_mask = torch.zeros((num_rows, num_bays), dtype=torch.bool, device=self.device)
        self.operational_area_mask[:, self.start_bay:self.end_bay+1] = True

    def reset(self, position: Tuple[int, int] = None):
        """Reset the crane to its initial position."""
        if position is None:
            self.current_position = (self.start_bay, 0)
        else:
            self.current_position = position
        self.movement_time_cache = {}

    def generate_valid_source_mask(self, 
                               storage_yard: Any, 
                               trucks_mask: torch.Tensor = None,
                               trains_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Generate a boolean mask for valid source positions.
        
        Args:
            storage_yard: TensorStorageYard object
            trucks_mask: Boolean mask for truck positions with containers [num_positions]
            trains_mask: Boolean mask for train positions with containers [num_positions]
            
        Returns:
            Boolean mask of valid source positions [num_rows, num_bays]
        """
        # Start with storage yard extraction mask (positions where containers can be picked up)
        valid_source_mask = storage_yard.generate_extraction_mask()
        
        # Limit to this crane's operational area
        valid_source_mask = valid_source_mask & self.operational_area_mask
        
        return valid_source_mask

    def generate_valid_destination_mask(self, 
                                    source_position: str,
                                    container: Any,
                                    storage_yard: Any) -> torch.Tensor:
        """
        Generate a boolean mask for valid destinations from a specific source.
        
        Args:
            source_position: Source position string
            container: Container object to move
            storage_yard: TensorStorageYard object
            
        Returns:
            Boolean mask of valid destination positions [num_rows, num_bays]
        """
        # Get valid placement mask for this container
        valid_dest_mask = storage_yard.get_valid_placement_mask(container)
        
        # Limit to this crane's operational area
        valid_dest_mask = valid_dest_mask & self.operational_area_mask
        
        # Parse source position
        if source_position in storage_yard.position_to_indices:
            src_row_idx, src_bay_idx = storage_yard.position_to_indices[source_position]
            
            # For reshuffling within storage - enforce distance constraint (max 5 bays)
            if storage_yard._is_storage_position(source_position):
                # Create a distance mask
                bay_indices = torch.arange(storage_yard.num_bays, device=self.device)
                bay_distances = torch.abs(bay_indices - src_bay_idx)
                
                # Limit to 5 bay distance for reshuffling
                distance_mask = bay_distances <= 5
                distance_mask = distance_mask.unsqueeze(0).expand(storage_yard.num_rows, -1)
                
                # Apply distance constraint
                valid_dest_mask = valid_dest_mask & distance_mask
            
            # Cannot move a container to its own position
            valid_dest_mask[src_row_idx, src_bay_idx] = False
        
        return valid_dest_mask

    def calculate_movement_time_tensor(self, 
                                    source_indices: Tuple[int, int], 
                                    dest_indices: Tuple[int, int],
                                    container_type: str = "default") -> float:
        """
        Calculate the time needed to move a container using tensor indices.
        
        Args:
            source_indices: Source position indices (row_idx, bay_idx)
            dest_indices: Destination position indices (row_idx, bay_idx)
            container_type: Type of container for height calculation
            
        Returns:
            Time in seconds for the movement
        """
        # Check if we have position coordinates from the terminal
        if hasattr(self.terminal, 'positions') and hasattr(self.terminal, 'get_distance'):
            # Convert indices to position strings
            source_position = None
            dest_position = None
            
            # If storage yard is available, use its conversion method
            if hasattr(self.terminal, 'storage_yard') and hasattr(self.terminal.storage_yard, 'indices_to_position'):
                source_position = self.terminal.storage_yard.indices_to_position(*source_indices)
                dest_position = self.terminal.storage_yard.indices_to_position(*dest_indices)
            
            # If positions are valid and in terminal, use actual distance calculation
            if (source_position and dest_position and 
                source_position in self.terminal.positions and 
                dest_position in self.terminal.positions):
                
                return self._calculate_movement_time_from_positions(
                    source_position, dest_position, container_type)
        
        # Fallback to simplified calculation based on indices
        src_row_idx, src_bay_idx = source_indices
        dst_row_idx, dst_bay_idx = dest_indices
        
        # Calculate distances
        bay_distance = abs(src_bay_idx - dst_bay_idx)
        row_distance = abs(src_row_idx - dst_row_idx)
        
        # Approximate distances in meters
        bay_meters = bay_distance * 6  # Assume 6 meters per bay
        row_meters = row_distance * 12  # Assume 12 meters per row
        
        # Calculate movement components
        trolley_time = self._calculate_travel_time(bay_meters, self.trolley_speed, self.trolley_acceleration)
        gantry_time = self._calculate_travel_time(row_meters, self.gantry_speed, self.gantry_acceleration)
        
        # Determine if significant gantry movement is needed
        significant_gantry_movement = row_meters > 1.0  # threshold in meters
        
        # Vertical movement (hoisting) - use a simple approximation
        vertical_time = 30.0  # seconds
        
        # Calculate total time with proper sequencing
        horizontal_movement_time = max(trolley_time, gantry_time)
        
        # Lifting can happen while trolley moves (after gantry movement)
        total_time = horizontal_movement_time + vertical_time
        
        # Add fixed time for attaching/detaching
        attach_detach_time = 10.0
        total_time += attach_detach_time
        
        # Add time for current position to source
        current_to_source_time = self._calculate_position_to_source_time(source_indices)
        
        total_time += current_to_source_time
        
        return total_time

    def _calculate_movement_time_from_positions(self, 
                                            source_position: str, 
                                            dest_position: str,
                                            container_type: str = "default") -> float:
        """Calculate movement time using actual terminal positions."""
        # Cache key
        cache_key = (source_position, dest_position, container_type)
        if cache_key in self.movement_time_cache:
            return self.movement_time_cache[cache_key]
            
        # Get positions from terminal
        src_pos = self.terminal.positions[source_position]
        dst_pos = self.terminal.positions[dest_position]
        
        # 1. Gantry movement (movement of entire crane along the rails)
        gantry_distance = abs(src_pos[1] - dst_pos[1])
        
        # 2. Trolley movement (movement of trolley across the crane bridge)
        trolley_distance = abs(src_pos[0] - dst_pos[0])
        
        # Calculate gantry and trolley times
        gantry_time = self._calculate_travel_time(
            gantry_distance, self.gantry_speed, self.gantry_acceleration
        )
        
        trolley_time = self._calculate_travel_time(
            trolley_distance, self.trolley_speed, self.trolley_acceleration
        )
        
        # Vertical movement (simplified)
        vertical_time = 30.0  # seconds
        
        # Total time calculation
        # If significant gantry movement needed, it must happen before trolley
        if gantry_distance > 1.0:
            # Gantry moves first, then trolley and hoisting together
            total_time = gantry_time + max(trolley_time, vertical_time)
        else:
            # No significant gantry movement, trolley and hoisting happen together
            total_time = max(trolley_time, vertical_time)
        
        # Add fixed time for attaching/detaching
        attach_detach_time = 10.0
        total_time += attach_detach_time
        
        # Add time from current position to source
        if hasattr(self.terminal.storage_yard, 'position_to_indices'):
            src_indices = self.terminal.storage_yard.position_to_indices.get(source_position)
            if src_indices:
                current_to_source_time = self._calculate_position_to_source_time(src_indices)
                total_time += current_to_source_time
        
        # Cache the result
        self.movement_time_cache[cache_key] = total_time
        
        return total_time

    def _calculate_position_to_source_time(self, source_indices: Tuple[int, int]) -> float:
        """Calculate time for crane to move from current position to source position."""
        current_bay, current_row = self.current_position
        src_row_idx, src_bay_idx = source_indices
        
        # Calculate bay and row distances
        bay_distance = abs(current_bay - src_bay_idx) * 6  # Assume 6 meters per bay
        row_distance = abs(current_row - src_row_idx) * 12  # Assume 12 meters per row
        
        # Calculate movement times
        bay_time = self._calculate_travel_time(
            bay_distance, self.trolley_speed, self.trolley_acceleration
        )
        
        row_time = self._calculate_travel_time(
            row_distance, self.gantry_speed, self.gantry_acceleration
        )
        
        # Return the maximum (movements can happen in parallel)
        return max(bay_time, row_time)

    def _calculate_travel_time(self, distance: float, max_speed: float, acceleration: float) -> float:
        """Calculate travel time with acceleration and deceleration."""
        # No movement needed
        if distance <= 0:
            return 0.0
        
        # Calculate the distance needed to reach max speed
        accel_distance = 0.5 * max_speed**2 / acceleration
        
        # If we can't reach max speed (distance is too short)
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

    def generate_movement_time_matrix(self, storage_yard: Any) -> torch.Tensor:
        """
        Generate a matrix of movement times between all positions in operational area.
        
        Args:
            storage_yard: TensorStorageYard object
            
        Returns:
            Tensor of movement times [num_rows, num_bays, num_rows, num_bays]
        """
        num_rows, num_bays = storage_yard.num_rows, storage_yard.num_bays
        
        # Initialize movement time matrix with zeros
        time_matrix = torch.zeros((num_rows, num_bays, num_rows, num_bays), device=self.device)
        
        # Limit calculations to operational area (using masked operations could be faster)
        operational_positions = torch.nonzero(self.operational_area_mask)
        
        for pos_idx in range(operational_positions.size(0)):
            src_row, src_bay = operational_positions[pos_idx]
            
            for dst_idx in range(operational_positions.size(0)):
                dst_row, dst_bay = operational_positions[dst_idx]
                
                # Skip same position
                if src_row == dst_row and src_bay == dst_bay:
                    continue
                
                # Calculate movement time
                time = self.calculate_movement_time_tensor(
                    (src_row.item(), src_bay.item()),
                    (dst_row.item(), dst_bay.item())
                )
                
                # Store in matrix
                time_matrix[src_row, src_bay, dst_row, dst_bay] = time
        
        return time_matrix

    def get_valid_move_mask(self, 
                        storage_yard: Any,
                        trucks_in_terminal: Dict[str, Any] = None,
                        trains_in_terminal: Dict[str, Any] = None
                        ) -> torch.Tensor:
        """
        Generate a combined mask of valid moves for all containers.
        
        Args:
            storage_yard: TensorStorageYard object
            trucks_in_terminal: Dictionary of trucks at positions
            trains_in_terminal: Dictionary of trains at positions
            
        Returns:
            Boolean tensor of valid moves [num_rows, num_bays, num_rows, num_bays]
        """
        num_rows, num_bays = storage_yard.num_rows, storage_yard.num_bays
        
        # Initialize with zeros
        valid_moves = torch.zeros((num_rows, num_bays, num_rows, num_bays), 
                                 dtype=torch.bool, device=self.device)
        
        # Generate mask of valid source positions
        valid_sources = storage_yard.generate_extraction_mask() & self.operational_area_mask
        
        # For each valid source, identify valid destinations
        source_indices = torch.nonzero(valid_sources)
        
        for idx in range(source_indices.size(0)):
            src_row, src_bay = source_indices[idx]
            position = storage_yard.indices_to_position(src_row.item(), src_bay.item())
            
            if position:
                # Get container at this position
                container, _ = storage_yard.get_top_container(position)
                
                if container:
                    # Generate valid destinations for this container
                    valid_dests = self.generate_valid_destination_mask(position, container, storage_yard)
                    
                    # Set valid moves in the mask
                    dest_indices = torch.nonzero(valid_dests)
                    for dest_idx in range(dest_indices.size(0)):
                        dst_row, dst_bay = dest_indices[dest_idx]
                        valid_moves[src_row, src_bay, dst_row, dst_bay] = True
        
        return valid_moves

    def get_movement_mask_with_times(self, storage_yard: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate both valid move mask and movement times in one pass.
        
        Args:
            storage_yard: TensorStorageYard object
            
        Returns:
            Tuple of (valid_moves_mask, movement_times)
        """
        num_rows, num_bays = storage_yard.num_rows, storage_yard.num_bays
        
        # Initialize tensors
        valid_moves = torch.zeros((num_rows, num_bays, num_rows, num_bays), 
                                 dtype=torch.bool, device=self.device)
        move_times = torch.zeros((num_rows, num_bays, num_rows, num_bays), 
                                dtype=torch.float32, device=self.device)
        
        # Generate mask of valid source positions
        valid_sources = storage_yard.generate_extraction_mask() & self.operational_area_mask
        
        # For each valid source, identify valid destinations
        source_indices = torch.nonzero(valid_sources)
        
        for idx in range(source_indices.size(0)):
            src_row, src_bay = source_indices[idx]
            position = storage_yard.indices_to_position(src_row.item(), src_bay.item())
            
            if position:
                # Get container at this position
                container, _ = storage_yard.get_top_container(position)
                
                if container:
                    # Generate valid destinations for this container
                    valid_dests = self.generate_valid_destination_mask(position, container, storage_yard)
                    
                    # Set valid moves and calculate times
                    dest_indices = torch.nonzero(valid_dests)
                    for dest_idx in range(dest_indices.size(0)):
                        dst_row, dst_bay = dest_indices[dest_idx]
                        
                        # Mark as valid move
                        valid_moves[src_row, src_bay, dst_row, dst_bay] = True
                        
                        # Calculate movement time
                        time = self.calculate_movement_time_tensor(
                            (src_row.item(), src_bay.item()),
                            (dst_row.item(), dst_bay.item()),
                            container.container_type if hasattr(container, 'container_type') else "default"
                        )
                        
                        # Store in matrix
                        move_times[src_row, src_bay, dst_row, dst_bay] = time
        
        return valid_moves, move_times

    def move_container(self, 
                      source_position: str, 
                      destination_position: str,
                      storage_yard: Any,
                      trucks_in_terminal: Dict[str, Any] = None,
                      trains_in_terminal: Dict[str, Any] = None) -> Tuple[Optional[Any], float]:
        """
        Move a container from source to destination.
        
        Args:
            source_position: Source position string
            destination_position: Destination position string
            storage_yard: TensorStorageYard object
            trucks_in_terminal: Dictionary of trucks at positions
            trains_in_terminal: Dictionary of trains at positions
            
        Returns:
            Tuple of (moved_container, time_taken)
        """
        # Check if both positions are in this crane's operational area
        if not (self._is_position_in_crane_area(source_position, storage_yard) and 
                self._is_position_in_crane_area(destination_position, storage_yard)):
            return None, 0
        
        # Check if there's a container at the source
        container, _ = storage_yard.get_top_container(source_position)
        
        if container is None:
            return None, 0
        
        # Check if the destination is valid for this container
        src_row_idx, src_bay_idx = storage_yard.position_to_indices.get(source_position, (None, None))
        dst_row_idx, dst_bay_idx = storage_yard.position_to_indices.get(destination_position, (None, None))
        
        if src_row_idx is None or dst_row_idx is None:
            return None, 0
            
        # Get valid destinations mask for this container
        valid_dests = self.generate_valid_destination_mask(source_position, container, storage_yard)
        
        # Check if destination is valid
        if not valid_dests[dst_row_idx, dst_bay_idx]:
            return None, 0
        
        # Calculate the time required for the move
        time_taken = self.calculate_movement_time_tensor(
            (src_row_idx, src_bay_idx),
            (dst_row_idx, dst_bay_idx),
            container.container_type if hasattr(container, 'container_type') else "default"
        )
        
        # Remove the container from its source
        removed_container = storage_yard.remove_container(source_position)
        
        if removed_container is None:
            return None, 0
        
        # Place the container at its destination
        success = storage_yard.add_container(destination_position, removed_container)
        
        # Update the crane's position and caches
        if success:
            # Update crane position to destination
            self.current_position = (dst_bay_idx, dst_row_idx)
            
            return removed_container, time_taken
        else:
            # If the placement failed, put the container back
            storage_yard.add_container(source_position, removed_container)
            return None, 0

    def _is_position_in_crane_area(self, position: str, storage_yard: Any) -> bool:
        """Check if a position is within this crane's operational area."""
        if position not in storage_yard.position_to_indices:
            return False
            
        row_idx, bay_idx = storage_yard.position_to_indices[position]
        return self.start_bay <= bay_idx <= self.end_bay

    def to_cpu(self):
        """Move all tensors to CPU for saving."""
        self.operational_area_mask = self.operational_area_mask.cpu()
        self.device = 'cpu'

    def to_device(self, device: str):
        """Move all tensors to specified device."""
        self.device = device
        self.operational_area_mask = self.operational_area_mask.to(device)

    def __str__(self):
        """String representation of the crane."""
        return (f"TensorRMGCrane {self.crane_id}: position={self.current_position}, "
                f"operational_area=[{self.start_bay}-{self.end_bay}], device={self.device}")