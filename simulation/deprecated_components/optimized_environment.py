import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
import os
import pickle
from datetime import datetime, timedelta
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any, Set
import numba

# Import original components for compatibility
from simulation.terminal_layout.CTSimulator import ContainerTerminal
from simulation.terminal_components.Container import Container, ContainerFactory
from simulation.terminal_components.Train import Train
from simulation.terminal_components.Truck import Truck
from simulation.terminal_components.TerminalTruck import TerminalTruck
from simulation.terminal_components.Vehicle_Queue import VehicleQueue

class FastStorageYard:
    """CPU-optimized version of the storage yard."""
    
    def __init__(self, 
                num_rows: int, 
                num_bays: int, 
                max_tier_height: int = 5,
                row_names: List[str] = None,
                special_areas: Dict[str, List[Tuple[str, int, int]]] = None):
        """Initialize optimized storage yard."""
        self.num_rows = num_rows
        self.num_bays = num_bays
        self.max_tier_height = max_tier_height
        
        # Initialize row names if not provided
        if row_names is None:
            self.row_names = [chr(65 + i) for i in range(num_rows)]  # A, B, C, ...
        else:
            self.row_names = row_names[:num_rows]  # Use provided names up to num_rows
        
        # Special areas for different container types
        if special_areas is None:
            # Default special areas
            self.special_areas = {
                'reefer': [('A', 1, 5)],  # Row A, bays 1-5
                'dangerous': [('F', 6, 10)],  # Row F, bays 6-10
                'trailer': [('A', 15, 25)],  # Row A, bays 15-25
                'swap_body': [('A', 30, 40)]  # Row A, bays 30-40
            }
        else:
            self.special_areas = special_areas
        
        # Initialize storage for containers - dictionary for compatibility with existing code
        self.yard = {row: {bay: {} for bay in range(1, num_bays + 1)} for row in self.row_names}
        
        # Create efficient numpy arrays for state representation
        self._create_efficient_arrays()
        
        # Create position mapping for fast lookups
        self.position_to_indices = {}
        self.indices_to_position = {}
        self._build_position_mapping()
        
        # Create pre-computed masks for special areas
        self._create_special_area_masks()

        # Performance tracking
        self.container_access_times = []
        self.state_update_times = []
    
    def _create_efficient_arrays(self):
        """Create optimized numpy arrays for state representation."""
        # Main occupancy array: [row, bay, tier]
        self.occupancy_array = np.zeros((self.num_rows, self.num_bays, self.max_tier_height), 
                                        dtype=bool)
        
        # Container type codes (0=regular, 1=reefer, 2=dangerous, 3=trailer, 4=swap_body)
        self.type_array = np.zeros((self.num_rows, self.num_bays, self.max_tier_height), 
                                  dtype=np.int8)
        
        # Container size codes (0=20ft, 1=30ft, 2=40ft, 3=45ft)
        self.size_array = np.zeros((self.num_rows, self.num_bays, self.max_tier_height), 
                                  dtype=np.int8)
        
        # Priority array (0-100, higher values = higher priority)
        self.priority_array = np.zeros((self.num_rows, self.num_bays, self.max_tier_height), 
                                      dtype=np.int16)
        
        # Current height of each stack (0 = empty)
        self.stack_height_array = np.zeros((self.num_rows, self.num_bays), 
                                          dtype=np.int8)
    
    def _build_position_mapping(self):
        """Build mappings between position strings and array indices."""
        for row_idx, row in enumerate(self.row_names):
            for bay_idx in range(self.num_bays):
                position = f"{row}{bay_idx+1}"
                self.position_to_indices[position] = (row_idx, bay_idx)
                self.indices_to_position[(row_idx, bay_idx)] = position
    
    def _create_special_area_masks(self):
        """Create pre-computed masks for special areas."""
        # Initialize masks for each special area type
        self.special_area_masks = {}
        
        # Create mask for each special area type
        for area_type, areas in self.special_areas.items():
            # Initialize mask with False
            mask = np.zeros((self.num_rows, self.num_bays), dtype=bool)
            
            # Set mask to True for positions in this special area
            for area_row, start_bay, end_bay in areas:
                if area_row in self.row_names:
                    row_idx = self.row_names.index(area_row)
                    # Adjust for 0-based indexing
                    start_bay_idx = max(0, start_bay - 1)
                    end_bay_idx = min(self.num_bays - 1, end_bay - 1)
                    
                    # Set the mask for this area
                    mask[row_idx, start_bay_idx:end_bay_idx+1] = True
            
            # Store the mask
            self.special_area_masks[area_type] = mask
        
        # For regular positions, create a mask where no special areas exist
        regular_mask = np.ones((self.num_rows, self.num_bays), dtype=bool)
        for mask in self.special_area_masks.values():
            regular_mask = regular_mask & ~mask
        
        self.special_area_masks['regular'] = regular_mask
    
    def clear(self):
        """Clear the entire storage yard."""
        # Clear dictionary representation
        for row in self.row_names:
            for bay in range(1, self.num_bays + 1):
                self.yard[row][bay] = {}
        
        # Reset all arrays
        self.occupancy_array.fill(0)
        self.type_array.fill(0)
        self.size_array.fill(0)
        self.priority_array.fill(0)
        self.stack_height_array.fill(0)
    
    def _get_type_code(self, container) -> int:
        """Convert container type to numeric code for efficient storage."""
        if hasattr(container, 'container_type') and hasattr(container, 'goods_type'):
            if container.container_type == "Trailer":
                return 3
            elif container.container_type == "Swap Body":
                return 4
            elif container.goods_type == "Reefer":
                return 1
            elif container.goods_type == "Dangerous":
                return 2
            else:
                return 0  # Regular
        return 0  # Default to regular
    
    def _get_size_code(self, container) -> int:
        """Convert container dimensions to numeric code."""
        if hasattr(container, 'container_type'):
            if container.container_type == "TWEU":  # 20ft
                return 0
            elif container.container_type == "THEU":  # 30ft
                return 1
            elif container.container_type == "FEU":  # 40ft
                return 2
            elif container.container_type == "FFEU":  # 45ft
                return 3
            elif container.container_type == "Trailer" or container.container_type == "Swap Body":
                return 2  # Treat as 40ft
        return 2  # Default to 40ft
    
    def add_container(self, position: str, container: Any, tier: int = None) -> bool:
        """Add a container to the specified position."""
        start_time = time.time()
        
        # Check if position is valid
        if position not in self.position_to_indices:
            return False
        
        # Check if container can be accepted at this position
        if not self._can_accept_container(position, container):
            return False
        
        row_idx, bay_idx = self.position_to_indices[position]
        row, bay = self._position_to_row_bay(position)
        
        # If tier not specified, add to the top
        current_height = self.stack_height_array[row_idx, bay_idx]
        if tier is None:
            tier = current_height + 1
        
        # Check tier bounds
        if tier < 1 or tier > self.max_tier_height:
            return False
        
        # Check if the tier is already occupied
        if tier <= current_height and self.occupancy_array[row_idx, bay_idx, tier-1]:
            return False
        
        # Add to dictionary (for compatibility with existing code)
        self.yard[row][bay][tier] = container
        
        # Update arrays
        self.occupancy_array[row_idx, bay_idx, tier-1] = True
        self.type_array[row_idx, bay_idx, tier-1] = self._get_type_code(container)
        self.size_array[row_idx, bay_idx, tier-1] = self._get_size_code(container)
        
        # Update priority (use raw value)
        priority = getattr(container, 'priority', 50)  # Default to 50
        self.priority_array[row_idx, bay_idx, tier-1] = max(0, min(100, priority))
        
        # Update stack height if this is a new top container
        if tier > current_height:
            self.stack_height_array[row_idx, bay_idx] = tier
        
        # Track performance
        self.container_access_times.append(time.time() - start_time)
        
        return True
    
    def remove_container(self, position: str, tier: int = None) -> Optional[Any]:
        """Remove a container from the specified position."""
        start_time = time.time()
        
        # Check if position is valid
        if position not in self.position_to_indices:
            return None
        
        row_idx, bay_idx = self.position_to_indices[position]
        row, bay = self._position_to_row_bay(position)
        
        # Get current height
        current_height = self.stack_height_array[row_idx, bay_idx]
        
        # If tier not specified, remove from the top
        if tier is None:
            tier = current_height
            if tier == 0:  # No containers in this position
                return None
        
        # Check if there's a container at the specified tier
        if tier not in self.yard[row][bay] or tier > current_height:
            return None
        
        # Check if there are containers above this one
        if tier < current_height:
            # Can't remove container with containers on top
            return None
        
        # Remove from dictionary
        container = self.yard[row][bay].pop(tier)
        
        # Update arrays
        self.occupancy_array[row_idx, bay_idx, tier-1] = False
        self.type_array[row_idx, bay_idx, tier-1] = 0
        self.size_array[row_idx, bay_idx, tier-1] = 0
        self.priority_array[row_idx, bay_idx, tier-1] = 0
        
        # Update stack height
        if tier == current_height:
            # Find the new top container
            new_height = 0
            for t in range(tier-1, 0, -1):
                if self.occupancy_array[row_idx, bay_idx, t-1]:
                    new_height = t
                    break
            self.stack_height_array[row_idx, bay_idx] = new_height
        
        # Track performance
        self.container_access_times.append(time.time() - start_time)
        
        return container
    
    def _can_accept_container(self, position: str, container: Any) -> bool:
        """Check if a container can be added to the specified position."""
        # Check if position is valid
        if position not in self.position_to_indices:
            return False
        
        row_idx, bay_idx = self.position_to_indices[position]
        
        # Check height constraints
        current_height = self.stack_height_array[row_idx, bay_idx]
        if current_height >= self.max_tier_height:
            return False
        
        # Get container type and special area requirements
        type_code = self._get_type_code(container)
        
        # Check special area constraints using masks
        if type_code == 1:  # Reefer
            if not self.special_area_masks['reefer'][row_idx, bay_idx]:
                return False
        elif type_code == 2:  # Dangerous
            if not self.special_area_masks['dangerous'][row_idx, bay_idx]:
                return False
        elif type_code == 3:  # Trailer
            if not self.special_area_masks['trailer'][row_idx, bay_idx]:
                return False
        elif type_code == 4:  # Swap Body
            if not self.special_area_masks['swap_body'][row_idx, bay_idx]:
                return False
        
        # Check stacking compatibility with container below
        if current_height > 0:
            # Get row and bay
            row, bay = self._position_to_row_bay(position)
            container_below = self.yard[row][bay][current_height]
            if not self._can_stack(container, container_below):
                return False
        
        return True
    
    def _can_stack(self, upper_container, lower_container) -> bool:
        """Check if one container can stack on another."""
        if hasattr(upper_container, 'can_stack_with'):
            return upper_container.can_stack_with(lower_container)
        
        # Basic compatibility checks
        if (hasattr(upper_container, 'is_stackable') and not upper_container.is_stackable or
            hasattr(lower_container, 'is_stackable') and not lower_container.is_stackable):
            return False
        
        if (hasattr(upper_container, 'container_type') and 
            upper_container.container_type in ["Trailer", "Swap Body"]):
            return False
        
        if (hasattr(upper_container, 'container_type') and 
            hasattr(lower_container, 'container_type') and
            upper_container.container_type != lower_container.container_type):
            return False
        
        return True
    
    def get_top_container(self, position: str) -> Tuple[Optional[Any], Optional[int]]:
        """Get the top container at the specified position."""
        # Check if position is valid
        if position not in self.position_to_indices:
            return None, None
        
        row_idx, bay_idx = self.position_to_indices[position]
        row, bay = self._position_to_row_bay(position)
        
        # Get current height
        current_height = self.stack_height_array[row_idx, bay_idx]
        if current_height == 0:  # No containers in this position
            return None, None
        
        # Return the container at the highest tier
        return self.yard[row][bay][current_height], current_height
    
    def get_containers_at_position(self, position: str) -> Dict[int, Any]:
        """Get all containers at a specified position."""
        # Check if position is valid
        if position not in self.position_to_indices:
            return {}
        
        row, bay = self._position_to_row_bay(position)
        
        # Return a copy of the containers dictionary
        return dict(self.yard[row][bay])
    
    def get_state_representation(self) -> np.ndarray:
        """
        Get a numpy array representation of the storage yard state.
        Returns a [rows, bays, features] array with efficiently encoded features.
        """
        start_time = time.time()
        
        # Create a state representation with multiple features
        # [occupancy, stack_height/max_height, container_types(5), size(4), priority]
        state = np.zeros((self.num_rows, self.num_bays, 11), dtype=np.float32)
        
        # Feature 0: Occupancy (1 if bay has any containers)
        state[:, :, 0] = (self.stack_height_array > 0).astype(np.float32)
        
        # Feature 1: Normalized stack height
        state[:, :, 1] = self.stack_height_array.astype(np.float32) / self.max_tier_height
        
        # Loop optimization: use vectorized operations for top container features
        for row_idx in range(self.num_rows):
            for bay_idx in range(self.num_bays):
                height = self.stack_height_array[row_idx, bay_idx]
                if height > 0:
                    # Features 2-6: Container types at the top
                    type_code = self.type_array[row_idx, bay_idx, height-1]
                    if 0 <= type_code < 5:
                        state[row_idx, bay_idx, 2 + type_code] = 1.0
                    
                    # Features 7-10: Container sizes at the top
                    size_code = self.size_array[row_idx, bay_idx, height-1]
                    if 0 <= size_code < 4:
                        state[row_idx, bay_idx, 7 + size_code] = 1.0
                    
                    # Feature 11: Priority of top container (normalized to 0-1)
                    state[row_idx, bay_idx, -1] = self.priority_array[row_idx, bay_idx, height-1] / 100.0
        
        # Track performance
        self.state_update_times.append(time.time() - start_time)
        
        return state
    
    def _position_to_row_bay(self, position: str) -> Tuple[str, int]:
        """Convert a position string to (row, bay)."""
        if position not in self.position_to_indices:
            raise ValueError(f"Invalid position: {position}")
        
        row_idx, bay_idx = self.position_to_indices[position]
        row = self.row_names[row_idx]
        bay = bay_idx + 1  # Convert to 1-based indexing
        
        return row, bay
    
    def get_extraction_mask(self) -> np.ndarray:
        """
        Generate a mask of positions where containers can be extracted.
        Only top containers can be extracted.
        """
        # A container can be extracted if it's at the top of a stack
        # This is simply where stack_height > 0
        return self.stack_height_array > 0
    
    def get_valid_placement_mask(self, container: Any) -> np.ndarray:
        """Get a boolean mask of valid placement positions for a container."""
        # Initialize mask with all positions
        valid_mask = np.ones((self.num_rows, self.num_bays), dtype=bool)
        
        # Filter out positions where stack is at max height
        valid_mask = valid_mask & (self.stack_height_array < self.max_tier_height)
        
        # Handle special area constraints
        type_code = self._get_type_code(container)
        
        if type_code == 1:  # Reefer
            valid_mask = valid_mask & self.special_area_masks['reefer']
        elif type_code == 2:  # Dangerous
            valid_mask = valid_mask & self.special_area_masks['dangerous']
        elif type_code == 3:  # Trailer
            valid_mask = valid_mask & self.special_area_masks['trailer']
        elif type_code == 4:  # Swap Body
            valid_mask = valid_mask & self.special_area_masks['swap_body']
        elif type_code == 0:  # Regular
            # Regular containers can't go in special areas
            for area_type in ['reefer', 'dangerous', 'trailer', 'swap_body']:
                valid_mask = valid_mask & ~self.special_area_masks[area_type]
        
        # For each valid position, check stacking compatibility
        for row_idx in range(self.num_rows):
            for bay_idx in range(self.num_bays):
                if valid_mask[row_idx, bay_idx]:
                    current_height = self.stack_height_array[row_idx, bay_idx]
                    if current_height > 0:
                        position = self.indices_to_position.get((row_idx, bay_idx))
                        if position:
                            row, bay = self._position_to_row_bay(position)
                            container_below = self.yard[row][bay][current_height]
                            
                            # Check if stacking is allowed
                            if not self._can_stack(container, container_below):
                                valid_mask[row_idx, bay_idx] = False
        
        return valid_mask
    
    def find_container(self, container_id: str) -> List[str]:
        """Find a container by its ID and return its position(s)."""
        positions = []
        
        # Search through all positions
        for row in self.row_names:
            for bay in range(1, self.num_bays + 1):
                for tier, container in self.yard[row][bay].items():
                    if hasattr(container, 'container_id') and container.container_id == container_id:
                        positions.append(f"{row}{bay}")
                        break
        
        return positions

# OptimizedRMGCrane class
class FastRMGCrane:
    """CPU-optimized version of the RMG crane."""
    
    def __init__(self, 
                 crane_id: str,
                 terminal: Any,
                 start_bay: int,
                 end_bay: int,
                 current_position: Tuple[int, int] = (0, 0)):
        """Initialize optimized RMG crane."""
        self.crane_id = crane_id
        self.terminal = terminal
        self.start_bay = start_bay
        self.end_bay = end_bay
        self.current_position = current_position
        
        # Crane specifications
        self.trolley_speed = 70.0 / 60.0  # m/s (converted from m/min)
        self.hoisting_speed = 28.0 / 60.0  # m/s (converted from m/min)
        self.gantry_speed = 130.0 / 60.0  # m/s (converted from m/min)
        self.trolley_acceleration = 0.3  # m/s²
        self.hoisting_acceleration = 0.2  # m/s²
        self.gantry_acceleration = 0.1  # m/s²
        self.max_height = 20.0  # meters
        self.ground_vehicle_height = 1.5  # meters
        
        # Container heights lookup
        self.container_heights = {
            "TWEU": 2.59,
            "THEU": 2.59,
            "FEU": 2.59,
            "FFEU": 2.59,
            "default": 2.59
        }
        
        # Movement time cache for repeated operations
        self.movement_time_cache = {}
        
        # Previous position for distance calculations
        self.previous_position = current_position
    
    def reset(self, position: Tuple[int, int] = None):
        """Reset the crane to its initial position."""
        if position is None:
            self.current_position = (self.start_bay, 0)
        else:
            self.current_position = position
            
        self.previous_position = self.current_position
        self.movement_time_cache = {}
    
    def move_container(self, 
                      source_position: str, 
                      destination_position: str,
                      storage_yard: Any,
                      trucks_in_terminal: Dict[str, Any] = None,
                      trains_in_terminal: Dict[str, Any] = None) -> Tuple[Optional[Any], float]:
        """Move a container from source to destination."""
        # Check if both positions are in this crane's operational area
        if not (self._is_position_in_crane_area(source_position, storage_yard) and 
                self._is_position_in_crane_area(destination_position, storage_yard)):
            return None, 0
        
        # Check if there's a container at the source
        container, _ = storage_yard.get_top_container(source_position)
        
        if container is None:
            return None, 0
        
        # Get position indices
        src_indices = storage_yard.position_to_indices.get(source_position, (None, None))
        dst_indices = storage_yard.position_to_indices.get(destination_position, (None, None))
        
        if src_indices[0] is None or dst_indices[0] is None:
            return None, 0
        
        # Check if destination is valid using storage yard placement mask
        if isinstance(storage_yard, FastStorageYard):
            # Optimized version uses numpy masks
            valid_mask = storage_yard.get_valid_placement_mask(container)
            row_idx, bay_idx = dst_indices
            if not valid_mask[row_idx, bay_idx]:
                return None, 0
        else:
            # Fallback for compatibility with original storage yard
            if not storage_yard._can_accept_container(destination_position, container):
                return None, 0
        
        # Calculate the time required for the move using the cached version
        time_taken = self._calculate_movement_time(
            src_indices, dst_indices, getattr(container, 'container_type', "default"))
        
        # Remove the container from its source
        removed_container = storage_yard.remove_container(source_position)
        
        if removed_container is None:
            return None, 0
        
        # Place the container at its destination
        success = storage_yard.add_container(destination_position, removed_container)
        
        # Update the crane's position and caches
        if success:
            # Store previous position
            self.previous_position = self.current_position
            # Update to new position
            self.current_position = (dst_indices[1], dst_indices[0])  # (bay, row)
            
            return removed_container, time_taken
        else:
            # If placement failed, put the container back
            storage_yard.add_container(source_position, removed_container)
            return None, 0
    
    def _calculate_movement_time(self, 
                               source_indices: Tuple[int, int], 
                               dest_indices: Tuple[int, int],
                               container_type: str = "default") -> float:
        """Calculate movement time using efficient numpy operations."""
        # Check cache first
        cache_key = (source_indices, dest_indices, container_type)
        if cache_key in self.movement_time_cache:
            return self.movement_time_cache[cache_key]
        
        # Unpack indices
        src_row_idx, src_bay_idx = source_indices
        dst_row_idx, dst_bay_idx = dest_indices
        
        # Calculate position coordinates (if available)
        if hasattr(self.terminal, 'positions'):
            source_position = None
            dest_position = None
            
            # Try to find positions in terminal
            for pos, coords in self.terminal.positions.items():
                if pos.startswith(f"{chr(65+src_row_idx)}{src_bay_idx+1}"):
                    source_position = pos
                if pos.startswith(f"{chr(65+dst_row_idx)}{dst_bay_idx+1}"):
                    dest_position = pos
            
            # If found, use terminal distance calculation
            if source_position and dest_position:
                try:
                    # Calculate based on actual positions
                    src_pos = self.terminal.positions[source_position]
                    dst_pos = self.terminal.positions[dest_position]
                    
                    # Component distances
                    gantry_distance = abs(src_pos[1] - dst_pos[1])
                    trolley_distance = abs(src_pos[0] - dst_pos[0])
                    
                    # Calculate time for each component
                    gantry_time = self._calculate_travel_time(
                        gantry_distance, self.gantry_speed, self.gantry_acceleration)
                    
                    trolley_time = self._calculate_travel_time(
                        trolley_distance, self.trolley_speed, self.trolley_acceleration)
                    
                    # Simplified vertical time
                    vertical_time = 30.0  # seconds (approximate)
                    
                    # Total time calculation
                    if gantry_distance > 1.0:
                        # Gantry moves first, then trolley and vertical
                        time_taken = gantry_time + max(trolley_time, vertical_time)
                    else:
                        # Simultaneous trolley and vertical
                        time_taken = max(trolley_time, vertical_time)
                    
                    # Add handling time
                    time_taken += 10.0
                    
                    # Add time from previous position
                    # This accounts for empty crane movement
                    prev_bay, prev_row = self.previous_position
                    prev_to_source_distance = np.sqrt(
                        ((src_bay_idx - prev_bay) * 6)**2 + 
                        ((src_row_idx - prev_row) * 12)**2
                    )
                    prev_to_source_time = prev_to_source_distance / self.gantry_speed
                    time_taken += prev_to_source_time
                    
                    # Cache result
                    self.movement_time_cache[cache_key] = time_taken
                    return time_taken
                    
                except (KeyError, ValueError):
                    pass
        
        # Fallback to simplified calculation
        # Calculate distances in grid units
        bay_distance = abs(src_bay_idx - dst_bay_idx)
        row_distance = abs(src_row_idx - dst_row_idx)
        
        # Convert to approximate meters
        bay_meters = bay_distance * 6  # Assume 6 meters per bay
        row_meters = row_distance * 12  # Assume 12 meters per row
        
        # Calculate movement times
        trolley_time = self._calculate_travel_time(bay_meters, self.trolley_speed, self.trolley_acceleration)
        gantry_time = self._calculate_travel_time(row_meters, self.gantry_speed, self.gantry_acceleration)
        
        # Simplified vertical time
        vertical_time = 30.0  # seconds (approximate)
        
        # Calculate total time based on sequence constraints
        if row_meters > 1.0:  # Significant gantry movement needed
            horizontal_time = gantry_time + trolley_time
        else:
            horizontal_time = trolley_time
        
        # Add vertical movement and handling time
        total_time = horizontal_time + vertical_time + 10.0  # Add 10s for attachment/detachment
        
        # Add time from previous position (for empty crane movement)
        prev_bay, prev_row = self.previous_position
        prev_to_source_distance = np.sqrt(
            ((src_bay_idx - prev_bay) * 6)**2 + 
            ((src_row_idx - prev_row) * 12)**2
        )
        prev_to_source_time = prev_to_source_distance / max(self.gantry_speed, 1.0)
        total_time += prev_to_source_time
        
        # Cache result
        self.movement_time_cache[cache_key] = total_time
        
        return total_time
    
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
    
    def _is_position_in_crane_area(self, position: str, storage_yard: Any) -> bool:
        """Check if a position is within this crane's operational area."""
        # Find position indices
        if hasattr(storage_yard, 'position_to_indices'):
            indices = storage_yard.position_to_indices.get(position)
            if indices:
                row_idx, bay_idx = indices
                return self.start_bay <= bay_idx <= self.end_bay
        
        # Fallback for other position types
        if position[0].isalpha() and position[1:].isdigit():
            bay = int(position[1:]) - 1  # Convert to 0-based
            return self.start_bay <= bay <= self.end_bay
        
        # Rail and parking positions are considered in operational area
        # if they align with the crane's bay range
        if '_' in position:
            parts = position.split('_')
            if len(parts) == 2 and parts[1].isdigit():
                slot = int(parts[1])
                # Very approximate check - assumes slots roughly align with bays
                return self.start_bay <= slot - 1 <= self.end_bay
        
        return False


class FastActionMaskGenerator:
    """CPU-optimized version for generating action masks."""
    
    def __init__(self, environment):
        """Initialize the mask generator with an environment reference."""
        self.env = environment
        
        # Get dimensions
        self.num_cranes = len(self.env.cranes)
        self.num_positions = len(self.env.position_to_idx)
        
        # Initialize position type arrays for fast lookup
        self._initialize_position_arrays()
        
        # Pre-allocate mask arrays
        self._preallocate_masks()
        
        # Cache for action masks
        self.mask_cache = {}
        self.last_update_time = -1
        
        # Performance tracking
        self.mask_generation_times = []
    
    def _initialize_position_arrays(self):
        """Initialize arrays for position types."""
        # Create position type arrays
        self.storage_positions = np.zeros(self.num_positions, dtype=bool)
        self.rail_positions = np.zeros(self.num_positions, dtype=bool)
        self.truck_positions = np.zeros(self.num_positions, dtype=bool)
        
        # Fill arrays based on position types
        for pos, idx in self.env.position_to_idx.items():
            pos_type = self.env._get_position_type(pos)
            if pos_type == 'storage':
                self.storage_positions[idx] = True
            elif pos_type == 'train':
                self.rail_positions[idx] = True
            elif pos_type == 'truck':
                self.truck_positions[idx] = True
    
    def _preallocate_masks(self):
        """Pre-allocate numpy arrays for masks to avoid repeated allocation."""
        # Crane movement mask: [crane, source, destination]
        self.crane_mask = np.zeros((self.num_cranes, self.num_positions, self.num_positions), 
                                  dtype=np.int8)
        
        # Truck parking mask: [truck, parking spot]
        self.truck_mask = np.zeros((10, len(self.env.parking_spots)), 
                                  dtype=np.int8)
        
        # Terminal truck mask: [terminal truck, source, destination]
        self.terminal_truck_mask = np.zeros((len(self.env.terminal_trucks), 
                                           self.num_positions, self.num_positions), 
                                          dtype=np.int8)
    
    def generate_masks(self) -> Dict[str, np.ndarray]:
        """Generate all action masks efficiently."""
        # Check if we can use cached masks
        current_time = self.env.current_simulation_time
        if current_time == self.last_update_time and self.mask_cache:
            return self.mask_cache
        
        # Track generation time
        start_time = time.time()
        
        # Reset all masks
        self.crane_mask.fill(0)
        self.truck_mask.fill(0)
        self.terminal_truck_mask.fill(0)
        
        # Generate each type of mask
        self._generate_crane_mask()
        self._generate_truck_parking_mask()
        self._generate_terminal_truck_mask()
        
        # Create mask dictionary
        masks = {
            'crane_movement': self.crane_mask,
            'truck_parking': self.truck_mask,
            'terminal_truck': self.terminal_truck_mask
        }
        
        # Update cache
        self.mask_cache = masks
        self.last_update_time = current_time
        
        # Track performance
        self.mask_generation_times.append(time.time() - start_time)
        
        return masks
    
    def _generate_crane_mask(self):
        """Generate crane movement mask using numpy operations."""
        # For each crane
        for i, crane in enumerate(self.env.cranes):
            # Skip if crane is not available yet
            if self.env.current_simulation_time < self.env.crane_available_times[i]:
                continue
            
            # Find valid source positions (where containers can be picked up)
            for src_idx in range(self.num_positions):
                # Get source position
                source_position = self.env.idx_to_position.get(src_idx)
                if source_position is None:
                    continue
                
                # Check if position is in crane's area
                if not crane._is_position_in_crane_area(source_position, self.env.storage_yard):
                    continue
                
                # Check if there's a container at this position
                container = self.env._get_container_at_position(source_position)
                if container is None:
                    continue
                
                # For each potential destination
                for dst_idx in range(self.num_positions):
                    # Get destination position
                    dest_position = self.env.idx_to_position.get(dst_idx)
                    if dest_position is None:
                        continue
                    
                    # Check if destination is in crane's area
                    if not crane._is_position_in_crane_area(dest_position, self.env.storage_yard):
                        continue
                    
                    # Skip rail-to-rail movements
                    if self.rail_positions[src_idx] and self.rail_positions[dst_idx]:
                        continue
                    
                    # Check storage-to-storage premarshalling constraints
                    if self.storage_positions[src_idx] and self.storage_positions[dst_idx]:
                        # Extract bay numbers for distance check
                        if source_position[0].isalpha() and source_position[1:].isdigit() and \
                           dest_position[0].isalpha() and dest_position[1:].isdigit():
                            source_bay = int(source_position[1:]) - 1
                            dest_bay = int(dest_position[1:]) - 1
                            
                            # Check pre-marshalling distance constraint
                            if abs(source_bay - dest_bay) > 5:
                                continue
                    
                    # Check if destination can accept container
                    if self.storage_positions[dst_idx]:
                        if not self.env.storage_yard._can_accept_container(dest_position, container):
                            continue
                    elif self.truck_positions[dst_idx]:
                        # Check if truck exists and can accept container
                        truck = self.env.trucks_in_terminal.get(dest_position)
                        if truck is None or truck.is_full():
                            continue
                        
                        # If it's a pickup truck, check if container matches
                        if hasattr(truck, 'pickup_container_ids') and \
                           hasattr(container, 'container_id') and \
                           truck.pickup_container_ids and \
                           container.container_id not in truck.pickup_container_ids:
                            continue
                    elif self.rail_positions[dst_idx]:
                        # Check if train exists and has matching pickup ID
                        if '_' in dest_position:
                            track_id = f"T{dest_position.split('_')[0][1:]}"
                            train = self.env.trains_in_terminal.get(track_id)
                            
                            if train is None:
                                continue
                            
                            # Check if container is in train's pickup list
                            has_pickup = False
                            if hasattr(container, 'container_id'):
                                for wagon in train.wagons:
                                    if container.container_id in wagon.pickup_container_ids:
                                        has_pickup = True
                                        break
                                
                                if not has_pickup:
                                    continue
                    
                    # This is a valid movement
                    self.crane_mask[i, src_idx, dst_idx] = 1
    
    def _generate_truck_parking_mask(self):
        """Generate truck parking mask using numpy operations."""
        # Get available parking spots
        available_spots = np.ones(len(self.env.parking_spots), dtype=bool)
        for i, spot in enumerate(self.env.parking_spots):
            if spot in self.env.trucks_in_terminal:
                available_spots[i] = False
        
        # Get trucks from queue
        trucks_in_queue = list(self.env.truck_queue.vehicles.queue)
        
        # Mark available spots for each truck
        for truck_idx, truck in enumerate(trucks_in_queue):
            if truck_idx >= 10:  # Only consider first 10 trucks
                break
            
            # Mark all available spots as valid
            for spot_idx in range(len(self.env.parking_spots)):
                if available_spots[spot_idx]:
                    self.truck_mask[truck_idx, spot_idx] = 1
    
    def _generate_terminal_truck_mask(self):
        """Generate terminal truck mask using numpy operations."""
        # For each terminal truck
        for truck_idx, truck in enumerate(self.env.terminal_trucks):
            # Skip if truck is not available yet
            if self.env.current_simulation_time < self.env.terminal_truck_available_times[truck_idx]:
                continue
            
            # Find all trailer and swap body source positions
            for src_idx in range(self.num_positions):
                # Only consider storage positions
                if not self.storage_positions[src_idx]:
                    continue
                
                source_position = self.env.idx_to_position.get(src_idx)
                if source_position is None:
                    continue
                
                # Check if there's a container at this position
                container = self.env._get_container_at_position(source_position)
                if container is None or not hasattr(container, 'container_type'):
                    continue
                
                # Only terminal trucks can move trailers and swap bodies
                if container.container_type not in ["Trailer", "Swap Body"]:
                    continue
                
                # Find valid destinations
                for dst_idx in range(self.num_positions):
                    # Only consider storage positions
                    if not self.storage_positions[dst_idx]:
                        continue
                    
                    dest_position = self.env.idx_to_position.get(dst_idx)
                    if dest_position is None:
                        continue
                    
                    # Check if destination is appropriate for this container type
                    if container.container_type == "Trailer" and not self.env._is_in_special_area(dest_position, 'trailer'):
                        continue
                    elif container.container_type == "Swap Body" and not self.env._is_in_special_area(dest_position, 'swap_body'):
                        continue
                    
                    # Check if destination is empty
                    if self.env.storage_yard.get_top_container(dest_position)[0] is not None:
                        continue
                    
                    # Can't move to same position
                    if src_idx == dst_idx:
                        continue
                    
                    # This is a valid movement
                    self.terminal_truck_mask[truck_idx, src_idx, dst_idx] = 1


class OptimizedTerminalEnvironment(gym.Env):
    """
    Optimized container terminal environment that prioritizes CPU performance.
    Vectorizes operations and uses JIT compilation for critical functions.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, 
                terminal_config_path: str = None,
                terminal_config=None,
                distance_matrix_path: str = None,
                max_simulation_time: float = 86400,  # 24 hours in seconds
                num_cranes: int = 2,
                num_terminal_trucks: int = 3,
                use_existing_env=None):
        """Initialize optimized terminal environment."""
        super(OptimizedTerminalEnvironment, self).__init__()
        
        # Performance logging
        self.log_performance = False
        self.step_times = []
        self.mask_times = []
        self.action_times = []
        
        # If we're given an existing environment, clone from it
        if use_existing_env is not None:
            self._clone_from_environment(use_existing_env)
            return
        
        # Load or create terminal configuration
        from simulation.config import TerminalConfig
        self.config = terminal_config or TerminalConfig(terminal_config_path)
        
        # Create terminal
        self.terminal = self._create_terminal()
        
        # Load distance matrix if available
        if distance_matrix_path and os.path.exists(distance_matrix_path):
            self.terminal.load_distance_matrix(distance_matrix_path)
        
        # Initialize caches and lookup tables
        self._position_type_cache = {}
        
        # Create position mappings
        self._setup_position_mappings()
        
        # Initialize environment components with optimized versions
        self.storage_yard = self._create_optimized_storage_yard()
        self.cranes = self._create_optimized_cranes(num_cranes)
        self.truck_queue = VehicleQueue(vehicle_type="Truck")
        self.train_queue = VehicleQueue(vehicle_type="Train")
        
        # Terminal trucks for handling swap bodies and trailers
        self.terminal_trucks = [TerminalTruck(f"TTR{i+1}") for i in range(num_terminal_trucks)]
        self.terminal_truck_available_times = np.zeros(num_terminal_trucks)
        
        # Track current state
        self.current_simulation_time = 0.0
        self.max_simulation_time = max_simulation_time
        self.crane_available_times = np.zeros(num_cranes)
        
        self.trucks_in_terminal = {}
        self.trains_in_terminal = {}
        
        # Create optimized action mask generator
        self.action_mask_generator = FastActionMaskGenerator(self)
        
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
        
        # Container ID storage
        self.stored_container_ids = []
        
        # Initialize the environment state
        self.reset()
    
    def _clone_from_environment(self, env):
        """Clone state from an existing environment."""
        # Copy configuration
        self.config = env.config
        self.terminal = env.terminal
        
        # Copy position mappings
        self.position_to_idx = env.position_to_idx
        self.idx_to_position = env.idx_to_position
        self._position_type_cache = env._position_type_cache.copy()
        
        # Copy basic attributes
        self.max_simulation_time = env.max_simulation_time
        self.base_simulation_date = env.base_simulation_date
        self.current_simulation_datetime = env.current_simulation_datetime
        self.current_simulation_time = env.current_simulation_time
        self.train_schedule = env.train_schedule
        self.parking_spots = env.parking_spots
        
        # Create optimized components
        # Create storage yard
        self.storage_yard = self._create_optimized_storage_yard()
        
        # Clone storage yard state
        for row in env.storage_yard.row_names:
            for bay in range(1, env.storage_yard.num_bays + 1):
                position = f"{row}{bay}"
                containers = env.storage_yard.get_containers_at_position(position)
                for tier, container in containers.items():
                    self.storage_yard.add_container(position, container, tier)
        
        # Create cranes
        self.cranes = self._create_optimized_cranes(len(env.cranes))
        
        # Clone crane state
        for i, (crane, orig_crane) in enumerate(zip(self.cranes, env.cranes)):
            crane.current_position = orig_crane.current_position
            crane.previous_position = getattr(orig_crane, 'previous_position', crane.current_position)
        
        # Clone queue state
        self.truck_queue = env.truck_queue
        self.train_queue = env.train_queue
        
        # Clone terminal trucks
        self.terminal_trucks = env.terminal_trucks
        self.terminal_truck_available_times = env.terminal_truck_available_times.copy()
        
        # Clone crane availability
        self.crane_available_times = env.crane_available_times.copy()
        
        # Clone vehicle state
        self.trucks_in_terminal = env.trucks_in_terminal.copy()
        self.trains_in_terminal = env.trains_in_terminal.copy()
        
        # Create action mask generator
        self.action_mask_generator = FastActionMaskGenerator(self)
        
        # Setup spaces
        self._setup_spaces()
        
        # Copy stored container IDs
        self.stored_container_ids = env.stored_container_ids.copy() if hasattr(env, 'stored_container_ids') else []
    
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
        """Create an optimized storage yard."""
        # Define special areas for different container types
        special_areas = {
            'reefer': [],
            'dangerous': [],
            'trailer': [],        # Section for trailers
            'swap_body': []       # Section for swap bodies
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
        
        return FastStorageYard(
            num_rows=self.terminal.num_storage_rows,
            num_bays=self.terminal.num_storage_slots_per_row,
            max_tier_height=5,
            row_names=self.terminal.storage_row_names,
            special_areas=special_areas
        )
    
    def _create_optimized_cranes(self, num_cranes):
        """Create optimized RMG cranes with divided operational areas."""
        cranes = []
        bays_per_crane = self.terminal.num_storage_slots_per_row // num_cranes
        
        for i in range(num_cranes):
            start_bay = i * bays_per_crane
            end_bay = (i + 1) * bays_per_crane - 1 if i < num_cranes - 1 else self.terminal.num_storage_slots_per_row - 1
            
            crane = FastRMGCrane(
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
            
        # Add parking spots
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
                shape=(self.terminal.num_storage_rows, self.terminal.num_storage_slots_per_row, 11),
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
        self.crane_available_times = np.zeros(len(self.cranes))
        self.terminal_truck_available_times = np.zeros(len(self.terminal_trucks))
        
        # Clear terminal state
        self.trucks_in_terminal = {}
        self.trains_in_terminal = {}
        self.truck_queue.clear()
        self.train_queue.clear()
        
        # Reset storage yard
        self.storage_yard.clear()
        
        # Reset cranes to initial positions
        for i, crane in enumerate(self.cranes):
            start_bay = i * (self.terminal.num_storage_slots_per_row // len(self.cranes))
            crane.reset(position=(start_bay, 0))
        
        # Reset terminal trucks
        for truck in self.terminal_trucks:
            truck.containers = []
        
        # Initialize with some random containers in the storage yard
        self._initialize_storage_yard()
        
        # Schedule trains and trucks
        self._schedule_trains()
        self._schedule_trucks_for_existing_containers()
        
        # Get initial observation
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def create_wait_action(self):
        """Create a properly formatted wait action with valid indices."""
        # Use the first available indices to ensure they're valid
        first_crane_idx = 0
        first_position_idx = 0  # First valid position index
        
        # Ensure we have valid position indices
        if self.idx_to_position:
            first_position_idx = min(self.idx_to_position.keys())
        
        # Create the wait action
        wait_action = {
            'action_type': 0,  # Crane movement (arbitrary, but must be valid)
            'crane_movement': np.array([first_crane_idx, first_position_idx, first_position_idx], dtype=np.int32),
            'truck_parking': np.array([0, 0], dtype=np.int32),
            'terminal_truck': np.array([0, first_position_idx, first_position_idx], dtype=np.int32)
        }
        
        return wait_action
    
    def step(self, action):
        """
        Take a step in the environment with optimized performance.
        
        Args:
            action: Dictionary with action type and details
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Track overall step time
        step_start_time = time.time()
        
        # Get current observation with action masks
        mask_start_time = time.time()
        current_obs = self._get_observation()
        mask_time = time.time() - mask_start_time
        
        # Determine action type
        action_type = action['action_type']
        
        # Detect if this is a wait action (no movement)
        is_wait_action = False
        if action_type == 0:  # Crane movement
            crane_movement = action['crane_movement']
            if np.array_equal(crane_movement, np.array([0, 0, 0])):
                is_wait_action = True
        
        # Track action execution time
        action_start_time = time.time()
        
        # Handle wait action specially - advance time without actual movement
        if is_wait_action:
            # Small time advancement (1 minute)
            time_advance = 60
            self.current_simulation_time += time_advance
            self.current_simulation_datetime += timedelta(seconds=time_advance)
            
            # Process vehicle arrivals and departures
            self._process_vehicle_arrivals(time_advance)
            self._process_vehicle_departures()
            
            # Return updated observation with no reward
            observation = self._get_observation()
            reward = 0
            terminated = self.current_simulation_time >= self.max_simulation_time
            truncated = False
            info = {"action": "wait", "time_advanced": time_advance}
            
            # Track performance
            action_time = time.time() - action_start_time
            step_time = time.time() - step_start_time
            
            if self.log_performance:
                self.mask_times.append(mask_time)
                self.action_times.append(action_time)
                self.step_times.append(step_time)
            
            return observation, reward, terminated, truncated, info
        
        # Execute the action based on type
        if action_type == 0:  # Crane movement
            crane_idx, source_idx, destination_idx = action['crane_movement']
            
            # Check if the crane movement action is valid according to the mask
            crane_mask = current_obs['action_mask']['crane_movement']
            valid = crane_mask[crane_idx, source_idx, destination_idx] == 1
            
            if not valid:
                # Try to find a valid action instead
                valid_actions = np.argwhere(crane_mask == 1)
                
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
                    
                    # Track performance
                    action_time = time.time() - action_start_time
                    step_time = time.time() - step_start_time
                    
                    if self.log_performance:
                        self.mask_times.append(mask_time)
                        self.action_times.append(action_time)
                        self.step_times.append(step_time)
                    
                    return observation, reward, terminated, truncated, info
            
            # Execute crane movement action
            result = self._execute_crane_movement(int(crane_idx), int(source_idx), int(destination_idx))
        
        elif action_type == 1:  # Truck parking
            truck_idx, parking_spot_idx = action['truck_parking']
            
            # Check if the truck parking action is valid
            truck_mask = current_obs['action_mask']['truck_parking']
            valid = truck_mask[truck_idx, parking_spot_idx] == 1
            
            if not valid:
                # Try to find a valid action instead
                valid_actions = np.argwhere(truck_mask == 1)
                
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
                    
                    # Track performance
                    action_time = time.time() - action_start_time
                    step_time = time.time() - step_start_time
                    
                    if self.log_performance:
                        self.mask_times.append(mask_time)
                        self.action_times.append(action_time)
                        self.step_times.append(step_time)
                    
                    return observation, reward, terminated, truncated, info
            
            # Execute truck parking action
            result = self._execute_truck_parking(int(truck_idx), int(parking_spot_idx))
        
        elif action_type == 2:  # Terminal truck action
            truck_idx, source_idx, destination_idx = action['terminal_truck']
            
            # Check if the terminal truck action is valid
            terminal_mask = current_obs['action_mask']['terminal_truck']
            valid = terminal_mask[truck_idx, source_idx, destination_idx] == 1
            
            if not valid:
                # Try to find a valid action instead
                valid_actions = np.argwhere(terminal_mask == 1)
                
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
                    
                    # Track performance
                    action_time = time.time() - action_start_time
                    step_time = time.time() - step_start_time
                    
                    if self.log_performance:
                        self.mask_times.append(mask_time)
                        self.action_times.append(action_time)
                        self.step_times.append(step_time)
                    
                    return observation, reward, terminated, truncated, info
            
            # Execute terminal truck action
            result = self._execute_terminal_truck_movement(int(truck_idx), int(source_idx), int(destination_idx))
        
        # Record performance metrics
        action_time = time.time() - action_start_time
        step_time = time.time() - step_start_time
        
        if self.log_performance:
            self.mask_times.append(mask_time)
            self.action_times.append(action_time)
            self.step_times.append(step_time)
        
        # Return the step result
        return result
    
    def _execute_crane_movement(self, crane_idx, source_idx, destination_idx):
        """Execute a crane movement action using optimized functions."""
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
        if not np.any(self.crane_available_times <= self.current_simulation_time):
            # All cranes busy, advance to earliest available
            next_available_time = np.min(self.crane_available_times)
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
            "current_time": self.current_simulation_time
        }
        
        return observation, reward, terminated, truncated, info
    
    def _execute_truck_parking(self, truck_idx, parking_spot_idx):
        """Execute a truck parking action using optimized functions."""
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
            "current_time": self.current_simulation_time
        }
        
        return observation, reward, terminated, truncated, info
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
    def _execute_terminal_truck_movement(self, truck_idx, source_idx, destination_idx):
        """Execute a terminal truck movement action."""
        source_position = self.idx_to_position[source_idx]
        destination_position = self.idx_to_position[destination_idx]
        
        # Check if the selected terminal truck is available
        if self.current_simulation_time < self.terminal_truck_available_times[truck_idx]:
            # Truck is not available yet - skip to when it becomes available
            time_advanced = self.terminal_truck_available_times[truck_idx] - self.current_simulation_time
            self.current_simulation_time = self.terminal_truck_available_times[truck_idx]
            
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
        
        # Remove container from source position
        if self._is_storage_position(source_position):
            removed_container = self.storage_yard.remove_container(source_position)
        else:
            removed_container = None  # Not implemented for other source types
        
        # Place container at destination (only storage positions supported)
        success = False
        if removed_container is not None and self._is_storage_position(destination_position):
            success = self.storage_yard.add_container(destination_position, removed_container)
        
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
        
        # Update terminal truck availability time
        self.terminal_truck_available_times[truck_idx] = self.current_simulation_time + time_taken
        
        # Check if any trucks are still available
        if not any(t <= self.current_simulation_time for t in self.terminal_truck_available_times):
            # All terminal trucks busy, advance to earliest available
            next_available_time = min(self.terminal_truck_available_times)
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
            "current_time": self.current_simulation_time
        }
        
        return observation, reward, terminated, truncated, info
    
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
                # Check if this is a valid area for trailer/swap body
                if container.container_type == "Trailer" and not self._is_in_special_area(position, 'trailer'):
                    container = ContainerFactory.create_random(config=self.config)
                    if container.container_type in ["Trailer", "Swap Body"]:
                        continue
                elif container.container_type == "Swap Body" and not self._is_in_special_area(position, 'swap_body'):
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
    
    def _process_time_advancement(self, time_advanced):
        """Process events that occur during time advancement."""
        # Update current simulation datetime
        self.current_simulation_datetime += timedelta(seconds=time_advanced)
        
        # Process vehicle arrivals based on time advancement
        self._process_vehicle_arrivals(time_advanced)
        
        # Process vehicle departures
        self._process_vehicle_departures()
        
        # Signal for mask regeneration
        self.action_mask_generator.last_update_time = -1  # Force mask regeneration
    
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
        """Get current observation efficiently with pre-allocated arrays."""
        # Create crane positions array
        crane_positions = np.array([crane.current_position for crane in self.cranes], dtype=np.float32)
        
        # Create time arrays
        crane_available_times = np.array(self.crane_available_times, dtype=np.float32)
        terminal_truck_available_times = np.array(self.terminal_truck_available_times, dtype=np.float32)
        current_time = np.array([self.current_simulation_time], dtype=np.float32)
        
        # Get yard state efficiently
        yard_state = self.storage_yard.get_state_representation()
        
        # Create parking status array
        parking_status = np.zeros(len(self.parking_spots), dtype=np.int32)
        for i, spot in enumerate(self.parking_spots):
            if spot in self.trucks_in_terminal:
                parking_status[i] = 1
        
        # Create rail status array
        rail_status = np.zeros((len(self.terminal.track_names), self.terminal.num_railslots_per_track), 
                              dtype=np.int32)
        for i, track in enumerate(self.terminal.track_names):
            if track in self.trains_in_terminal:
                rail_status[i, :] = 1
        
        # Create queue sizes array
        queue_sizes = np.array([self.truck_queue.size(), self.train_queue.size()], dtype=np.int32)
        
        # Get action masks using optimized generator
        action_mask = self.action_mask_generator.generate_masks()
        
        # Return combined observation
        return {
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
    
    def _calculate_reward(self, container, source_position, destination_position, time_taken):
        """Calculate the reward for moving a container."""
        reward = 0.0  # Base reward
        
        # Get source and destination types
        source_type = self._get_position_type(source_position)
        dest_type = self._get_position_type(destination_position)
        
        # Calculate distance for container movement (source to destination)
        source_to_dest_distance = 0.0
        if hasattr(self.terminal, 'get_distance'):
            try:
                source_to_dest_distance = self.terminal.get_distance(source_position, destination_position)
            except:
                # If distance calculation fails, estimate based on time
                source_to_dest_distance = time_taken * 0.4  # Rough estimate
        
        # Get crane that performed the operation
        crane_idx = None
        for i, crane in enumerate(self.cranes):
            if crane._is_position_in_crane_area(source_position, self.storage_yard) and crane._is_position_in_crane_area(destination_position, self.storage_yard):
                crane_idx = i
                break
        
        # Calculate crane position to source distance (for empty movement penalty)
        crane_to_source_distance = 0.0
        crane_to_source_time = 0.0
        if crane_idx is not None:
            crane = self.cranes[crane_idx]
            prev_position = crane.previous_position  # Position before this move
            
            # Calculate approximate distance from previous position to source
            if hasattr(crane, '_calculate_position_to_source_time'):
                # Extract row and bay from source position
                if hasattr(self.storage_yard, 'position_to_indices') and source_position in self.storage_yard.position_to_indices:
                    src_indices = self.storage_yard.position_to_indices[source_position]
                    crane_to_source_time = getattr(crane, '_calculate_position_to_source_time', lambda x: 0)(src_indices)
                    # Rough distance estimate based on time and speed
                    crane_to_source_distance = crane_to_source_time * crane.gantry_speed * 0.5
        
        # Empty crane movement penalty
        if container is None:
            empty_move_penalty = -5.0
            distance_time_penalty = -0.05 * source_to_dest_distance - time_taken / 60.0
            return empty_move_penalty + distance_time_penalty
        
        # Add penalty for empty crane movement from previous position to source
        empty_movement_penalty = 0.0
        if crane_to_source_distance > 0:
            # Apply smaller penalty for this implicit empty movement
            empty_movement_penalty = -0.02 * crane_to_source_distance - crane_to_source_time / 120.0
        
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
        
        # DISTANCE AND TIME PENALTY for the actual container movement
        source_to_dest_penalty = -0.02 * source_to_dest_distance  # -0.02 per meter
        time_penalty = -min(time_taken / 120, 1.0)  # Cap at -1 for moves over 2 minutes
        
        # Combine penalties for both empty movement and container movement
        total_distance_time_penalty = source_to_dest_penalty + time_penalty + empty_movement_penalty
        reward += total_distance_time_penalty
        
        # Special bonus for moving swap bodies and trailers
        if container and hasattr(container, 'container_type') and container.container_type in ["Trailer", "Swap Body"]:
            # Check if moved to appropriate area
            if dest_type == 'storage':
                if container.container_type == "Trailer" and self._is_in_special_area(destination_position, 'trailer'):
                    reward += 2.0  # Bonus for placing trailer in correct area
                elif container.container_type == "Swap Body" and self._is_in_special_area(destination_position, 'swap_body'):
                    reward += 2.0  # Bonus for placing swap body in correct area
            
            # Bonus for handling these special containers
            reward += 1.0
            
        return reward
    
    def _calculate_truck_parking_reward(self, truck, parking_spot):
        """Calculate reward for parking a truck."""
        # Base reward for successful truck parking
        reward = 1.0
        
        # Check if truck is empty (pickup truck)
        if hasattr(truck, 'is_pickup_truck') and truck.is_pickup_truck and truck.pickup_container_ids:
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
        elif hasattr(truck, 'is_pickup_truck') and not truck.is_pickup_truck and truck.containers:
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
    
    def _is_in_special_area(self, position, area_type):
        """Check if a position is in a special area like trailer or swap body section."""
        if not self._is_storage_position(position):
            return False
        
        if hasattr(self.storage_yard, 'special_area_masks'):
            # Use optimized arrays if available
            row_idx, bay_idx = self.storage_yard.position_to_indices.get(position, (None, None))
            if row_idx is not None and bay_idx is not None:
                if area_type in self.storage_yard.special_area_masks:
                    return self.storage_yard.special_area_masks[area_type][row_idx, bay_idx]
        
        # Fallback to checking configuration directly
        if position[0].isalpha() and position[1:].isdigit():
            row = position[0]
            bay = int(position[1:])
            
            for area_row, start_bay, end_bay in self.storage_yard.special_areas.get(area_type, []):
                if row == area_row and start_bay <= bay <= end_bay:
                    return True
        
        return False
    
    def _get_container_at_position(self, position):
        """Helper to get container at a position."""
        # Use cached position type if available
        position_type = self._position_type_cache.get(position)
        
        if position_type == 'storage':
            # Get top container from storage
            container, _ = self.storage_yard.get_top_container(position)
            return container
        elif position_type == 'truck':
            # Get container from truck
            truck = self.trucks_in_terminal.get(position)
            if truck and hasattr(truck, 'containers') and truck.containers:
                return truck.containers[0]  # Return the first container
            return None
        elif position_type == 'train':
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
        """Determine the type of a position (train, truck, storage) with caching."""
        if position in self._position_type_cache:
            return self._position_type_cache[position]
        else:
            pos_type = self._get_position_type_direct(position)
            self._position_type_cache[position] = pos_type
            return pos_type
    
    def _get_position_type_direct(self, position):
        """Directly determine position type without caching."""
        if position.startswith('t') and '_' in position:
            return 'train'
        elif position.startswith('p_'):
            return 'truck'
        else:
            return 'storage'
    
    def _is_storage_position(self, position):
        """Check if a position is in the storage yard."""
        return self._get_position_type(position) == 'storage'
    
    def _is_truck_position(self, position):
        """Check if a position is a truck parking spot."""
        return self._get_position_type(position) == 'truck'
    
    def _is_rail_position(self, position):
        """Check if a position is a rail slot."""
        return self._get_position_type(position) == 'train'
    
    def set_simplified_rendering(self, simplified=True):
        """Set simplified rendering mode for faster training."""
        self.simplified_rendering = simplified
    
    def render(self, mode='human'):
        """Render the terminal environment with optimization for training."""
        if hasattr(self, 'simplified_rendering') and self.simplified_rendering and mode == 'human':
            # During training, don't actually render to save time
            # Just return a dummy figure object
            from matplotlib.figure import Figure
            return Figure()
        
        # Regular rendering for human viewing or rgb_array mode
        if mode == 'human':
            # Render the terminal for human viewing
            fig, ax = self.terminal.visualize(figsize=(15, 10), show_labels=True)
            
            # Add key simulation information
            title = f"Terminal Simulation - Time: {self.current_simulation_time:.1f}s"
            ax.set_title(title, fontsize=16)
            
            import matplotlib.pyplot as plt
            plt.close()  # Close the figure to avoid memory issues
            return fig
        
        elif mode == 'rgb_array':
            # Return a numpy array representation of the rendering
            fig = self.render(mode='human')
            
            # Convert figure to RGB array
            fig.canvas.draw()
            img = np.array(fig.canvas.renderer.buffer_rgba())
            import matplotlib.pyplot as plt
            plt.close(fig)
            
            return img
    
    def close(self):
        """Clean up environment resources."""
        import matplotlib.pyplot as plt
        plt.close('all')
    
    def print_performance_stats(self):
        """Print performance statistics of the environment."""
        if not self.log_performance:
            print("Performance logging is disabled. Set env.log_performance = True to enable.")
            return
        
        print("\n=== Performance Statistics ===")
        
        # Environment step times
        step_times = np.array(self.step_times)
        print(f"Step time: {step_times.mean()*1000:.2f}ms avg, {step_times.min()*1000:.2f}ms min, {step_times.max()*1000:.2f}ms max")
        
        # Action mask generation times
        if hasattr(self.action_mask_generator, 'mask_generation_times'):
            mask_times = np.array(self.action_mask_generator.mask_generation_times)
            print(f"Mask generation: {mask_times.mean()*1000:.2f}ms avg, {mask_times.min()*1000:.2f}ms min, {mask_times.max()*1000:.2f}ms max")
        
        # Action execution times
        action_times = np.array(self.action_times)
        print(f"Action execution: {action_times.mean()*1000:.2f}ms avg, {action_times.min()*1000:.2f}ms min, {action_times.max()*1000:.2f}ms max")
        
        # Storage yard operations if available
        if hasattr(self.storage_yard, 'container_access_times') and self.storage_yard.container_access_times:
            container_times = np.array(self.storage_yard.container_access_times)
            print(f"Container access: {container_times.mean()*1000:.2f}ms avg, {container_times.max()*1000:.2f}ms max")


# Implementation of training wrapper for easy switching between environments
class TerminalTrainingWrapper:
    """
    Wrapper that provides a consistent interface for different terminal environment implementations.
    Allows easy switching between the original and optimized environment.
    """
    
    def __init__(self, use_optimized=True, terminal_config_path=None, num_cranes=2, max_simulation_time=86400, clone_from=None):
        """
        Initialize the wrapper with either original or optimized environment.
        
        Args:
            use_optimized: Whether to use the optimized environment implementation
            terminal_config_path: Path to terminal configuration file
            num_cranes: Number of cranes in the terminal
            max_simulation_time: Maximum simulation time in seconds
            clone_from: Existing environment to clone from (optional)
        """
        self.use_optimized = use_optimized
        
        if use_optimized:
            if clone_from is not None:
                # Clone from existing environment
                if hasattr(clone_from, 'env'):
                    base_env = clone_from.env
                else:
                    base_env = clone_from
                    
                # Create optimized environment from existing one
                self.env = OptimizedTerminalEnvironment(
                    use_existing_env=base_env
                )
            else:
                # Create new optimized environment
                self.env = OptimizedTerminalEnvironment(
                    terminal_config_path=terminal_config_path,
                    num_cranes=num_cranes,
                    max_simulation_time=max_simulation_time
                )
                
            # Enable simplified rendering for faster training
            self.env.set_simplified_rendering(True)
            
            # Enable performance logging
            self.env.log_performance = True
        else:
            # Import the original environment implementation
            from simulation.deprecated_components.EnhancedTerminalEnvironment import EnhancedTerminalEnvironment
            
            if clone_from is not None:
                # Cloning not directly supported by original implementation
                # Just create a new instance
                self.env = EnhancedTerminalEnvironment(
                    terminal_config_path=terminal_config_path,
                    num_cranes=num_cranes,
                    max_simulation_time=max_simulation_time
                )
            else:
                # Create new original environment
                self.env = EnhancedTerminalEnvironment(
                    terminal_config_path=terminal_config_path,
                    num_cranes=num_cranes,
                    max_simulation_time=max_simulation_time
                )
        
        # Create properties that mirror the environment interface
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    
    def step(self, action):
        """Take a step in the environment."""
        return self.env.step(action)
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        return self.env.reset(seed=seed, options=options)
    
    def render(self, mode='human'):
        """Render the environment."""
        return self.env.render(mode=mode)
    
    def close(self):
        """Close the environment."""
        self.env.close()
    
    def create_wait_action(self):
        """Create a wait action compatible with the environment."""
        return self.env.create_wait_action()
    
    def print_performance_stats(self):
        """Print performance statistics if available."""
        if hasattr(self.env, 'print_performance_stats'):
            self.env.print_performance_stats()
        else:
            print("Performance statistics not available for the original environment.")


# Performance comparison function
def compare_environment_performance(episode_count=3, steps_per_episode=100):
    """
    Compare performance between original and optimized environments.
    
    Args:
        episode_count: Number of episodes to run
        steps_per_episode: Number of steps per episode
    
    Returns:
        Dictionary with performance metrics
    """
    import time
    
    results = {
        'original': {'reset_time': [], 'step_time': [], 'total_time': []},
        'optimized': {'reset_time': [], 'step_time': [], 'total_time': []}
    }
    
    # Test original environment
    from simulation.deprecated_components.EnhancedTerminalEnvironment import EnhancedTerminalEnvironment
    print("Testing original environment...")
    
    for episode in range(episode_count):
        # Create environment
        start_time = time.time()
        env = EnhancedTerminalEnvironment(num_cranes=2)
        obs, _ = env.reset()
        reset_time = time.time() - start_time
        results['original']['reset_time'].append(reset_time)
        
        # Run steps
        episode_start_time = time.time()
        step_times = []
        
        for step in range(steps_per_episode):
            # Use wait action for consistent comparison
            wait_action = env.create_wait_action()
            
            step_start_time = time.time()
            obs, reward, terminated, truncated, info = env.step(wait_action)
            step_time = time.time() - step_start_time
            step_times.append(step_time)
            
            if terminated or truncated:
                break
        
        avg_step_time = np.mean(step_times)
        results['original']['step_time'].append(avg_step_time)
        
        episode_time = time.time() - episode_start_time
        results['original']['total_time'].append(episode_time)
        
        env.close()
        print(f"  Episode {episode+1}: Reset: {reset_time:.3f}s, Avg Step: {avg_step_time:.3f}s, Total: {episode_time:.3f}s")
    
    # Test optimized environment
    print("\nTesting optimized environment...")
    
    for episode in range(episode_count):
        # Create environment
        start_time = time.time()
        env = OptimizedTerminalEnvironment(num_cranes=2)
        obs, _ = env.reset()
        reset_time = time.time() - start_time
        results['optimized']['reset_time'].append(reset_time)
        
        # Run steps
        episode_start_time = time.time()
        step_times = []
        
        for step in range(steps_per_episode):
            # Use wait action for consistent comparison
            wait_action = env.create_wait_action()
            
            step_start_time = time.time()
            obs, reward, terminated, truncated, info = env.step(wait_action)
            step_time = time.time() - step_start_time
            step_times.append(step_time)
            
            if terminated or truncated:
                break
        
        avg_step_time = np.mean(step_times)
        results['optimized']['step_time'].append(avg_step_time)
        
        episode_time = time.time() - episode_start_time
        results['optimized']['total_time'].append(episode_time)
        
        env.close()
        print(f"  Episode {episode+1}: Reset: {reset_time:.3f}s, Avg Step: {avg_step_time:.3f}s, Total: {episode_time:.3f}s")
    
    # Calculate summary statistics
    for env_type in ['original', 'optimized']:
        print(f"\n{env_type.title()} Environment Summary:")
        avg_reset = np.mean(results[env_type]['reset_time'])
        avg_step = np.mean(results[env_type]['step_time'])
        avg_total = np.mean(results[env_type]['total_time'])
        
        print(f"  Avg Reset Time: {avg_reset:.3f}s")
        print(f"  Avg Step Time: {avg_step*1000:.2f}ms")
        print(f"  Avg Episode Time: {avg_total:.3f}s")
    
    # Calculate speedup factors
    reset_speedup = np.mean(results['original']['reset_time']) / np.mean(results['optimized']['reset_time'])
    step_speedup = np.mean(results['original']['step_time']) / np.mean(results['optimized']['step_time'])
    total_speedup = np.mean(results['original']['total_time']) / np.mean(results['optimized']['total_time'])
    
    print("\nSpeedup Factors:")
    print(f"  Reset: {reset_speedup:.2f}x")
    print(f"  Step: {step_speedup:.2f}x")
    print(f"  Total: {total_speedup:.2f}x")
    
    return results


# Example usage
if __name__ == "__main__":
    # Compare environments
    compare_environment_performance()
    
    # Create training wrapper with optimized environment
    env = TerminalTrainingWrapper(use_optimized=True)
    
    # Run a few episodes for demonstration
    for episode in range(3):
        obs, _ = env.reset()
        episode_reward = 0
        
        for step in range(100):
            # Use wait action for demonstration
            wait_action = env.create_wait_action()
            
            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(wait_action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode+1}: Reward = {episode_reward}, Steps = {step+1}")
    
    # Print performance statistics
    env.print_performance_stats()
    
    # Close environment
    env.close()